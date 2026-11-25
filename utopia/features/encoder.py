"""The UNI encoder and the PCA projection, shared by every feature extractor.

Each extractor runs the same ViT-L/16 checkpoint over 224-pixel windows and keeps two token
streams: the class token ("global") and the 16-pixel patch tokens, one per superpixel ("local").
The PCA is exact and fitted jointly over all samples, so every sample's features share one basis.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..config import Config

#: Side of the encoder's input window, in pixels.
PATCH = 224
#: Distance between neighbouring windows in the fast extractor, in pixels; half of
#: :data:`PATCH`, which is what lets fusion split windows into parity groups.
STRIDE = 112
#: Side of one ViT token, in pixels; equals one superpixel.
TOKEN = 16
#: Tokens along one side of a window.
GRID = PATCH // TOKEN
#: Index of the token at the centre of a window, which is the only local token
#: the slow and very_slow extractors keep.
CENTRE_TOKEN = (GRID // 2) * GRID + GRID // 2
#: Width of one raw token stream.
RAW_DIM = 1024

_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def normalise(tile: np.ndarray) -> torch.Tensor:
    """Turn one ``(224, 224, 3)`` uint8 window into a normalised CHW tensor."""
    tensor = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float().div_(255.0)
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def load_encoder(model_dir: str, device: torch.device) -> torch.nn.Module:
    """Build the UNI ViT-L/16 encoder and load ``pytorch_model.bin`` from ``model_dir``.

    The model is returned in eval mode and outputs all 197 tokens per window.
    """
    import timm

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=PATCH,
        patch_size=TOKEN,
        init_values=1e-5,
        num_classes=0,
        global_pool="",
    )
    state = torch.load(Path(model_dir) / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@torch.inference_mode()
def encode(model: torch.nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode windows into ``(B, 1024)`` class tokens and ``(B, 196, 1024)`` patch tokens."""
    tokens = model(batch)
    if tokens.shape[1] != GRID * GRID + 1:
        raise RuntimeError(
            f"expected {GRID * GRID + 1} tokens per window, got {tokens.shape[1]}; "
            "the checkpoint does not match vit_large_patch16_224"
        )
    return tokens[:, 0].float(), tokens[:, 1:].float()


def encode_on_device(
    model: torch.nn.Module, batch: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a batch of windows to ``device`` and encode it, in half precision on a GPU."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        return encode(model, batch.to(device, non_blocking=True))


class CropDataset(Dataset):
    """Yields one normalised 224x224 crop per position with its index, in ``positions`` order.

    ``crop(image, row, column)`` cuts the uint8 window that belongs to one position.
    """

    def __init__(
        self,
        image: np.ndarray,
        positions: np.ndarray,
        crop: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> None:
        """Keep the image, the ``(N, 2)`` positions and the crop function."""
        self.image = image
        self.positions = positions
        self.crop = crop

    def __len__(self) -> int:
        """Number of positions."""
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """The normalised crop of position ``index``, and ``index`` itself."""
        row, column = self.positions[index]
        return normalise(self.crop(self.image, int(row), int(column))), index


def make_loader(dataset: Dataset, config: Config, indices: np.ndarray | None = None) -> DataLoader:
    """Wrap a dataset, or the subset of it at ``indices``, in an unshuffled loader."""
    if indices is not None:
        dataset = Subset(dataset, indices.tolist())
    return DataLoader(
        dataset,
        batch_size=config.features.batch_size,
        shuffle=False,
        num_workers=config.features.num_workers,
        pin_memory=True,
    )


class Projection:
    """A fitted PCA applied on the GPU to streaming token embeddings."""

    def __init__(self, mean: np.ndarray, components: np.ndarray, device: torch.device) -> None:
        """Move the ``(1024,)`` mean and the ``(dim, 1024)`` components to ``device``."""
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.components = torch.as_tensor(components.T, dtype=torch.float32, device=device)
        self.dim = components.shape[0]

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Centre and project ``(..., 1024)`` embeddings onto the principal axes."""
        return (values - self.mean) @ self.components


class Moments:
    """Running count, mean and covariance of a stream of rows, kept on the GPU.

    Memory is ``dim x dim`` whatever the number of rows, so a PCA can be fitted on a whole slide.
    """

    def __init__(self, dim: int, device: torch.device) -> None:
        """Start with no rows."""
        self.count = 0
        self.shift: torch.Tensor | None = None
        self.total = torch.zeros(dim, dtype=torch.float64, device=device)
        self.outer = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def update(self, rows: torch.Tensor) -> None:
        """Add ``(n, dim)`` rows."""
        rows = rows.float()
        # Shift by the first batch's mean so the float32 products lose nothing to cancellation.
        if self.shift is None:
            self.shift = rows.mean(dim=0)
        shifted = rows - self.shift
        self.count += len(rows)
        self.total += shifted.sum(dim=0).double()
        self.outer += (shifted.T @ shifted).double()

    def mean_and_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        """The rows' mean and their population covariance, as float64 arrays."""
        if self.count == 0:
            raise ValueError("no rows to fit a PCA on")
        centre = self.total / self.count
        covariance = self.outer / self.count - torch.outer(centre, centre)
        return (self.shift.double() + centre).cpu().numpy(), covariance.cpu().numpy()


class PcaFit:
    """Accumulates the rows both PCAs are fitted on, over any number of samples.

    The global PCA sees each window's class token; the local PCA sees ``tokens_per_window``
    randomly drawn patch tokens per window, or only the centre token when that is None.
    """

    def __init__(
        self, device: torch.device, tokens_per_window: int | None, random_state: int
    ) -> None:
        """Start an empty fit whose token draws are seeded with ``random_state``."""
        if tokens_per_window is not None and not 1 <= tokens_per_window <= GRID * GRID:
            raise ValueError(
                f"tokens_per_window must lie in 1..{GRID * GRID}, got {tokens_per_window}"
            )
        self.device = device
        self.tokens_per_window = tokens_per_window
        self.generator = torch.Generator(device=device).manual_seed(random_state)
        self.global_moments = Moments(RAW_DIM, device)
        self.local_moments = Moments(RAW_DIM, device)
        self.n_windows = 0

    def add(self, loader: DataLoader, model: torch.nn.Module) -> None:
        """Encode every window of ``loader`` and add its tokens to the moments."""
        for batch in loader:
            # Both extractors' datasets yield the windows first, then their index or position.
            tiles = batch[0]
            global_tokens, local_tokens = encode_on_device(model, tiles, self.device)
            self.global_moments.update(global_tokens)
            if self.tokens_per_window is None:
                self.local_moments.update(local_tokens[:, CENTRE_TOKEN])
            else:
                draws = torch.rand(
                    len(local_tokens), GRID * GRID, generator=self.generator, device=self.device
                )
                picked = draws.argsort(dim=1)[:, : self.tokens_per_window]
                window = torch.arange(len(local_tokens), device=self.device)[:, None]
                drawn = local_tokens[window, picked].reshape(-1, local_tokens.shape[-1])
                self.local_moments.update(drawn)
            self.n_windows += len(tiles)

    def projections(self, n_components: int) -> tuple["Projection", "Projection"]:
        """Fit one exact PCA per stream from the accumulated moments."""
        print(
            f"[features] fitting PCA on {self.global_moments.count} windows / "
            f"{self.local_moments.count} tokens"
        )
        fitted, explained = [], []
        for moments in (self.global_moments, self.local_moments):
            mean, covariance = moments.mean_and_covariance()
            values, vectors = np.linalg.eigh(covariance)
            order = np.argsort(values)[::-1]
            width = min(n_components, moments.count, len(values))
            values, vectors = np.clip(values[order], 0, None), vectors[:, order[:width]]
            # An axis's sign is arbitrary; making its largest-magnitude loading positive keeps
            # the fit deterministic.
            vectors *= np.sign(vectors[np.abs(vectors).argmax(axis=0), np.arange(width)])
            explained.append(values[:width].sum() / max(values.sum(), 1e-12))
            fitted.append(
                Projection(mean.astype(np.float32), vectors.T.astype(np.float32), self.device)
            )
        print(
            f"[features]   explained variance: global {explained[0]:.3f}, "
            f"local {explained[1]:.3f}"
        )
        return fitted[0], fitted[1]
