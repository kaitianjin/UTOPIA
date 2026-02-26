import os
import torch
from torchvision import transforms
import timm
import numpy as np
from ..utils.io import load_image
from PIL import Image
import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import sys 


class PatchDataset(Dataset):
    def __init__(self, image, patch_size=16, stride=16):
        self.image = image
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        self.shape_ori = np.array(image.shape[:2])
        self.num_patches = ((self.shape_ori - patch_size) // stride + 1)
        self.total_patches = self.num_patches[0] * self.num_patches[1]

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        i = (idx // self.num_patches[1]) * self.stride
        j = (idx % self.num_patches[1]) * self.stride
        
        # Extract 224x224 patch centered on the 16x16 patch
        center_i, center_j = i + 8, j + 8
        start_i, start_j = max(0, center_i - 112), max(0, center_j - 112)
        end_i, end_j = min(self.shape_ori[0], center_i + 112), min(self.shape_ori[1], center_j + 112)
        
        patch = self.image[start_i:end_i, start_j:end_j]
        
        # Pad if necessary to ensure 224x224 size
        if patch.shape[0] < 224 or patch.shape[1] < 224:
            padded_patch = np.zeros((224, 224, 3), dtype=patch.dtype)
            padded_patch[(224-patch.shape[0])//2:(224-patch.shape[0])//2+patch.shape[0], 
                         (224-patch.shape[1])//2:(224-patch.shape[1])//2+patch.shape[1]] = patch
            patch = padded_patch
        
        patch = Image.fromarray(patch.astype('uint8')).convert('RGB')
        return self.transform(patch), (i, j)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--down_samp_step', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=64)
    return parser.parse_args()

def create_model(local_dir):
    model = timm.create_model(
        "vit_large_patch16_224", 
        img_size=224, 
        patch_size=16, 
        init_values=1e-5, 
        num_classes=0,  # This ensures no classification head
        global_pool='',  # This removes global pooling
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=False)
    return model

@torch.inference_mode()
def extract_features(model: torch.nn.Module, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #Extract both 224-level and 16-level features.
    feature_emb = model(batch)
    final_output, _ = model.forward_intermediates(batch, return_prefix_tokens=False)
    local_emb = final_output[:,1:]
    patch_emb = local_emb.permute(0, 2, 1).reshape(batch.shape[0], 1024, 14, 14)
    return feature_emb, patch_emb
    

@torch.inference_mode()
def main():
    args = get_args()

    model = create_model(args.model_path)
    
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    
    he = load_image(os.path.join(args.data_path, "HE", "he.png"))
    dataset = PatchDataset(he, stride=16*args.down_samp_step)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    patch_embeddings = []
    part_cnts = 0
    for batch_idx, (patches, positions) in enumerate(tqdm.tqdm(dataloader, total=len(dataloader))):

        patches = patches.to(device, non_blocking=True)
        
        feature_emb, patch_emb = extract_features(model, patches)
        
        # Process each patch
        for idx in range(len(positions[0])):
            
            # Extract features
            center_feature = feature_emb[idx, 0]  # Use the [CLS] token as the 224-level feature
            patch_feature = patch_emb[idx, :, 7, 7]  # Use the center patch feature
            
            # Concatenate 224-level and 16-level features
            combined_feature = torch.cat([center_feature, patch_feature])
            patch_embeddings.append(combined_feature.cpu().numpy())

        # Save embeddings when it accumulates around 8 GB in RAM
        if (batch_idx*args.batch_size)//1000000 < ((batch_idx+1)*args.batch_size)//1000000 or batch_idx == len(dataloader) - 1:
            patch_embeddings = np.array(patch_embeddings)
            np.save(os.path.join(args.data_path, "embeddings", f"embeddings_part_{part_cnts}.npy"), patch_embeddings)
            patch_embeddings = []
            part_cnts += 1

if __name__ == '__main__':
    sys.exit(main())