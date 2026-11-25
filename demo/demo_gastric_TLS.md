# UTOPIA demo dataset: in-sample

Download the dataset from https://upenn.box.com/v/UtopiaDemoGastricTLS

The demo version of the Xenium gastric dataset used in Figure 2. The demo dataset
only includes ST data for two ROIs on 50 HVGs and TLS genes. The H&E image was 
compressed using jpeg.

```
demo_gastric_TLS/
├── config.yaml                 # configurations for UTOPIA
├── targets.yaml                # targets for UTOPIA to infer confidence on 
└── samples/gastric/
    ├── he.jpg                  # H&E image, 48384 x 19712 px at 0.5 um/px
    ├── genes/
    │   ├── positions.npy       # (N, 2) int32 coordinates in (row, col) of H&E pixel, indicating each superpixel's top-left corner
    │   ├── expression.npy      # (N, 63) uint16 counts, the measured expression of 63 genes for these N superpixels
    │   └── gene_names.json     # 15 TLS genes and the top 50 highly variable genes with 2 overlaps.
    └── cell_types/
        ├── positions.npy       # (M, 2) int32 coordinates in (row, col) of H&E pixel, indicating each superpixel's top-left corner
        ├── labels.npy          # (M,) the cell type index for each of the M superpixels
        └── ct_names.json       # the 14 cell types in this dataset
```

Run it:

```bash
cd demo_gastric_TLS
utopia check config.yaml        
utopia run   config.yaml
utopia view  UTOPIA_output
```

The output of UTOPIA is written to `demo_gastric_TLS/UTOPIA_output`. 

Before you run, make sure you download `pytorch_model.bin` from [UNI](https://huggingface.co/MahmoodLab/UNI) and modify `features.model_dir` in `config.yaml`.
