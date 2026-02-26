# Multiscale confidence quantification for virtual spatial transcriptomics with UTOPIA
Kaitian Jin, Zihao Chen, Xiaokang Yu, Musu Yuan, Amelia Schroeder, Bernhard Dumoulin, Yunhe Liu, Linghua Wang, Jeong Hwan Park, Tae Hyun Hwang, Katalin Susztak, Zhimei Ren, Nancy R. Zhang12, Mingyao Li

<p align="center">
  <img src="assets/pipeline.jpeg" alt="UTOPIA Pipeline" />
</p>


UTOPIA (Uncertainty-aware Trustworthy Tools for Spatial Omics Predictions in All Scales), a general, model-agnostic framework for quantifying prediction confidence in virtual ST.

# Quick start
UTOPIA is designed to work flexibly with any Python environments with numpy, pytorch, pandas installed. We tested UTOPIA in the following environment

```bash
git clone https://github.com/kaitianjin/UTOPIA
cd UTOPIA
conda create -n utopia python=3.9
conda activate utopia
pip install -r requirements.txt
pip install -e .
```

***If the raw H&E image is in .svs, .ome.tif, .btf format, you also need to install `openslide` and/or `tifffile`. See `requirements.txt`.***

## Demo
We have prepared a demo dataset (the same as Figure 2 in the paper) for you to get familiar with UTOPIA. We have already aligned and processed the Xenium data and put them in the 
correct format UTOPIA needs. We also skip the first two time-consuming steps to save you time. You can play around with the demo dataset using the following command. 
You can modify `targets_to_infer.json` to select genes or cell types you'd like to explore. 
For full list of genes and cell types, see `gene_list.txt` and `cell_type_list.txt`.

```bash
cd demo
# utopia --config utopia_config --get_tissue_mask
# utopia --config utopia_config --feature_extraction
utopia --config utopia_config --process_roi_data
utopia --config utopia_config --histology_clustering
utopia --config utopia_config --calibration
utopia --config utopia_config --inference
```

# Instructions

## Input data needed for UTOPIA

UTOPIA requires the following seven files, organized within a structured data directory.

### H&E Image

**`he.png`** — A scaled histology image at 0.5 µm/px whose width and height are each divisible by 224. To generate this file, run:

```bash
utopia --rescale_raw_he --he_raw <RAW_HE_IMAGE_FILE> --pixel_size_raw <PIXEL_SIZE>
```

where `<RAW_HE_IMAGE_FILE>` is the the raw H&E image (supported formats: `.jpg`, `.png`, `.tiff`, `.svs`, `.btf`, `.ome.tif`), and `<PIXEL_SIZE>` is the original resolution in microns per pixel.

### Gene Expression Data

**`tr_image_scale_1.npy`** — A NumPy array of shape `(W/16, L/16, N)` and dtype `np.uint16`, where `W` and `L` are the width and height of `he.png` in pixels, and `N` is the number of genes. Each entry represents raw gene counts at the corresponding 16×16 px superpixel.

**`idx_to_gene.pickle`** — A Python dictionary mapping array indices (along the last dimension of `tr_image_scale_1.npy`) to gene names.

### Cell Type Annotation Data

**`ct_image_scale_1.npy`** — A 2D NumPy array of shape `(W/16, L/16)` and dtype `np.float64` containing integer cell type labels. Labels should range from `0` to `M-1`, where `M` is the total number of cell types. Superpixels without annotation should be assigned a value of `-1`.

**`idx_to_ct.pickle`** — A Python dictionary mapping integer labels (`0` to `M-1`) to their corresponding cell type names.

### Configuration Files

**`utopia_config`** — The main UTOPIA configuration file. Users should modify this file to specify data-dependent settings, including the location(s) of region(s) of interest (ROIs).

**`targets_to_infer.json`** — A JSON file specifying the genes, meta-genes, cell types, and composite cell types for which prediction confidence will be inferred.

### Directory Structure

Please organize the data as follows.

```
data/
├── targets_to_infer.json        # list of genes/meta-genes/cell types to infer confidence
├── utopia_config                # UTOPIA configuration file
├── HE/                          # folder containing H&E raw image
│   └── he.png                   # the histology image in 0.5 um/px
├── transcripts/                 # folder containing spatial transcriptomics data
│   ├── tr_image_scale_1.npy     # ground-truth gene expression at scale 1 (8 um/spx)
│   └── idx_to_gene.pickle       # mapping from array index to gene name
└── cell_type/                   # folder containing cell type annotation data
    ├── ct_image_scale_1.npy     # annotated cell types at scale 1 (8 um/spx)
    └── idx_to_ct.pickle         # mapping from array index to cell type label
```

Then run 
```bash
cd data
utopia --config utopia_config --get_tissue_mask
# feature_extraction can take more than 24 hours for large H&E image
utopia --config utopia_config --feature_extraction
utopia --config utopia_config --process_roi_data
utopia --config utopia_config --histology_clustering
utopia --config utopia_config --calibration
utopia --config utopia_config --inference
```

## Output data
After running UTOPIA, the data folder will become

```
data/
├── targets_to_infer.json        # list of genes/meta-genes/cell types to infer confidence
├── utopia_config                # UTOPIA configuration file
├── HE/                          # folder containing H&E raw image
│   └── ...
├── transcripts/                 # folder containing spatial transcriptomics data
│   └── ...
├── cell_type/                   # folder containing cell type annotation data
│   └── ...
├── embeddings/                  # folder containing image feature embeddings from UNI
├── histology_clustering/        # folder containing histology clustering results
├── HistoSweep_Output/           # folder generated by HistoSweep
├── mask/                        # folder containing various masks
├── calibration/                 # folder containing calibration data
└── results/                     # folder containing UTOPIA-inferred confidence results
```

You can see the confidence scores and raw predictions in the `results` folder.