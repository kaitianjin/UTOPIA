# Documentations for parameters in `config.yaml`

The `config.yaml` directs how UTOPIA runs. Here we provide the full template of `config.yaml`, though the minimal template provided in `README.md` works for general cases.

Contents:

1. [Templates for using the built-in model and external models](#1-templates)
2. [Parameters of the built-in model](#2-parameters-of-the-built-in-model)
3. [Parameters of an external model](#3-parameters-of-an-external-model)

---

## 1. Templates

### 1.1 Built-in model

If you just want to use the default built-in prediction model in UTOPIA, check out the following template.
At minimal, you only need to set `pixel_size_raw`, `targets.yaml`, `features.model_dir` and `samples`.

```yaml
mode: builtin                 
workspace: workspace          
targets: targets.yaml        
pixel_size_raw: 0.5           
superpixel_size: 8            
coordinate_type: raw_pixel    
scales: [1]                   
n_clusters: [15]              
predict: [gene]               
predict_genes: null           
benchmark: false              

samples:
  - name: lung               
    he: he_raw.png            
    pixel_size_raw: 0.5      
    rois:                     
      - {top_left: [672, 672], height: 1792, width: 1792, roles: [train, calib]}

tissue_mask:
  density_thresh: 100.0
  min_size: 50
  clean_background: true
  glcm_levels: 64

features:
  model_dir: /path/to/UNI     
  method: fast                
  harmony: auto              
  harmony_iterations: 20
  pca: 100
  batch_size: 64
  num_workers: 8
  device: cuda
  fit_patch_stride: null
  fit_tokens_per_patch: 49
  tissue_windows: true
  random_state: 42

clustering:
  source: null
  pca_dim: 80
  random_state: 42

default:                      
  epochs: 20
  batch_size: 8096
  learning_rate: 0.001
  dropout: 0.1
  loss: amse                  
  beta: 2.0
  seed: 42
  predict_batch_size: 32768

calibration:
  fold_widths: [200, 100, 50]
  safe_boundary: 16
  fold_n_clusters: null
  parallel_processes: 8

sampling:                     
  n_draws: 100
  coverage: 0.95
  dropout: null
  seed: 1000
  batch_size: 32768

inference:
  methods: [UTOPIA]
  bh_scope: sample_cluster   
  min_calib_units: null
  p_min: 0.01
  p_max: 0.5
  visualize: true
```

### 1.2 External model, sample folders

When you supply the predictions made by other models, UTOPIA only quantifies their
confidence. 

```yaml
mode: external
model: my_model              
workspace: workspace
targets: targets.yaml
pixel_size_raw: 0.25        
superpixel_size: 4           
n_clusters: [15]
he_batch_effect_correction: false

spot:
  shape: circle               
  radius: 30               

samples:
  - name: slide_A             
    role: calibration         
    he: he.jpg
    pixel_size_raw: 0.2512    
    spot: {radius: 50}        
  - name: slide_B
    role: test
    he: he.jpg

features:
  model_dir: /path/to/UNI
  batch_size: 64
  num_workers: 8
  device: cuda
  harmony_iterations: 20
  random_state: 42

clustering:
  pca_dim: 100
  random_state: 42

tissue_mask:
  density_thresh: 100.0
  min_size: 50
  clean_background: true
  glcm_levels: 64

inference:
  methods: [UTOPIA]               
  bh_scope: sample_cluster
  min_calib_units: null
  p_min: 0.01
  p_max: 0.5
  visualize: true
```

---

## 2. Parameters of the built-in model

### 2.1 General

| Key | Default | Meaning |
|---|---|---|
| `workspace` | `workspace` | Folder every output is written to. UTOPIA will create a subfolder with this name in the folder where `config.yaml` is. |
| `targets` | `targets.yaml` | File that lists the genes and cell types to infer confidence. |
| `mode` | `builtin` | `builtin`: UTOPIA trains and predicts using the built-in model. `external`: predictions are made by external models. |
| `model` | `default_model` | Name the calibration data, predictions and results are stored under. The built-in model is always `default_model`. |

### 2.2 Image and resolution

| Key | Default | Meaning |
|---|---|---|
| `pixel_size_raw` | required | Resolution of the H&E image, in microns.  |
| `superpixel_size` | `8` | Size of one superpixel, in microns. The superpixel is the finest unit UTOPIA predicts at. For example, 8 means that one superpixel is 8 um x 8 um.|
| `coordinate_type` | `raw_pixel` | Coordinates used in `positions.npy` and the locations of ROIs. `raw_pixel`: pixels of the H&E image you provide. `superpixel`: row and column index on the superpixel grid. |
| `scales` | `[1]` | Resolutions for UTOPIA to report confidence at. Scale `s` aggregates `s × s` superpixels into one unit. Therefore using the default 8 µm superpixels, `[1, 2, 4]` reports at 8, 16 and 32 µm. |

UTOPIA rescales each H&E image to `superpixel_size / 16` microns per pixel so that a superpixel is 16 × 16 pixels.
When `pixel_size_raw` is larger than `superpixel_size / 16`, the H&E image has to be up-scaled, which adds no detail. 
UTOPIA then gives out a warning.

### 2.3 What to predict

| Key | Default | Meaning |
|---|---|---|
| `predict` | `[gene]` | What to predict: `gene`, `cell_type`, or both. Each one needs its input folder (`genes/` or `cell_types/`) in every sample. |
| `predict_genes` | `null` | Genes to predict and store. `null` predicts every gene in `gene_names.json`. Otherwise a list of gene names, or the path of a JSON file that holds such a list. It must include every gene named in `targets.yaml`. A short list makes the run faster and the output folder `workspace` smaller. |
| `benchmark` | `false` | This should be `false` for normal users. Set to `true` when ST measurements also exist for the test area. UTOPIA then writes the false discovery proportion curves. |
| `n_clusters` | `[15]` | Numbers of histology clusters to be used. Users can try a set of numbers such as `[15,20,25]`.|


### `samples`

A list with one entry per sample. A sample is one H&E image and its folder
`samples/<name>/`.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Folder name under `samples/`. Names must be unique. |
| `he` | `he_raw.png` | File name of the H&E image inside the sample folder. |
| `pixel_size_raw` | the global value | Resolution of the H&E image, in microns. This per-sample setting overrides the global `pixel_size_raw`. |
| `rois` | none | Regions of interest where ST measurement is available. |

Each ROI takes four keys:

| Key | Default | Meaning |
|---|---|---|
| `top_left` | required | `[row, col]` of the top-left corner. |
| `height` | required | Number of rows the ROI spans. |
| `width` | required | Number of columns the ROI spans. |
| `roles` | `[train, calib]` | What the ROI is used for (below). |

`top_left`, `height` and `width` are in the units named by `coordinate_type`. 
By default, these coordinates should be in terms of the pixels in the H&E image as specified by `sample.he`.

The roles decide how the run is calibrated:

| `roles` | The ROI is | 
|---|---|
| `[train, calib]` | used for both training models and calibration | 
| `[train]` | used only for training |
| `[calib]` | used only for calibration | 

- Everything outside all ROIs is the test area, where UTOPIA reports
  confidence. A sample without `rois` is test area as a whole. Therefore, users can provide a sample without `rois` for out-of-sample prediction.
- Over all samples, at least one ROI must have the `train` role and at least
  one the `calib` role.


### 2.5 `tissue_mask`

Settings of HistoSweep that filters background non-tissue superpixels.  

| Key | Default | Meaning |
|---|---|---|
| `density_thresh` | `100.0` | A superpixel whose color is rarer on the slide than this count is a background candidate. A larger value removes more. |
| `min_size` | `50` | Smallest piece of tissue to be kept, in superpixels. Smaller isolated pieces are removed. Decrease the value if you want to retain more tissue areas. Increase the value if noise such as stains, dirty spots on H&E are kept. |
| `clean_background` | `true` | Whether to remove the pieces smaller than `min_size`. |
| `glcm_levels` | `64` | Number of grey levels used by the texture check that refines the background. |

### 2.6 `features`

How the feature extraction on H&E works.

| Key | Default | Meaning |
|---|---|---|
| `model_dir` | required | Folder that holds UNI's `pytorch_model.bin`. |
| `method` | `fast` | Feature extractor (below). |
| `harmony` | `auto` | Whether Harmony removes slide-to-slide differences from the features. `auto` turns it on for an out-of-sample run with at least two samples. `true` needs at least two samples and the `fast` method. |
| `harmony_iterations` | `20` | Maximum number of Harmony iterations. |
| `pca` | `100` | Dimensions kept for each of the global and local features. A feature vector has `2 × pca` values. `null` keeps the raw 1024 + 1024 dimensions. |
| `batch_size` | `64` | How many 224x224 image patches per forward pass of UNI. |
| `num_workers` | `8` | Worker processes that load the image patches. |
| `device` | `cuda` | Device UNI runs on. The CPU is used when no GPU is available. |
| `fit_patch_stride` | `null` | The PCA is fitted on every n-th window. `null` uses every 224x224 patch for `fast`, and every 16th tissue superpixel for `slow` and `very_slow`. |
| `fit_tokens_per_patch` | `49` | `fast` only: local tokens drawn from each 14x14 patch (224x224 px) to fit the PCA, from 1 to 196. |
| `tissue_windows` | `true` | `fast` only: extract features only on tissue areas. `false` extract features on all of the image. |
| `random_state` | `42` | Seed of the PCA and of Harmony. |

The three extractors:

| `method` | What it does | Speed |
|---|---|---|
| `fast` | Uses CATCH to extract features | fastest |
| `slow` | One forward pass per superpixel that is in the tissue area | slow |
| `very_slow` | One forward pass per superpixel of the whole slide | slowest |

### 2.7 `clustering`

| Key | Default | Meaning |
|---|---|---|
| `source` | `null` | Name the cluster files are stored under (`clusters_<source>.npz`). `null` uses the name of the feature extractor. |
| `pca_dim` | `80` | Dimensions the features are reduced to before KMeans. Features in dimensions less than `pca_dim` are used as they are. |
| `random_state` | `42` | Seed of the PCA and of KMeans. |


### 2.8 `default`

Hyper-parameters of the built-in prediction model.

| Key | Default | Meaning |
|---|---|---|
| `epochs` | `20` | Training epochs. |
| `batch_size` | `8096` | Superpixels per training batch. |
| `learning_rate` | `0.001` | Learning rate of the Adam optimiser. |
| `dropout` | `0.1` | Dropout rate. |
| `loss` | `amse` | Loss of the gene model. `amse`: asymetric mean squared error. `rmse`: root mean squared error. |
| `beta` | `2.0` | Weight on `amse` loss. |
| `seed` | `42` | Training seed. |
| `predict_batch_size` | `32768` | Superpixels per forward pass when predicting. |

### 2.9 `calibration`

| Key | Default | Meaning |
|---|---|---|
| `fold_widths` | `[200, 100, 50]` | The adaptive fold sizes in superpixels. 200 means a rectangle of 200x200 superpixels. |
| `safe_boundary` | `16` | Width of the margin inside each fold that is held out but not used for calibration, in superpixels. |
| `fold_n_clusters` | `null` | Only used when `n_clusters` has more than one value. `null` uses the first value of `n_clusters`. If you want to use 15 clusters to decide how to cut folds, set it to 15. The histology clusters used here determine how UTOPIA cut calibration ROIs into folds with adaptive sizes. |
| `parallel_processes` | `8` | Number of models trained at the same time. `1` trains them one after another. Lower it when the GPU runs out of memory. |

`fold_widths`, `safe_boundary` and `tissue_mask.min_size` are always in
superpixels, whatever `coordinate_type` is.

### 2.10 `sampling`

Only needed when `inference.methods` uses `mc_dropout`, `ensemble` or
`resampling`. 

| Key | Default | Meaning |
|---|---|---|
| `n_draws` | `100` | Number of draws K, at least 2.  |
| `coverage` | `0.95` | Coverage of the prediction interval reported with each prediction, between 0 and 1. |
| `dropout` | `null` | Dropout rate of the `mc_dropout` draws. `null` uses the rate the model was trained with. |
| `seed` | `1000` | Base seed. draw `k` uses `seed + k`. |
| `batch_size` | `32768` | Units per forward pass when a draw is generated. |

### 2.11 `inference`

How the p-values and confidence scores are computed and reported.

| Key | Default | Meaning |
|---|---|---|
| `methods` | `null` | Uncertainty methods to report (below). `null` reports `UTOPIA` alone. |
| `bh_scope` | `sample_cluster` | Group of units the Benjamini–Hochberg correction is applied within (below). |
| `min_calib_units` | `null` | `UTOPIA` only: calibration units a cluster needs to be tested. A cluster with fewer keeps a p-value of 1. `null` asks for 200 units or 5 % of the cluster, whichever is larger. |
| `p_min` | `0.01` | Adjusted p-value shown as a confidence of 1. |
| `p_max` | `0.5` | Adjusted p-value shown as a confidence of 0. |
| `visualize` | `true` | Whether to write the prediction maps, the confidence maps and the benchmark curves as images. |

**Methods.** `methods` is a list of any of:

| Method | What it reports | 
|---|---|
| `UTOPIA` | Conformal p-values, calibrated within each histology cluster. Reported once per value of `n_clusters`. |
| `conformal_marginal` | Conformal confidence without histology clusters. | 
| `none` | The prediction only, without confidence. | 
| `mc_dropout` | Confidence from `sampling.n_draws` passes of one model with dropout left on. | 
| `ensemble` | Confidence from `sampling.n_draws` models trained with different seeds. | 
| `resampling` | Confidence from `sampling.n_draws` models trained on bootstrap resamples of the training data. | 

**Benjamini–Hochberg scope.** The adjusted p-values control the false
discovery rate within each group named by `bh_scope`, separately for each
target:

| `bh_scope` | One group is |
|---|---|
| `sample_cluster` | one histology cluster of one sample |
| `sample` | one sample |
| `cluster` | one histology cluster, over all samples |
| `global` | all test units |

**Confidence score.** The confidence shown on the maps is the adjusted p-value
on a logarithmic scale: 1 at `p_min` or below, 0 at `p_max` or above.

## 3. Parameters of an external model

With `mode: external` UTOPIA does not train or predict. 

### 3.1 General

| Key | Default | Meaning |
|---|---|---|
| `mode` | | Must be `external`. |
| `model` | required | Name of the external model. Results are stored under it. Any name except `default_model`. |
| `workspace`, `targets` | | As in section 2.1. |
| `pixel_size_raw` | required | As in section 2.2. |
| `superpixel_size` | `4` | For external models, the confidence is only reported at the resolution where the prediction is made at. Therefore, `superpixel_size` does not change the resolution of the predictions. Instead, it controls the resolution of the output figures. The output figure will be at `superpixel_size` um/px. By default, the output figure is at 4 um/px. |
| `n_clusters` | `[15]` | Numbers of histology clusters UTOPIA makes.  |
| `he_batch_effect_correction` | `false` | `true`: Harmony removes slide-to-slide differences from the spot features before they are clustered. |

### 3.2 `spot`

The geometry of the ST spot on H&E image.

| Key | Default | Meaning |
|---|---|---|
| `shape` | `circle` | `circle` or `square`, the same for every sample. |
| `radius` | | `circle` only: radius in microns. This determines how H&E patch for a spot is cropped. For Visium, we choose a slightly larger radius `30` um for the 55-um spot.|
| `width` | | `square` only: side length in microns. |


### 3.3 `samples`

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Folder name under `samples/`. Names must be unique. |
| `role` | required | `calibration`: the spots have predictions and measurements, and calibrate the test. `test`: the spots whose confidence is reported. |
| `he` | none | File name of the H&E image inside the sample folder. |
| `pixel_size_raw` | the global value | Resolution of the H&E image, in microns. This per-sample setting overrides the global `pixel_size_raw`. |
| `spot` | the global `spot` | This sample's own `radius` or `width`, in microns. The shape cannot be changed per sample. |

A run needs at least one `calibration` sample and at least one `test` sample.

### 3.4 `features`

How the feature extraction on H&E works. Only needed when UTOPIA makes the
histology clusters, that is, when you do not provide cluster labels yourself.

| Key | Default | Meaning |
|---|---|---|
| `model_dir` | required | Folder that holds UNI's `pytorch_model.bin`. |
| `batch_size` | `64` | How many 224x224 image patches per forward pass of UNI. |
| `num_workers` | `8` | Worker processes that load the image patches. |
| `device` | `cuda` | Device UNI runs on. The CPU is used when no GPU is available. |
| `harmony_iterations` | `20` | Maximum number of Harmony iterations. Only used with `he_batch_effect_correction: true`. |
| `random_state` | `42` | Seed of Harmony. |

The other keys of section 2.6 belong to the built-in model and are not accepted
here.

### 3.5 `clustering`

| Key | Default | Meaning |
|---|---|---|
| `pca_dim` | `100` | Dimensions the features are reduced to before KMeans. Features in dimensions less than `pca_dim` are used as they are. |
| `random_state` | `42` | Seed of the PCA and of KMeans. |

### 3.6 `tissue_mask`

Settings of HistoSweep that filters background non-tissue superpixels. For
external models the mask is only drawn in grey behind the spots on the output
figures.

| Key | Default | Meaning |
|---|---|---|
| `density_thresh` | `100.0` | A superpixel whose color is rarer on the slide than this count is a background candidate. A larger value removes more. |
| `min_size` | `50` | Smallest piece of tissue to be kept, in superpixels. Smaller isolated pieces are removed. Decrease the value if you want to retain more tissue areas. Increase the value if noise such as stains, dirty spots on H&E are kept. |
| `clean_background` | `true` | Whether to remove the pieces smaller than `min_size`. |
| `glcm_levels` | `64` | Number of grey levels used by the texture check that refines the background. |

### 3.7 `inference`

How the p-values and confidence scores are computed and reported.

| Key | Default | Meaning |
|---|---|---|
| `methods` | `null` | Uncertainty methods to report (below). `null` reports `UTOPIA` alone. |
| `bh_scope` | `sample_cluster` | Group of units the Benjamini–Hochberg correction is applied within (below). |
| `min_calib_units` | `null` | `UTOPIA` only: calibration units a cluster needs to be tested. A cluster with fewer keeps a p-value of 1. `null` asks for 200 units or 5 % of the cluster, whichever is larger. |
| `p_min` | `0.01` | Adjusted p-value shown as a confidence of 1. |
| `p_max` | `0.5` | Adjusted p-value shown as a confidence of 0. |
| `visualize` | `true` | Whether to write the prediction maps, the confidence maps and the benchmark curves as images. |

**Methods.** `methods` is a list of any of the following. `mc_dropout`,
`ensemble` and `resampling` need the built-in model and are not available here.

| Method | What it reports | 
|---|---|
| `UTOPIA` | Conformal p-values, calibrated within each histology cluster. Reported once per value of `n_clusters`. |
| `conformal_marginal` | Conformal confidence without histology clusters. | 
| `none` | The prediction only, without confidence. | 

**Benjamini–Hochberg scope.** The adjusted p-values control the false
discovery rate within each group named by `bh_scope`, separately for each
target:

| `bh_scope` | One group is |
|---|---|
| `sample_cluster` | one histology cluster of one sample |
| `sample` | one sample |
| `cluster` | one histology cluster, over all samples |
| `global` | all test units |

**Confidence score.** The confidence shown on the maps is the adjusted p-value
on a logarithmic scale: 1 at `p_min` or below, 0 at `p_max` or above.
