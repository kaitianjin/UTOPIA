# UTOPIA demo dataset: external model

Download the dataset from https://upenn.box.com/v/UtopiaDemoLuad

Predictions made by Loki on five lung adenocarcinoma Visium slides (Fig 5 of the
paper). UTOPIA calibrates on four of them and reports the confidence on the test slide. 


```
UTOPIA_demo_external/
├── config.yaml                 
├── targets.yaml                
└── samples/
    ├── P6_LUAD/  P7_LUAD2/  P14_LUAD/  P15_LUAD/     # calibration slides
    │   ├── he.jpg           
    │   └── genes/
    │       ├── positions.npy   # (N, 2)  locations of each Visium spots
    │       ├── prediction.npy  # (N, 13)  Loki's prediction in CP10k
    │       ├── expression.npy  # (N, 13)  measured Visium expression in CP10k
    │       └── gene_names.json # the gene names
    └── P20_LUAD/                                      # test slide
        ├── he.jpg
        └── genes/  positions.npy  prediction.npy  gene_names.json
```

```bash
cd demo_luad
utopia check config.yaml
utopia run   config.yaml    
utopia view  UTOPIA_output
```


The output of UTOPIA is written to `demo_luad/UTOPIA_output`. 

Before you run, make sure you download `pytorch_model.bin` from [UNI](https://huggingface.co/MahmoodLab/UNI) and modify `features.model_dir` in `config.yaml`.
