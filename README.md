# Vistion Transformer for ECG Classification (Super Class)

1D vision transformer (vit) for ecg interpretable classification.

---

## Installation

python version == `3.10.12`  
dependencies in `requirements.txt`

## Data Preprocessing

You can find the steps to process raw dataset for our training in `ecg-ptb-data-preprocessing.ipynb` under notebooks folder.

## Training

You can follow `ViT-ECG-5-class-classification-model.ipynb` to train the model or you can directly run below command:

```bash
python train.py <path-to-your-processed-dataset-folder>
```

## Plot attension

```bash
python plot_attention.py  <wfdb_file> <model_weights_path> <lead_index>`  
```
