# ViMusic Data Version Control (DVC)

DVC system for viMusic, version controlling dataset and model files. 

It's linked to the remote `ml-server` as `dustin` credential in folder: `/mnt/sdc1/Projects/viMusic/dvc`

# Installation
```bash
pip install dvc
pip install dvc[ssh] or pip install 'dvc[ssh]'
```

# Getting Started

The datasets and models should be stored in `models` folder. 

The files with **.dvc** suffix track individual models and datasets

### Pull models and datasets

* You can pull specific datasets and models by targeting to specific **.dvc** file
```bash
dvc pull models/remi/checkpoint.dvc or

dvc pull models/remi/datasets.dvc #Warning, this is huge
```

* or pull all the dvc files

```bash
dvc pull #Warning, this is huge
```

### Check dvc status

* Check status of all dvc files with local files

```bash
dvc status
```

* or specific dvc file

```bash
dvc status models/remi/checkpoint.dvc
```

* Or status on **remote vs local** dvc

```bash
dvc status --remote ml-server
```

### Add new dvc

* To track new model folder or stage the changes, this will create  new **folder_name.dvc** file or change the existing **.dvc** file
```bash
dvc add models/folder_name
```

* Apply the model changes by committing git changes and run:
```bash
dvc push 
```

* or specific dvc

```bash
dvc push models/remi/checkpoint.dvc
```

For more informations, please refer: https://dvc.org/doc/get-started