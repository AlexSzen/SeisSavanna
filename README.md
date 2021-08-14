# SeisSavanna

This repo reproduces the species classification results from the paper Seismic savanna: Machine learning for classifying wildlife and behaviours using ground-based vibration field recordings.

## Download the data 

You can download the datasets from: . There are three main folders:
- 'dset_allspec_150' containing animal signals from 10 different species, up to 150m distance. This is the dataset we are using in the codes included here.
- 'dset_elbeh_150' containing elephant signals up to 150m distance, annotated either for running or walking.
- 'dset_rumbles_60' containing elephant rumbles and elephant locomotion up to 60m distance.

## How to work with Docker

### Get the image
#### Option 1: build the image

Be in the root directory of this repository. Running the following builds a Docker image tagged with the name `kenya`.

```
docker build -t kenya .
```

#### Option2 : pull the image

Not available yet.

### Run the image

This instantiates a container from the `kenya` Docker image and gets us an interactive terminal inside the container.
```
docker run --rm -it kenya
```

We can do the same and also mount a host file system folder to the container. The following mounts your directory  containing the data (replace <DATA_DIR> with the absolute path of the directory containing your data) into the `/data` directory within the container. Any changes in this directory will be persistent and reflected in the host file system.

```
docker run --rm -it -v <DATA_DIR>:/data kenya
```

## Reproduce results

There are three stages:
- Produce a dataset at the desired maximal distance.
- Given a dataset, compute train/val/test splits and normalisation quantities (mean and std).
- Train the model

Step 1: Produce a dataset at the desired maximal distance

After editing the inputs in ```data_scripts/inputs/params_reduce_dset.py``` with the desired distance, run ```python -m data_scripts.reduce_dataset```. This will produce a new reduced distance dataset in the specified data folder.

Step 2: Given a dataset, compute train/val/test splits and normalisation quantities (mean and std).

After editing ```data_scripts/inputs/params_split_norm.py```, run ```python -m data_scripts.make_split_norm```. This will output indices for train/val/test splits and normalisation quantities in the specified dataset folder.

Step 3: Train the model.

For general flags as well as parameters for model training, we use the Hydra package, for which configuration files are passed as jsons. Edit general flags in ```ml_scripts/config/config.json```, hyperparameter specific flags in ```ml_scripts/config/hp/```and data specific flags in ```ml_scripts/config/data```. Subsequently, train the model by running ```python -m ml_scripts.train_hydra```. 




