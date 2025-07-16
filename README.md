# RLDS Dataset Conversion for RAIL-DROID

This repo demonstrates how to convert an existing dataset into RLDS format for finetuning pizerofast, building off of Karl's RLDS dataset builder to make LeRobot conversion smoother.

## Installation

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow`, 
`tensorflow_datasets`, `tensorflow_hub`, `apache_beam`, `matplotlib`, `plotly` and `wandb`.


## Run Example RLDS Dataset Creation

Before modifying the code to convert your own dataset, run the provided example dataset creation script to ensure
everything is installed correctly. Run the following lines to create some dummy data and convert it to RLDS.
```
cd example_dataset
python3 create_example_data.py
tfds build
```

This should create a new dataset in `~/tensorflow_datasets/example_dataset`. Please verify that the example
conversion worked before moving on.


## Converting your Own Dataset to RLDS

Now we can modify the provided example to convert your own data. Follow the steps below:

1. **Subclass GenericDroidFinetuningSet in your own Folder**: Copy the `example_dataset` folder, and replace `example_dataset_dataset_builder.py` with a new file named after your dataset (e.g. `finetuningtask_dataset_dataset_builder.py`). The code in this file can look like the following:

```
from generic_droid_dataset_builder import GenericDroidFinetuningSet


class FinetuningTaskDataset(GenericDroidFinetuningSet):
    """DatasetBuilder subclass for finetuning dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(<RAW_DATASET_FILEPATH>, *args, **kwargs)
```

Where `RAW_DATASET_FILEPATH` is the (relative to this directory) filepath to your data, which should be a collection of folders that contain a `trajectory.h5` file, a `metadata_RAIL...` file, and a `recordings` file with mp4 videos of the trajectories.

Once you've created this file, (and filled out the CITATIONS.bib if you need to), you should be all set to run dataset conversion. Inside the dataset directory, run:
```
tfds build --overwrite
```
The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/tensorflow_datasets/<name_of_your_dataset>`.

(The rest of these instructions are from Karl's original repo.)

### Parallelizing Data Processing
By default, dataset conversion is single-threaded. If you are parsing a large dataset, you can use parallel processing.
For this, replace the last two lines of `_generate_examples()` with the commented-out `beam` commands. This will use 
Apache Beam to parallelize data processing. Before starting the processing, you need to install your dataset package 
by filling in the name of your dataset into `setup.py` and running `pip install -e .`

Then, make sure that no GPUs are used during data processing (`export CUDA_VISIBLE_DEVICES=`) and run:
```
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"
```
You can specify the desired number of workers with the `direct_num_workers` argument.

## Visualize Converted Dataset
To verify that the data is converted correctly, please run the data visualization script from the base directory:
```
python3 visualize_dataset.py <name_of_your_dataset>
``` 
This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension.
Note, if you are running on a headless server you can modify `WANDB_ENTITY` at the top of `visualize_dataset.py` and 
add your own WandB entity -- then the script will log all visualizations to WandB. 

## Add Transform for Target Spec

For X-embodiment training we are using specific inputs / outputs for the model: input is a single RGB camera, output
is an 8-dimensional action, consisting of end-effector position and orientation, gripper open/close and a episode termination
action.

The final step in adding your dataset to the training mix is to provide a transform function, that transforms a step
from your original dataset above to the required training spec. Please follow the two simple steps below:

1. **Modify Step Transform**: Modify the function `transform_step()` in `example_transform/transform.py`. The function 
takes in a step from your dataset above and is supposed to map it to the desired output spec. The file contains a detailed
description of the desired output spec.

2. **Test Transform**: We provide a script to verify that the resulting __transformed__ dataset outputs match the desired
output spec. Please run the following command: `python3 test_dataset_transform.py <name_of_your_dataset>`

If the test passes successfully, you are ready to upload your dataset!

## Upload Your Data

We provide a Google Cloud bucket that you can upload your data to. First, install `gsutil`, the Google cloud command 
line tool. You can follow the installation instructions [here](https://cloud.google.com/storage/docs/gsutil_install).

Next, authenticate your Google account with:
```
gcloud auth login
``` 
This will open a browser window that allows you to log into your Google account (if you're on a headless server, 
you can add the `--no-launch-browser` flag). Ideally, use the email address that
you used to communicate with Karl, since he will automatically grant permission to the bucket for this email address. 
If you want to upload data with a different email address / google account, please shoot Karl a quick email to ask 
to grant permissions to that Google account!

After logging in with a Google account that has access permissions, you can upload your data with the following 
command:
```
gsutil -m cp -r ~/tensorflow_datasets/<name_of_your_dataset> gs://xembodiment_data
``` 
This will upload all data using multiple threads. If your internet connection gets interrupted anytime during the upload
you can just rerun the command and it will resume the upload where it was interrupted. You can verify that the upload
was successful by inspecting the bucket [here](https://console.cloud.google.com/storage/browser/xembodiment_data).

The last step is to commit all changes to this repo and send Karl the link to the repo.

**Thanks a lot for contributing your data! :)**
