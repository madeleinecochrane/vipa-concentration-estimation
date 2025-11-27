# Molecular concentration estimation from laser absorption spectroscopy.

This code provides support to predict the concentration of multiple molecular compounds through machine learning and laser spectroscopy.
Concentrations can be predicted from absorption spectra and the image output of a Virtually Imaged Phased Array (VIPA) spectrometer.
More information on our experiments can be found in our paper: https://pubs.acs.org/doi/10.1021/acsomega.5c07120

## User guide

### Installation

To install our code, please clone this repository. 
Dependencies are listed in requirements.txt which can be used to set up your Python environment.

### Directory set-up

In brief, there are five top level directories in the file structure: data, dataset, log, modelling, toolbox.

**Data** is where the experimental data from each training run goes. 
This includes model checkpoint files, .mat files containing results, and loss and accuracy plots.

**Dataset** is where your dataset should go. 
You should create a directory underneath the dataset directory for your datasets e.g. VIPA-images.
We would also recommend creating a directory named processed within the directory for your dataset, to store things like training, validation and test sets.

**Log** is where log files from each training run will be saved.

**Modelling** stores code for models, datasets and training loops.

**Toolbox** contains general utility code.

### Customisation for your dataset

Unfortunately, we are unable to publicly share the dataset used in our experiment. 
In order to set up the code to run with your data, follow the steps below.

#### Dataset

The first thing you will need to do is create a custom dataset class in the [modelling/datasets](modelling/datasets) directory.
We have provided samples of our own dataset files for you to use as guides: [modelling/datasets/IsotopicCo2Spectra.py](modelling/datasets/IsotopicCo2Spectra.py) and [modelling/datasets/IsotopicCo2Images.py](modelling/datasets/IsotopicCo2Images.py).

The easiest way to do create your own class is to inherit from the existing `Spectroscopy` class in [modelling/datasets/Spectroscopy.py](datasets/Spectroscopy.py).
You will need to provide your own `__init__` and `_extract` methods. 

For compatibility with the existing `Spectroscopy` class, your `_extract` method should save your training, validation and test sets in three .pkl files: train.pkl, test.pkl and held-out-test.pkl.
Each .pkl file should contain two lists - data and labels.
For spectral analysis, the data should be a list of absorption spectra contained in one-dimensional numpy arrays.
For image analysis, the data should be a list of ages stored as 3 x w x h numpy arrays, where w is the image width and h the image height.
For both types data formats, the labels should be a list of numpy arrays containing the labels.
Labels should be provided in the following order: regression tasks, binary classification tasks and classification tasks.

If you would prefer to create your own dataset, without the `Spectroscopy` class, your custom dataset class will need to provide `__init__`, `__len__` and `__getitem__` methods.
The `__len__` method should return the number of samples in the dataset. 
Note that this length should be for the training, validation and test datasets, rather than the entire dataset combined.

The `__getitem__` method should accept idx as a parameter, which indicates the index of the requested item.
The method should return a tuple containing the source data (i.e. the data element directly from your .pkl file or similar), the data (after any transpositions or normalizations have been applied) and the label.
At minimum, you will need to convert the data and the source data from numpy arrays into Pytorch tensors. 
If you want to use the transformer model, you'll also need to reshape the tensor into the correct 2-dimensional shape for the model.

#### Modifying train.py

Once you have created your dataset, you will also need to set up `train.py` to use it.
You'll need to add your custom dataset class to the import statements at the top of the file.

After that you'll need to set up your transforms around line 30 of train.py.
The code should follow the template below:
```python
elif '<REPLACE WITH NAME OF THE FOLDER THAT STORES YOUR DATASET UNDER THE DATASET DIRECTORY>' in args.data_path: 
    print('Experiments for <REPLACE WITH YOUR EXPERIMENT NAME>')
    args.input_size = 3000 # replace 3000 with the total number of samples in your dataset
    train_transform = None # leave as None for spectra, replace with transforms.ToTensor() for images
    val_transform = None # leave as None for spectra, replace with transforms.ToTensor() for images
```

You'll also need to set up your tasks and data loader around line 70.
The code should follow the template below:
```python
if '<REPLACE WITH NAME OF THE FOLDER THAT STORES YOUR DATASET UNDER THE DATASET DIRECTORY>' in args.data_path:
    config['regression_tasks'] = ['reg1', 'reg2'] # replace with a list of the names of each molecule concentration you want to estimate or dict() if no regression tasks
    config['binary_cls_tasks'] = ['12CO2', '13CO2'] # replace with a list of molecules you want to identify presence of or dict() if no binary classification tasks
    config['cls_tasks'] = dict()
    train_dataset = IsotopicCo2Spectra(train=True, transform=train_transform, root=args.data_path, config=config) # replace IsotopicCo2Spectra with your custom dataset class
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    val_dataset = IsotopicCo2Spectra(train=False, transform=val_transform, root=args.data_path, config=config) # replace IsotopicCo2Spectra with your custom dataset class
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
```

#### Creating a run file for training and validation

We recommend creating a bash script to run your files. This will make loading configuration settings easier.
We have provided an example run file in `run_spec.sh` which you can modify for your own purposes.

The following table summarises the configuration settings you can use to calibrate the code for your own needs.

| Setting                      | Type   | Description                                                                                                                                   | Default value          |
|------------------------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| data_path                    | string | Absolute file path to your dataset                                                                                                            | dataset/IsotopicCo2Spectra |
| batch_size                   | int    | Batch size for model training                                                                                                                 | 32                     |
| num_workers                  | int    | Number of subprocesses to use for loading data                                                                                                | 0                      |
| num_epochs                   | int    | Number of epochs to use for training                                                                                                          | 95                     |
| save_path                    | string | Absolute file path to directory for saving model and results                                                                                  | data/results           |
| save_interval                | int    | Number of epochs between saving model checkpoints and results                                                                                 | 10                     |
| cam_interval                 | int    | Number of epochs between saving class activation maps                                                                                         | -1                     |
| display_interval             | int    | Number of tested batches between displaying results                                                                                           | 100                    |
| reload_path                  | string | Absolute file path to checkpoint for reloaded model                                                                                           | NA                     |
| reload_from_checkpoint       | string | Boolean indicating whether model should be loaded from checkpoint                                                                             | False                  |
| seed                         | int    | Seed used to initialise randomisers                                                                                                           | 0                      |
| optimizer                    | string | Optimizer to be used for learning. Valid options are 'sgd' for stochastic gradient descent and 'adam' for Adam                                | sgd                    | 
| learning_rate                | float | Initial learning rate                                                                                                                         | 1e-3                   |
| learning_rate_step_size      | int | Number of epochs between learning rate steps                                                                                                  | 30                     |
| learning_rate_gamma          | string | Learning rate decay                                                                                                                           | 0.1                    |
| network                      | string | Model type. Valid options are 'mlp', 'cnn', 'transformer', resnet35', 'resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19', swin_v2_t', 'swin_v2_b' | mlp                    | 
| pretrained                   | string | Boolean indicating whether to use pretrained weights provided by Pytorch for models.                                                          | True                   |
| num_training_image_per_class | int | Number of samples per class in the training set                                                                                               | 5                      |
| num_transformer_bins         | int | Number of bins used to divide data into two-dimensional tensor for transformer model (see paper for more details)                             | 100                    |
| limited_training_data        | string | Boolean flag indicating whether limited training data is being used to test model's performance on a subset of data                           | False                  |
| num_training_images          | int | Number of samples used in the training data set                                                                                               | 2400                   |
| num_translations             | int | Number of shifts to use for the shifted spectra data augmentation                                                                             | 0                      |
| noise_factor                 | float | Standard deviation of Gaussian distribution for noisy spectra data augmentation                                                               | 0.0                    |
| h1                           | int | Number of neurons in hidden layer 1 (for MLP only)                                                                                            | 5000                   |
| h2                           | int | Number of neurons in hidden layer 2 (for MLP only)                                                                                            | 2000                   |
| h3 | int | Number of neurons in hidden layer 3 (for MLP only)                                                                                            | 1000                   |
| d1 | float | Dropout to be used after hidden layer 1 (for MLP only) | 0                      |
| d2 | float | Dropout to be used after hidden layer 2 (for MLP only) | 0                      |
| d3 | float | Dropout to be used after hidden layer 3 (for MLP only) | 0 |
| verbose | string | Boolean indicating whether to print results and status information during training | True |

#### Testing on the test set

A script to evaluate MAE for regression tasks (i.e. concentration estimation) is provided in the `test_and_experimental_scripts` directory.
This script is designed to work with model checkpoints that have been obtained by running the `train.py` script.

To customise the provided script for your own model and tasks, edit the `config`, `network`, `model_path`, and `data_path` variables.
`network` should be a string like `'transformer'` that specifies the network you want to use.
`model_path` and `data_path` should also be strings with absolute paths to the directory containing the model checkpoint and the test pickle file, respectively.
If analysing VIPA images, you will also need to update the parameters in the `normalization` variable declared on line 70.

To run the test script, use the following command in the `concentration-estimation-code` root directory:
```bash
python3 -m test_and_experimental_scripts.calculate_mae_for_regression_tasks_on_test_set
```

#### Generating CAM images

The CAM Runner class provided in `modelling` is designed to generate CAM images for trained models.
The `test_and_experimental_scripts` directory contains a script that demonstrates how to use this class.
CAM generation is only supported for VIPA image analysis, not spectra.

In order to customise the provided script for your own model and tasks, you will need to edit the following values:
1. Replace `network` on line 21 with your desired network.
2. Replace the `save_path` on line 22 with an absolute path to the directory containing your model checkpoint. Note: this is also the directory where the CAMs will be saved.
3. Replace the string in `model_path` on line 23 with the name of your model checkpoint.
4. Replace the mean and std values in `normalization` on line 28.
5. Set `root` on line 31 to be the absolute path to your dataset directory.

If you want to run the CAM over different labels, as we did in our paper, you will need to update the separate data by label function.
Our function has been left at the top of the file as a guide.
If you just want an overall CAM for all values, you can comment out or delete lines 37-39.

To run the cam script, use the following command in the `concentration-estimation-code` root directory:
```bash
python3 -m test_and_experimental_scripts.get_average_cams
```

This code will save a set of CAM images as .jpg files in the specified `save_path` (which should also be the directory where your model is located).

## Contact

For any issues, questions, or requests for further information, please email [madeleine.cochrane@adelaide.edu.au](mailto:madeleine.cochrane@adelaide.edu.au)

