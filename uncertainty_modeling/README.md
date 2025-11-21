# Training and Inference of the models

This part of the repository contains the code for training the models and inferring on the test sets.  
With the repository now focused purely on 2D segmentation, ```main.py``` remains the training entry point and ```test_2D.py``` is the single inference entry point (its CLI helpers live in ```unc_mod_utils/test_utils.py```).  
The following sections will explain the configs for how to start a training and the settings for starting an inference in more detail. Note that it is assumed that you have preprocessed the datasets as described in the respecive ```datasets``` subfolder at this point.


## Running a training

For running a training, execute ```main.py``` with the appropriate configuration for the setup you want to train. For examples on how to configure a training, see the config files in the ```configs``` subfolder. Generally, there are entry files on the top level of this folder, e.g. ```softmax_config.yaml``` which themselves include different datamodules, models, etc., specified in the corresponding subfolders.  
The structure for the entry configuration is like this:

```yaml
defaults:
    - datamodule: <name of datamodule>
    - model: <name of the model>
    # if data augmentations are used
    - data_augmentations: <used data augmentations>

# Save directory for a specific experiment version is made up of save_dir/exp_name/version
exp_name: <name of experiment, mostly prediction model, e.g. "Softmax">
version: <name of the version, usually made up of seed and fold, e.g. fold0_seed123 or additional properties like pretrain epochs (for SSNs) etc. Basically everything that is unique about the experiment version>
save_dir: <base_path/to/save/experiments>
# datasets should be preprocessed as described in datasets subfolder
data_input_dir: <base_path/to/data>
seed: <seed for experiment>

# training params
max_epochs: <number of epochs to train>
batch_size: <batch size>
learning_rate: <learning rate>
weight_decay: <weight decay>
gpus: <gpus to use>

logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ${save_dir}
  name: ${exp_name}
progress_bar:
  _target_: pytorch_lightning.callbacks.TQDMProgressBar
  refresh_rate: 10

# optional, see gta config for example
optimizer:
  _target_: <optimizer to use>
  lr: ${learning_rate}
  weight_decay: ${weight_decay}
  # ... (additional params)
lr_scheduler:
  _target_: <lr scheduler to use>
  # ... (additional params)

### Checkpoint scheduling and EMA weights

The training configs additionally expose a ``ckpt_save_freq`` block and EMA controls:

```yaml
track_ema_weights: true   # enable exponential moving average tracking of model weights
ema_decay: 0.999          # decay factor used for the EMA update

ckpt_save_freq:
  use_linear_saving: true       # set to true to store checkpoints every ``linear_freq`` epochs
  use_exponential_saving: false # alternatively enable exponentially spaced checkpoints
  only_small_ckpts: true        # save weights-only checkpoints when true
  linear_freq: 25               # (linear) save at 25, 50, 75, ...
  exponent_base: 1.5            # (exponential) growth factor between checkpoints
  exponential_start: 10         # (exponential) first epoch to snapshot
  end: ${trainer.max_epochs}    # last epoch considered when building the schedule
```

Only one of the linear or exponential schemes can be enabled at a time. Linear schedules save at regular intervals (e.g. 25, 50, 75, ...), whereas exponential schedules create a logarithmic spacing (e.g. 10, 20, 40, 80, ...). The configured checkpoints are stored alongside the standard Lightning checkpoints; the final training checkpoint remains handled by Lightning itself. When ``only_small_ckpts`` is ``true`` the scheduled checkpoints only contain the model weights (plus EMA weights when tracking is enabled) which are sufficient for inference.
```

The datamodule and model configs are really dependent on the implementations of the datamodule and model they are instantiating. For examples, look in the corresponding subfolders.

## Running inference

For running inference after training, execute ```test_2D.py``` with the appropriate checkpoint arguments (see ```python uncertainty_modeling/test_2D.py -h``` for the up-to-date interface that is generated from ```unc_mod_utils/test_utils.py```). 
The only parameter that is not optional, is ```checkpoint_paths``` where you define the checkpoint based on which you want to perform the inference. For ensemble inference, specify all the paths that you want to use for inference.  
Another parameter that you mostly need to specify are the number of predictions that you want to make (```--n_pred```) if you want to sample multiple output segmentations.  
All other parameters like the dataset location etc. can be inferred from the checkpoint itself, although you may want to change them, e.g. if you train and infere on different machines or if your testset is located in a special directory.


After inference, the results are stored in a subfolder called ```test_results```, which has the following structure:


    test_results
    ├── <version>
        ├── aleatoric_uncertainty
        ├── epistemic_uncertainty
        ├── pred_entropy
        ├── gt_seg
        ├── input
        ├── pred_prob
        ├── pred_seg
        ├── metrics.json


This means that during testing, besides the segmentation prediction, also the uncertainty maps are generated and a metrics file which contains metrics regarding the segmentation performance (Dice) and ambiguity modeling (GED). Note that the aleatoric_uncertainty and epistemic_uncertainty directory only exist for methods that sample multiple segmentations, i.e. not for the plain softmax prediction model. Further, for the GTA5/Cityscapes dataset, the input and the predicted probabilities (pred_prob) are not saved for each experiment. For further analyzing and processing the results, see the ```evaluation``` subfolder of this repository.