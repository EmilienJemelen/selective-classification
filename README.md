Implementation of the paper: _Beyond Accuracy: Controlling Broad Error Types in Selective Classification_

## Setup and Reproducing Results

* Clone the repository. 
* Create a Python virtual environment. 
* Install dependencies (see _/requirements.txt_)

```bash
pip install -r /path/to/your/local/repo/requirements.txt
```

To reproduce the main results, simply run the notebooks in the _/experiments/_ folder.
* All functions are defined in the _/python_scripts/_ folder.
* Supplementary analyses are provided in the _/other_ntbks/_ folder:
    - parameters impact on bound tightness (_params_impact_) 
    - runtime scaling with dataset size (_runtime_study_) 
    - comparison of our bounds to existing bounds from the literature (_comparison_) 
* Trained models weights are in _/models_weights/_.

## Repository structure

```bash
repo/
├── requirements.txt      # Python dependencies
├── experiments/          # Core experiments
│   ├── CIFAR/           # CIFAR-10 binary classification (airplane detection)
│   │   ├── train_cnn.ipynb
│   │   ├── train_resnet.ipynb
│   │   ├── individual_control_{cnn,resnet}.ipynb
│   │   ├── joint_control_{cnn,resnet}.ipynb
│   │   └── sgp_set_{{cnn,cnn_MCD},resnet}
│   └── WSI/              # Whole Slide Image (tumor detection) experiments
│       ├── train_cnn.ipynb
│       ├── individual_control_cnn.ipynb
│       ├── joint_control_cnn.ipynb
│       └── sgp_set_{cnn,cnn_MCD}
├── other_ntbks/          # Supplementary analyses
|   ├── comparison.ipynb
│   ├── params_impact.ipynb
│   ├── runtime_study.ipynb
│   ├── exec_times_res
│   └── params_impact_res
├── python_scripts/       # Shared utility modules
│   ├── mcdropout.py      # Monte Carlo dropout
│   ├── sgp_utils.py      # SGP utilities
│   ├── preprocessing.py  # Data loading and preprocessing
│   ├── plotting.py       # Visualization
│   └── math_utils.py     # Mathematical helper functions
└── models_weights/       # Trained weights of the pytorch models used in the experiments/
    ├── cnn_cifar_binary_MCD_epoch9.pth      
    ├── cnn_wsi_binary_epoch0.pth      
    └── resnet18_cifar_binary_epoch19.pth   
```

## Training 

CIFAR (CIFAR-10 binary classification: airplane vs. rest)
* Models: 
    - small CNN
    - ResNet-18 
* Training notebooks:
    - experiments/CIFAR/train_cnn.ipynb
    - experiments/CIFAR/train_resnet.ipynb

WSI (Whole Slide Image tumor classification)
* Model: small CNN
* Training notebook:
    - experiments/WSI/train_cnn.ipynb

Trained models weights are in the _/models_weights/_ folder.

## Experiments

__Individual metric control__
* _individual_control_*.ipynb_ notebooks in both _CIFAR/_ and _WSI/_
__Joint metric control__
* _joint_control_*.ipynb_ notebooks in both _CIFAR/_ and _WSI/_
