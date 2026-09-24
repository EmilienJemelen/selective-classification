Aim of this package: provide tools to control broad error types in Selective Classification.

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
│   ├── CHEST/            # NIH ChestX-ray14 binary classification (consolidation detection)
│   │   ├── run_DenseNet.ipynb
│   │   ├── individual_control.ipynb
│   │   ├── joint_control.ipynb
│   │   └── sgp_set_densenet_SR # dataset with confidence, from run_DenseNet notebook
│   ├── CIFAR/           # CIFAR-10 binary classification (airplane detection)
│   │   ├── train_cnn.ipynb
│   │   ├── train_resnet.ipynb
│   │   ├── individual_control_{cnn,resnet}.ipynb
│   │   ├── joint_control_{cnn,resnet}.ipynb
│   │   └── sgp_set_{{cnn,cnn_MCD},resnet} # dataset with confidence, from train_{cnn, resnet} notebook
│   ├── DIABETES/         # UCI Diabetes 130-US hospitals (30-day readmission prediction)
│   │   ├── train_trees.ipynb
│   │   ├── individual_control.ipynb
│   │   ├── joint_control.ipynb
│   │   └── sgp_set_tabular_SR # dataset with confidence, from train_trees notebook
│   └── WSI/              # Whole Slide Image (tumor detection) experiments
│       ├── train_cnn.ipynb
│       ├── individual_control_cnn.ipynb
│       ├── joint_control_cnn.ipynb
│       └── sgp_set_{cnn,cnn_MCD}
├── other_ntbks/          # Supplementary analyses
|   ├── comparison.ipynb
│   ├── params_impact.ipynb
│   ├── runtime_study.ipynb
|   ├── failure_rates_control.ipynb
│   ├── exec_times_res # notebook results
|   ├── failure_rates # notebook results
│   └── params_impact_res # notebook results
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

CHEST (NIH ChestX-ray14 binary classification: consolidation vs. rest)
* Model: DenseNet-121 from _torchxrayvision_, pretrained on CheXpert (not trained by us, inference only)
* Inference notebook:
    - experiments/CHEST/run_DenseNet.ipynb
* Data: NIH images to download from https://nihcc.app.box.com/v/ChestXray-NIHCC

DIABETES (UCI Diabetes 130-US hospitals: 30-day readmission vs. rest)
* Model: gradient-boosted trees (scikit-learn _HistGradientBoostingClassifier_, predictions averaged over 5 seeds)
* Training notebook:
    - experiments/DIABETES/train_trees.ipynb
* Data: fetched automatically via _ucimlrepo_ (id 296) if no local CSV is found

Trained models weights are in the _/models_weights/_ folder.

## Experiments

__Individual metric control__
* _individual_control*.ipynb_ notebooks in _CIFAR/_, _WSI/_, _CHEST/_ and _DIABETES/_
__Joint metric control__
* _joint_control*.ipynb_ notebooks in _CIFAR/_, _WSI/_, _CHEST/_ and _DIABETES/_