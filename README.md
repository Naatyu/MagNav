Magnetic Navigation
==============================

GPS navigation is very popular nowadays, but the problem is that the signal can be jammed. It is therefore necessary to implement methods with the same advantages, i.e:
-Available in any weather
-Available at any location
-Available 24 hours a day and 7 days a week
But which cannot be jammed. This is where magnetic navigation comes in. In addition to having the advantages of GPS, magnetic navigation is much more difficult to jam or impossible in the air. 

However, there is a current blocking element, the magnetic disturbance of the carrier. In order to navigate by means of the magnetic field of the earth, measurements of the magnetic field are made from the earth and a map of magnetic anomalies is referred to. However, the carrier from which the measurements are made emits magnetic disturbances. In order to obtain good measurements, it is necessary to be able to remove the carrier disturbance from the magnetometer measurements. 
There are currently some techniques such as placing the magnetometer on a pole at about 2-3 meters from the aircraft and then perform a Tolles-Lawson calibration. This is very impractical and imposes constraints. 
This project aims to explore new solutions to use sensors placed in the aircraft. We are using deep learning based techniques.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
