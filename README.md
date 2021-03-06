Magnetic Navigation (currently in progress)
==============================
Magnetic navigation project by Nathan Laoué


Introduction
------------

GPS navigation is very popular nowadays, but the problem is that the signal can be jammed. It is therefore necessary to implement methods with the same advantages, i.e:
- Available in any weather
- Available at any location
- Available 24 hours a day and 7 days a week

but which cannot be jammed. This is where magnetic navigation comes in. In addition to having the advantages of GPS, magnetic navigation is much more difficult to jam or impossible in the air. 

However, there is a current blocking element, the magnetic disturbance of the carrier. In order to navigate by means of the magnetic field of the earth, measurements of the magnetic field are made from the earth and a map of magnetic anomalies is referred to. However, the carrier from which the measurements are made emits magnetic disturbances. In order to obtain good measurements, it is necessary to be able to remove the carrier disturbance from the magnetometer measurements.<br> 

There are currently some techniques such as placing the magnetometer on a pole at about 2-3 meters from the aircraft and then perform a Tolles-Lawson calibration. This is very impractical and imposes constraints. 
This project aims to explore new solutions to use the sensors placed in the aircraft by exploring new techniques such as deep learning.

Project Data
------------

The dataset is provided by the United States Air Force pursuant to Cooperative Agreement Number FA8750-19-2-1000.<br>
Albert R Gnadt, Joseph Belarge, Aaron Canciani, Lauren Conger, Joseph
Curro, Alan Edelman, Peter Morales, Michael F O’Keeffe, Jonathan Taylor,
and Christopher Rackauckas. [Signal enhancement for magnetic navigation challenge problem](/references/Signal%20Enhancement%20for%20Magnetic%20Navigation%20Challenge%20Problem.pdf). *arXiv e-prints*, pages arXiv–2007, 2020.

The data used for this project comes from an [MIT challenge](https://magnav.mit.edu/). The data can be downloaded [here](https://zenodo.org/record/4271804#.YnWQuIdBxD8). The goal of the challenge is the same as the one of our project but we do not take into account their restrictions on the dataset. A datasheet of the data is available [here](references/Challenge%20problem%20datasheet.pdf)<br>

This dataset was created by [SGL](http://www.sgl.com/). They have made several flights by placing magnetometers in several places of the plane and especially a magnetometer on a pole at the end of the plane which will serve as truth. They also took measurements of various elements of the aircraft such as roll angle and battery voltage.<br>

In this project we use 3 different magnetic anomaly maps. Maps of Renfrew and Eastern area are provided with the dataset. For the map of Canada, it is available from this [link](http://gdr.agg.nrcan.gc.ca/gdrdap/dap/info-eng.php). Also for the world map of magnetic anomalies, it is available from this [link](http://wdmam.org/).<br>

You can also download the IGRF model [here](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-020-01288-x).

How to run the Notebooks ?
------------
To use the same data as in this project, a kaggle dataset has been created. You just have to download the data from [here](https://www.kaggle.com/datasets/beff9edcc2a054c1b022fa08c0e0625f8a30b1e5fe113f704a6590c6a72d3aa0) and paste the dezipped data in ```data/raw```. All that remains is to run the notebooks in order and the additional data will be created automatically. There is also a requirements.txt file, just open a terminal and go to the project folder and do ```pip install requirements.txt```. This will download the packages needed to run the project. It is highly recommended to create a virtual environment beforehand.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README of this project
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details (not implemented)
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries (not implemented)
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc. (not implemented)
    │   └── figures        <- Generated graphics and figures to be used in reporting (not implemented)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported (not implemented)
    ├── src                <- Source code for use in this project. (not implemented)
    │   ├── __init__.py    <- Makes src a Python module (not implemented)
    │   │
    │   ├── data           <- Scripts to download or generate data (not implemented)
    │   │   └── make_dataset.py (not implemented)
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling (not implemented)
    │   │   └── build_features.py (not implemented)
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions (not implemented)
    │   │   ├── predict_model.py (not implemented)
    │   │   └── train_model.py (not implemented)
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations (not implemented)
    │       └── visualize.py (not implemented)
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io (not implemented)
