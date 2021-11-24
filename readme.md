# EasyPred: simply track your predictions
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub license](https://badgen.net/github/license/FilippoPisello/EasyPred)](https://github.com/FilippoPisello/EasyPred/blob/main/LICENSE)
## What is it?
EasyPred is a Python package that allows to easily store, investigate, assess and compare the predictions obtained through your Machine Learning models.

The package allows to create different types of model-agnostic prediction objects simply by passing the real and fitted data. These objects have properties and methods that return various accuracy and error metrics.

Why EasyPred can be useful:
- **All-in-one bundle**: having data and accuracy metrics in a single object means less stuff you need to keep an eye on
- **Minimize code redundancy**: pass the data once and get all the information and metrics you want
- **Easy and flexible comparison**: create the predictions first and then decide what to compare. Changed your mind? The object is there, simply access another method

## Quick Start
### Installation
You can install EasyPred via `pip`
```
pip install easypred
```
Alternatively, you can install EasyPred by cloning the project to your local directory
```
git clone https://github.com/FilippoPisello/EasyPred
```
And then run `setup.py`
```
python setup.py install
```
### Usage
To be added