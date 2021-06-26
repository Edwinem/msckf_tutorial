# MSCKF Tutorial

## Introduction
This project contains a basic Multi-Constraint Kalman Filter(MSCKF) implementation to solve the
visual inertial odometry(VIO) problem. The MSCKF is an extended kalman filter first introduced in
X, and is the main way to solve VIO within the EKF framework.

This project should serve as a tutorial. Hopefully, people can read through the codebase
and learn how an MSCKF works. It is a fairly basic implementation, and lacks some of the more modern upgrades such as
the observability constraints. Plus as it is implemented in Python it lacks the speed necessary to run
this on a actual system. If you do want a useful ,performant MSCKF solution then I recommend
the [OpenVINS](https://github.com/rpng/open_vins) project.

## Getting Started

### Installation
The project is developed with [poetry](https://python-poetry.org/docs/basic-usage/) to control the dependencies.
If possible I recommend to setup the project using it, and run the commands with it.

#### Python Virtual environment setup.

It is assumed you already have setup or can setup a python virtual environment.

Install the project to your virtualenv. I recommend using
```pip install -e .```.

### Running examples.

Download one of the runs from the [Euroc](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) 
dataset in the ASL format, and unzip the compressed file.

Run the VIO algorithm using the following command

```python ./examples/run_on_euroc.py --euroc_folder <PATH_TO_MH2_MAV0> --use_viewer --start_timestamp 1403636896901666560```

The ```start_timestamp``` is needed right now as we don't run an initialization scheme, and Euroc has
some drastic movement in the beginning to initialize the system. It should be set to timestamp where
the drone is about to take off after the bias initialization movement. Here the value ```1403636896901666560``` is for
the MH02 run.


## Primer on Transforms and Notation



## References