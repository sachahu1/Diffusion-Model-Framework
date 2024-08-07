# Diffusion Model Playground

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/sachahu1/Diffusion-Model-Framework/run-tests.yaml?branch=main&label=Tests)
![GitHub Release](https://img.shields.io/github/v/release/sachahu1/Diffusion-Model-Framework)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/diffusion-model-framework)
![GitHub Repo stars](https://img.shields.io/github/stars/sachahu1/Diffusion-Model-Framework)

This is a simplified framework to train and run diffusion models.
![](https://github.com/sachahu1/Diffusion-Model-Framework/blob/main/assets/sampling-small.gif?raw=true)

## Getting Started
To get started, simply install the python package from PyPI:
```shell
pip install diffusion-model-framework
```
Then, you're ready to start setting up your own training process. For help 
getting started with that, have a look at some of the [examples](https://github.com/sachahu1/Diffusion-Model-Framework/blob/main/examples).

### Installing the package
Thanks to poetry, installing this package is very simple and can be done in a single command. Simply run:
```shell
poetry install
```
That's it, the package is installed. Move to the next section to learn how to use this package.

## Building from source
### Installing Poetry
This tool uses poetry. If you already have poetry installed,
please skip to the next section. Otherwise, let's first setup poetry.

To install poetry, simply run this command:
```shell
curl -sSL https://install.python-poetry.org | python3 -
```
You can find out more about poetry installation [here](https://python-poetry.org/docs/master/#installation).

That's it, poetry is set up.
### Installing and building locally
Installing the package via poetry is very simple. Simply run:
```shell
poetry install
```
You can now start using the package to train your diffusion model. For example, try running:
```shell
poetry run python3 examples/train_model.py
```

Building a wheel is also possible via:
```shell
poetry build
```

## Building the documentation
First you will need to install the dependency group used for documentation.
```shell
poetry install --with documentation
```
Now you can build the documentation using Sphinx by running:
```shell
poetry run sphinx-build -M html docs/source/ docs/build
```
