# Introduction

This is a simplified framework to train and run diffusion models.
![](https://github.com/sachahu1/Diffusion-Model-Framework/blob/main/assets/sampling-small.gif?raw=true)

## Getting Started
To get started, simply install the python package from PyPI:
```shell
pip install diffusion-model-framework
```
Then, you're ready to start setting up your own training process. For help 
getting started with that, have a look at some of the [examples](Examples).



### Installing the package
Thanks to uv, installing this package is very simple and can be done in a single command. Simply run:
```shell
uv sync
```
That's it, the package is installed. Move to the next section to learn how to use this package.

## Building from source
### Installing uv
This tool uses [uv](https://docs.astral.sh/uv/). If you already have uv installed,
please skip to the next section. Otherwise, run:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```
You can find out more about uv installation [here](https://docs.astral.sh/uv/getting-started/installation/).

That's it, uv is set up.
### Installing and building locally
Installing the package via uv is very simple. Simply run:
```shell
uv sync
```
You can now start using the package to train your diffusion model. For example, try running:
```shell
uv run python3 examples/train_model.py
```

Building a wheel is also possible via:
```shell
uv build
```

## Building the documentation
First you will need to install the dependency group used for documentation.
```shell
uv sync --group documentation
```
Now you can build the documentation using Sphinx by running:
```shell
uv run sphinx-build -M html docs/source/ docs/build
```
