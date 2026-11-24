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
The base install only has what the models and samplers need. Extras add more:

| Extra | Adds | For |
| --- | --- | --- |
| `inference` | click, hydra | the `diffusion-models infer` CLI |
| `training` | `inference` + tensorboard, transformers | training |
| `evaluation` | `inference` + torchmetrics | FID / KID |

```shell
pip install "diffusion-model-framework[inference]"
```
Then, you're ready to start setting up your own training process. For help 
getting started with that, have a look at some of the [examples](https://github.com/sachahu1/Diffusion-Model-Framework/blob/main/examples).

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
uv run diffusion-models train diffuser
```

Tab completion for commands, config groups (`model=<TAB>`), presets
(`+config=<TAB>`) and config keys is available for bash, zsh and fish. For bash:
```shell
eval "$(uv run diffusion-models completion bash)"   # add to ~/.bashrc
diffusion-models configs list diffuser               # groups, options, presets
```
Completion reads a cached snapshot of the registered configs (in
`~/.cache/diffusion-models`), rebuilt automatically whenever the code changes, so
it stays fast.

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
