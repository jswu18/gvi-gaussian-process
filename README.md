# GVI-GP: Generalised Variational Inference for Gaussian Process Learning

This project leverages Generalised Variational Inference (GVI) to provide a highly flexable framework for constructing and learning Gaussian Processes (GP's) in the contexts of both regression and classification tasks. This framework allows one to build GVI loss objectives for learning GP's by combining any valid empirical risk with any valid regulariser. Regularisers are defined with respect to a distance metric for push-forward Gaussian measures in function spaces (i.e. the Wasserstein Metric on Hilbert Spaces). Specifically, the experiments in this project will explore the advantages of this framework for learning sparse GP's (sGP's) beyond their traditional stochastic variational GP (svGP) construction from Titsias 2009, which has restrictive conditions on the mean and kernel formulations. This includes neural network mean functions and extensions of the svGP kernel approach to new parameterisations. For a quick overview of GVI, see my <a href="https://jswu18.github.io/posts/2023/07/generalised-variational-inference/">blog post</a>. This project is built on the work of Gaussian Wasserstein Inference (GVI with the Wasserstein metric as the regulariser) developed in <a href="https://arxiv.org/pdf/2205.06342.pdf">this paper</a> by Veit Wild.

To get set up:

1. Install `poetry`

```shell
pip install poetry
```

2. Install dependencies

```shell
poetry install
```