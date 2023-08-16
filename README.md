# GVI-GPs: Generalised Variational Inference for Gaussian Process Learning

This project leverages Generalised Variational Inference (GVI) to provide a highly flexible framework for constructing and learning Gaussian Processes (GP's) in the contexts of both regression and classification tasks. The experiments in this project explore the advantages of the GVI framework for learning variational GP approximations beyond the traditional stochastic variational GP (svGP) construction from <a href="http://proceedings.mlr.press/v5/titsias09a.html">Titsias 2009</a>, which has a restrictive GP approximation space that limits their predictive performance. With GVI we can formulate variational GP's with mean functions like neural networks and new variational kernels that were previously unavailable in a traditional Bayesian variational inference framework. The GVI framework allows one to build loss objectives for learning GP's by combining any valid empirical risk (i.e. negative log likelihood) and any valid regulariser (i.e. the Wasserstein Metric on Hilbert Spaces). For a quick overview of GVI, see my <a href="https://jswu18.github.io/posts/2023/07/generalised-variational-inference/">blog post</a>. This project is built on the work of Gaussian Wasserstein Inference (GVI with the Wasserstein metric as the regulariser) developed in <a href="https://arxiv.org/pdf/2205.06342.pdf">this paper</a> by Veit D. Wild.

To get set up:

1. Install `poetry`

```shell
pip install poetry
```

2. Install dependencies

```shell
poetry install
```