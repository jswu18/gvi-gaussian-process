# Gaussian Wasserstein Inference in Function Spaces

Leveraging Generalised Variational Inference (GVI) to build a loss objective with the Wasserstein distance for image classifier sparse Gaussian Processes constructed with NNGP infinite-width kernels.

For a quick overview of GVI, see <a href="https://jswu18.github.io/posts/2023/07/generalised-variational-inference/">my blog post</a>.

Gaussian Wasserstein Inference is developed in <a href="https://arxiv.org/pdf/2205.06342.pdf">this paper</a> by Veit Wild.

To get set up:

1. Install `poetry`

```shell
pip install poetry
```

2. Install dependencies

```shell
poetry install
```

## Sample Regression Curves

<p align="middle">
  <img src="experiments/regression/outputs/curve0/reference.png" width="49%" />
  <img src="experiments/regression/outputs/curve0/tempered.png" width="49%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve1/reference.png" width="49%" />
  <img src="experiments/regression/outputs/curve1/tempered.png" width="49%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve2/reference.png" width="49%" />
  <img src="experiments/regression/outputs/curve2/tempered.png" width="49%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve3/reference.png" width="49%" />
  <img src="experiments/regression/outputs/curve3/tempered.png" width="49%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve4/reference.png" width="49%" />
  <img src="experiments/regression/outputs/curve4/tempered.png" width="49%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve5/reference.png" width="49%" />
  <img src="experiments/regression/outputs/curve5/tempered.png" width="49%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve6/reference.png" width="49%" />
  <img src="experiments/regression/outputs/curve6/tempered.png" width="49%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve7/reference.png" width="49%" />
  <img src="experiments/regression/outputs/curve7/tempered.png" width="49%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve8/reference.png" width="49%" />
  <img src="experiments/regression/outputs/curve8/tempered.png" width="49%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve9/reference.png" width="49%" />
  <img src="experiments/regression/outputs/curve9/tempered.png" width="49%" />
</p>

