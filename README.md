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

### Curve 0
<p align="middle">
  <img src="experiments/regression/outputs/curve0/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve0/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve0/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve0/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve0/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 1
<p align="middle">
  <img src="experiments/regression/outputs/curve1/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve1/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve1/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve1/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve1/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 2
<p align="middle">
  <img src="experiments/regression/outputs/curve2/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve2/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve2/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve2/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve2/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 3
<p align="middle">
  <img src="experiments/regression/outputs/curve3/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve3/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve3/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve3/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve3/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>


### Curve 4
<p align="middle">
  <img src="experiments/regression/outputs/curve4/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve4/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve4/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve4/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve4/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 5
<p align="middle">
  <img src="experiments/regression/outputs/curve5/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve5/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve5/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve5/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve5/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 6
<p align="middle">
  <img src="experiments/regression/outputs/curve6/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve6/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve6/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve6/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve6/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 7
<p align="middle">
  <img src="experiments/regression/outputs/curve7/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve7/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve7/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve7/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve7/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 8
<p align="middle">
  <img src="experiments/regression/outputs/curve8/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve8/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve8/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve8/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve8/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 9
<p align="middle">
  <img src="experiments/regression/outputs/curve9/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve9/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve9/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/outputs/curve9/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/outputs/curve9/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>