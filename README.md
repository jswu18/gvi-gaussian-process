# Generalised Variational Inference for Sparse Gaussian Process Learning

Leveraging Generalised Variational Inference (GVI) to construct loss objectives for sparse Gaussian Process (sGP) learning in the contexts of both regression and classification. This is a flexible framework accomodating any regulariser defined with respect to a distance metric for push-forward Gaussian measures in function spaces (i.e. the Wasserstein Metric on Hilbert Spaces). With GVI we can also flexibily define the sGP. An example is the use of infinite-width convolutional NNGP kernel for image classifier tasks.

For a quick overview of GVI, see <a href="https://jswu18.github.io/posts/2023/07/generalised-variational-inference/">my blog post</a>.

Gaussian Wasserstein Inference (GVI with the Wasserstein metric as the regulariser) is developed in <a href="https://arxiv.org/pdf/2205.06342.pdf">this paper</a> by Veit Wild.

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
  <img src="experiments/regression/toy_curves//outputs/curve0/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve0/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve0/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve0/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve0/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 1
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve1/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve1/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve1/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve1/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve1/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 2
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve2/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve2/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve2/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve2/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve2/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 3
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve3/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve3/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve3/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve3/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve3/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>


### Curve 4
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve4/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve4/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve4/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve4/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve4/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 5
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve5/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve5/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve5/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve5/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve5/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 6
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve6/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve6/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve6/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve6/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve6/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 7
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve7/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve7/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve7/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve7/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve7/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 8
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve8/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve8/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve8/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve8/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve8/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>

### Curve 9
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve9/tempered-PointWiseWassersteinRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve9/tempered-PointWiseBhattacharyyaRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve9/tempered-SquaredDifferenceRegularisation.png" width="49.5%" />
  <img src="experiments/regression/toy_curves//outputs/curve9/tempered-WassersteinRegularisation.png" width="49.5%" />
</p>
<p align="middle">
  <img src="experiments/regression/toy_curves//outputs/curve9/tempered-PointWiseKLRegularisation.png" width="49.5%" />
</p>
