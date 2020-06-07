# Estimation maximisation with gaussian mixture model (EM GMM)
EM GMM is an unsupervise learning algorithm by clustering data to different guassian model.

## Package
You will need the following package to run the model
- numpy
- scipy
- matplotlib

## Usage
Import the model.
```python
from EM_GMM import GMM_EM
```
Initialize the model and define the number of cluster.
```python
model = GMM_EM(k)
```
Train the model with your data and also the number of iteration. You can also plot the training performance by setting plot = True.
```python
model.train(data, epoch, plot = True) # default is True
```

<img src="https://github.com/chanyikchong/EM_GMM/blob/master/scatter.png" width="300"><br/>
You can get the negative log likelihood (do not use the likelihood function).
```python
model.li
#or get the final negative log likelihood
model.li[-1]
```
You can also plot the negative log likelihood in all training steps
```python
model.plot_li()
```
<img src="https://github.com/chanyikchong/EM_GMM/blob/master/nll.png" width="300"><br/>

## Issues
Sometimes the model may stop by a sigular matrix problem. This may due to unluckly initialization (The means and weights are randomly initialized) or unfitted number of cluster.

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
