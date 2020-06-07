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
!(https://github.com/chanyikchong/EM_GMM/blob/master/scatter.png)
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
