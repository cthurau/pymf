## PyMF - Python Matrix Factorization Module

### What is PyMF?
Python Matrix Factorization (PyMF) is a module for several constrained/unconstrained matrix factorization (and related) methods for both sparse and dense matrrices (for a brief introduction to factorization of gigantric matrices have a look at a <a href="https://sites.google.com/site/factorizinggiganticmatrices/">tutorial</a> we gave at ECML-PKDD 2011). It requires cvxopt, numpy, and scipy.

PyMF currently includes the following methods:

* Non-negative matrix factorization (NMF) [three different optimizations used]
* Convex non-negative matrix factorization (CNMF)
* Semi non-negative matrix factorization (SNMF)
* Archetypal analysis (AA)
* Simplex volume maximization (SiVM) [and SiVM for CUR, GSAT, ... ]
* Convex-hull non-negative matrix factorization (CHNMF)
* Binary matrix factorization (BNMF)
* Singular value decomposition (SVD)
* Principal component analysis (PCA)
* K-means clustering (Kmeans)
* C-means clustering (Cmeans)
* CUR decomposition (CUR)
* Compaxt matrix decomposition (CMD)

### Usage
Given a dataset, most factorization methods try to minimize the Frobenius norm <code>| data - W*H|</code> by finding a suitable set of basis vectors <code>W</code> and coefficients H. The syntax for calling the various methods is quite similar. Usually, one has to submit a desired number of basis vectors and the maximum number of iterations. For example, applying NMF to a dataset data aiming at 2 basis vectors within 10 iterations works as follows:

```python
>>> import pymf
>>> import numpy as np
>>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
>>> nmf_mdl = pymf.NMF(data, num_bases=2, niter=10)
>>> nmf_mdl.initialization()
>>> nmf_mdl.factorize()
```

The basis vectors are now stored in <code>nmf_mdl.W</code>, the coefficients in <code>nmf_mdl.H</code>. To compute coefficients for an existing set of basis vectors simply copy W to nmf_mdl.W, and set compW to False:

```python
>>> data = np.array([[1.5], [1.2]])
>>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
>>> nmf_mdl = pymf.NMF(data, num_bases=2, niter=1, compW=False)
>>> nmf_mdl.initialization()
>>> nmf_mdl.W = W
>>> nmf_mdl.factorize()
```

By changing pymf.NMF to e.g. pymf.AA or pymf.CNMF Archetypal Analysis or Convex-NMF can be applied. Some methods might allow other parameters, make sure to have a look at the corresponding <code>>>>help(pymf.AA)</code> documentation. For example, CUR, CMD, and SVD are handled slightly differently, as they factorize into three submatrices which requires appropriate arguments for row and column sampling.

### Very large datasets
For handling larger datasets pymf supports hdf5 via h5py. Usage is straight forward as h5py allows to map large numpy matrices to disk. Thus, instead of passing data as a np.array, you can simply send the corresponding hdf5 table. The following example shows how to apply pymf to a random matrix that is entirely stored on disk. In this example the dataset does not have to fit into memory, the resulting low-rank factors <code>W,H</code> have to.

```python
>>> import h5py
>>> import numpy as np
>>> import pymf
>>>
>>> file = h5py.File('myfile.hdf5', 'w')
>>> file['dataset'] = np.random.random((100,1000))
>>> sivm_mdl = pymf.SIVM(file['dataset'], num_bases=10)
>>> sivm_mdl.factorize()
```

If the low-rank matrices <code>W,H</code> also do not fit into memory, they can be initialized as a h5py matrix.

```python
>>> import h5py
>>> import numpy as np
>>> import pymf
>>>
>>> file = h5py.File('myfile.hdf5', 'w')
>>> file['dataset'] = np.random.random((100,1000))
>>> file['W'] = np.random.random((100,10))
>>> file['H'] = np.random.random((10,1000))
>>> sivm_mdl = pymf.SIVM(file['dataset'], num_bases=10)
>>> sivm_mdl.W = file['W']
>>> sivm_mdl.H = file['H']
>>> sivm_mdl.factorize()
```

Please note that currently not all methods work well with hdf5. While they all accept hdf5 input matrices, they sometimes lead to very high memory consumption on intermediate computation steps. This is difficult to avoid unless we switch to a completely disk-based storage.
