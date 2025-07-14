# k-SubMix: Common Subspace Clustering on Mixed-type Data

Here, you can find the source code of the 'k-SubMix: Common Subspace Clustering on Mixed-type Data' paper from the ECML-PKDD 2023.

## Preparation

An installation of python3 is required!

Furthermore, following python packages must be installed:
- numpy
- pandas
- scipy
- scikit-learn
- itertools
- seaborn
- matplotlib
- timeit
- os
- kmodes

## Data sets

The used data sets (exp1, exp2 and S3) were peprocessed by a 0-1 min-max scaling and saved to the Dataset folder.

## Execute

Everything is prepared in the main.py file.
To set the input parameters open main.py and set the parameter "dataset", "gammaList" and "number_of_iterations_kSubMix"
```python 
from dataset_config import *
import numpy as np
if __name__ == "__main__":
    """_________________USER INPUT____________________"""
    """SET DATASET, GAMMALIST AND NUMBER OF ITERATIONS"""
    """_______________________________________________"""
    dataset = exp1  #Choose from one of the datasets as defined in dataset_config.py
    gammaList=[0.1,1,0.1] #Choose gamma parameter (single or multiple) to regulate the trade-off between numerical and categorical costs, set range and step size
    number_of_iterations_kSubMix = 10 #Number of k-SubMix iterations
    """____________________________________________________"""
    """________________END USER INPUT______________________"""
```

You can now run the code by executing:

`python main.py`

or

`python3 main.py`


