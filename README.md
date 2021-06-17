# Clustering neurons with graph-based methods

A summer project that clusters cells from the motor cortex of mice.<br>
This is based on a Nature article by Scala, F., Kobak, D., Bernabucci, M. *et al.* The original article can be found [here](https://rdcu.be/cmgFA).

## The dataset

The dataset in this repo is from the [original work](https://github.com/berenslab/mini-atlas). Additional instructions on downloading data can be found in the Jupyter notebooks.

## How to run the analysis

1. Run ```allen-data-preprocess-mod.ipynb``` to preprocess the Allen Institute data. This notebook contains instructions on additional files you will need.
2. Run ```patch-seq-data-load.ipynb``` to load all the data and package together into a Python object.
3. Run ```ttype-assignment.ipynb``` to assign all cells to the t-types. This also creates a lot of image files.

## About the dataset and the code

The entire dataset and codes listed below (some with slight modifications) are borrowed from the original work.

* ```allen-data-preprocess-mod.ipynb``` (this one is slightly modified from original)
* ```rnaseqTools.py```
* ```patch-seq-data-load.ipynb```
* ```ttype-assignment.ipynb```
