# Clustering neurons with graph-based methods

A summer project that uses spectral clustering to confirm the findings on families of neurons.<br>
This is based on a Nature article by Scala, F., Kobak, D., Bernabucci, M. *et al.* The original article can be found [here](https://rdcu.be/cmgFA).

## The dataset

The dataset in this repo is from the [original work](https://github.com/berenslab/mini-atlas). Additional instructions on downloading data can be found in the Jupyter notebooks.

## How to run the analysis

### Preprocessing

1. Run ```allen-data-preprocess-mod.ipynb``` to preprocess the Allen Institute data. This notebook contains instructions on additional files you will need.
2. Run ```patch-seq-data-load.ipynb``` to load all the data and package together into a Python object.
3. Run ```ttype-assignment.ipynb``` to assign all cells to the t-types. This also creates a lot of image files.
4. Run ```preprocess-ephys-files-mod.ipynb``` to extract the electrophysiological features and create ```three_traces.pickle```. This notebook contains instructions on additional files you will need.

### Analysis

1. To get a "feel" of how the clustering methods work, run ```scikit_tsne_viplamp.ipynb``` to test [sci-kit learn clustering methods](https://scikit-learn.org/stable/modules/clustering.html) on the t-SNE representations of the original article, Figure 1c. If you find the article, dataset, or figure confusing, refer to ```about_figure_1c.ipynb``` - this breaks things down a bit.
2. From the data science prespective, there seems to be some evaluations missing for the kNN clusterings in the original article. ```revisit-confusion-matrices.ipynb``` breaks down what is done in the original article and then adds some figures to confirm if the conclusion of the article is true.

## About the borrowed datasets and the codes

The entire dataset and codes listed below (some with slight modifications) are borrowed from the original work.

* ```allen-data-preprocess-mod.ipynb``` (this one is slightly modified from original)
* ```rnaseqTools.py```
* ```patch-seq-data-load.ipynb```
* ```ttype-assignment.ipynb```
* ```ttype-coverage-mod.ipynb``` (modified from original)
* ```preprocess-ephys-files-mod.ipynb``` (modified from original)
