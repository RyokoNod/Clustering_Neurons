# Exploring the neuron clusters of Nature articles

A summer project that explores the recent findings on families of neurons.<br>
This is based on a Nature article by Scala, F., Kobak, D., Bernabucci, M. *et al.* The original article can be found [here](https://rdcu.be/cmgFA).

![Image of Yaktocat](https://github.com/RyokoNod/Clustering_Neurons/blob/main/figures/sample-figure.png)

## The dataset

The dataset in this repo is from the [original work](https://github.com/berenslab/mini-atlas). Additional instructions on downloading data can be found in the Jupyter notebooks.

## How to run the analysis

### Preprocessing

1. Before you do anything, clone the FIt-SNE repository at https://github.com/KlugerLab/FIt-SNE and place it at the same directory path as this repo. After cloning, follow the instructions on README to compile everything. Note that you need gfortran and FFTW installed to get the repo set up. If you find the README somewhat unclear, this Kaggle page may help you get started: https://www.kaggle.com/returnofsputnik/mnist-2d-t-sne-with-fit-sne-cpu-only.
2. Run ```allen-data-preprocess-mod.ipynb``` to preprocess the Allen Institute data. **(Update: as of summer 2021, most of this process can be skipped. Find more instrucitons inside the notebook)**
3. Run ```patch-seq-data-load.ipynb``` to load all the data and package together into a Python object.
4. Run ```ttype-assignment.ipynb``` to assign all cells to the t-types. This also creates a lot of image files.
5. Run ```preprocess-ephys-files-mod.ipynb``` to extract the electrophysiological features and create ```three_traces.pickle```. **(Update: as of summer 2021, most of this process can be skipped. Find more instrucitons inside the notebook)**

### Analysis

1. To get an idea of what is going on in the article, read the summary slides in `article-summaries`.`summary_scala.pdf` is the summary of the original Nature article. and `summary_yao.pdf` is a summary of an earlier article that identified the cell types.
2. To see what happens when you run different clustering methods on transcriptomic data, run ```codes/scikit_clusterings/scikit_tsne_viplamp.ipynb```. This tests [sci-kit learn clustering methods](https://scikit-learn.org/stable/modules/clustering.html) on the t-SNE representations of the original article, Figure 1c. If you find the article, dataset, or figure confusing, refer to ```about_figure_1c.ipynb``` - this breaks things down a bit.
3. From the data science perspective, there needs to be more on confusion matrices in the original article. ```codes/confusion/matrices/revisit-confusion-matrices.ipynb``` breaks down what is done in the original article and then adds some figures to confirm if the conclusion of the article is true.

## About the borrowed datasets and the codes

The entire dataset and codes listed below (some with slight modifications) are borrowed from the original work.

* ```allen-data-preprocess-mod.ipynb``` (this one is slightly modified from original)
* ```rnaseqTools.py```
* ```patch-seq-data-load.ipynb```
* ```ttype-assignment.ipynb```
* ```ttype-coverage-mod.ipynb``` (modified from original)
* ```preprocess-ephys-files-mod.ipynb``` (modified from original)
