import numpy as np
import pylab as plt
import seaborn as sns; sns.set()
import matplotlib
from sklearn.metrics import adjusted_mutual_info_score,fowlkes_mallows_score

def sns_styleset():
    sns.set(context='paper', style='ticks', font='Arial')
    matplotlib.rcParams['axes.linewidth']    = .5
    matplotlib.rcParams['xtick.major.width'] = .5
    matplotlib.rcParams['ytick.major.width'] = .5
    matplotlib.rcParams['xtick.major.size'] = 2
    matplotlib.rcParams['ytick.major.size'] = 2
    matplotlib.rcParams['xtick.minor.size'] = 1
    matplotlib.rcParams['ytick.minor.size'] = 1
    matplotlib.rcParams['font.size']       = 6
    matplotlib.rcParams['axes.titlesize']  = 6
    matplotlib.rcParams['axes.labelsize']  = 6
    matplotlib.rcParams['legend.fontsize'] = 6
    matplotlib.rcParams['xtick.labelsize'] = 6
    matplotlib.rcParams['ytick.labelsize'] = 6
    matplotlib.rcParams['figure.dpi'] = 120

sns_styleset()

def kNN_confusion_matrix_ff(pred, labels, classes):
    """
    A function to get the family-family confusion matrix for kNN.
    
    Attributes:
    - pred: the predictions given by the nearest neighbors
    - labels: the ground truth labels for the cells
    - classes: the list of cell families
    
    Output:
    The confusion matrix for family assignment with size (number of families, number of families)
    """
    C = np.zeros((classes.size, classes.size))
    for i, cl in enumerate(classes): #for every class
        num = 0 # counts how many cells there are within one family
        for ind in np.where(labels==cl)[0]: # for every cell that has that class as ground truth
            # pred[ind,:] gets the indices of the k nearest neighbors for that cell
            # labels[pred[ind,:]] gets the family assignments of the k nearest neighbors for that cell
            # u: the unique labels in the k nearest neighbor family list
            # count: the counts for the unique labels
            u, count = np.unique(labels[pred[ind,:]], return_counts=True)
            if u[np.argmax(count)] in classes: # if the family most often assigned to the k nearest neighbors is a valid family
                num += 1# add cell count

                # add count to the corresponding cell in confusion matrix
                # rows: ground truth, cols: assignment by majority vote of k nearest neighbors
                C[i, classes==u[np.argmax(count)]] += 1

        C[i,:] /= num #divide by cell count within family so that the raw counts become proprotions
        
    return C
    
def kNN_confusion_matrix_tf(pred, labels, classes, cell_selector, labelset, layerset, cutoff=10, restrictLayers=False, clusterN=88):
    """
    A function to get the transcriptomic type-family confusion matrix for kNN.
    
    Attributes:
    - pred: the predictions given by the nearest neighbors
    - labels: the ground truth labels for the cells
    - classes: the list of cell families
    - cell_selector: a Boolean numpy array that indicates which cells are used in analysis
    - labelset: the master data for transcriptomic type assignment
    - layerset: the numpy array that has the layer assignment of each cell
    - cutoff: how many cells there should be in a transcriptomic type to calculate the confusion matrix
    - restrictLayers: True - uses the cells from most common layer per ttype, False - uses every layer
    - clusterN: how many transcriptomic types there are in total
    
    Output:
    The confusion matrix for family assignment, aggregated by transcriptomic types.
    The size will be (number of total transcriptomic types, number of families).
    Rows where the cutoff is not satisfied will have value np.nan.
    """
    C = np.zeros((clusterN, classes.size)) * np.nan
    for t in range(clusterN):
        # this is a filter that gets the indices of the cell type t
        ind = (labelset['m1consensus_ass'][cell_selector].astype(int) == t) 

        # if restrictLayers is True, only keep the cells that came from the most common layer for cell type t
        if np.sum(ind) >= cutoff and restrictLayers:
            l, lc = np.unique(layerset[cell_selector][ind], return_counts=True)
            mostCommonLayer = l[np.argmax(lc)]
            ind &= (layerset[cell_selector] == mostCommonLayer)

        # only calculate the confusion matrix for cells that have at least 10 cells classified to that label
        if np.sum(ind) >= cutoff:
            C[t,:] = 0
            num = 0
            for i in np.where(ind)[0]:
                u, count = np.unique(labels[pred[i,:]], return_counts=True)
                if u[np.argmax(count)] in classes:
                    num += 1
                    C[t, classes==u[np.argmax(count)]] += 1
            C[t,:] /= num

    return C
    
def kNN_confusion_matrix_tt(pred, labels, cell_selector, layerset, cutoff=10, restrictLayers=False, clusterN=88):
    """
    A function to get the transcriptomic type-transcriptomic type confusion matrix for kNN.
    
    Attributes:
    - pred: the predictions given by the nearest neighbors
    - labels: the ground truth labels for the cells
    - cell_selector: a Boolean numpy array that indicates which cells are used in analysis
    - layerset: the numpy array that has the layer assignment of each cell
    - cutoff: how many cells there should be in a transcriptomic type to calculate the confusion matrix
    - restrictLayers: True - uses the cells from most common layer per ttype, False - uses every layer
    - clusterN: how many transcriptomic types there are in total
    
    Output:
    The confusion matrix for transcriptomic type assignment.
    The size will be (number of total transcriptomic types, number of total transcriptomic types).
    Rows where the cutoff is not satisfied will have value np.nan.
    """
    C = np.zeros((clusterN, clusterN)) * np.nan
    for t in range(clusterN):
        # this is a filter that gets the indices of the cell type t
        ind = (labels == t) 

        # if restrictLayers is True, only keep the cells that came from the most common layer for cell type t
        if np.sum(ind) >= cutoff and restrictLayers:
            l, lc = np.unique(layerset[cell_selector][ind], return_counts=True)
            mostCommonLayer = l[np.argmax(lc)]
            ind &= (layerset[cell_selector] == mostCommonLayer)

        # only calculate the confusion matrix for cells that have at least 10 cells classified to that label
        if np.sum(ind) >= cutoff:
            C[t,:] = 0
            num = 0
            for i in np.where(ind)[0]:
                neighbor_labels = [label for label in labels[pred[i,:]] if label in np.arange(clusterN)]
                #u, count = np.unique(labels[pred[i,:]], return_counts=True)
                u, count = np.unique(neighbor_labels, return_counts=True)
                #if u[np.argmax(count)] in np.arange(clusterN):
                num += 1
                C[t, u[np.argmax(count)]] += 1
            C[t,:] /= num

    return C

def kNN_plot_cm_ff(cm_dict, classes, titles, figsize):
    """
    Plots the family-family confusion matrices for each feature set.
    
    Arguments:
    - cm_dict: the confusion matrix dictionary
    - classes: the list of transcriptomic family names
    - titles: the title dictionary to be used for each subplot. Make sure you use the same keys as cm_dict
    - figsize: the figure size to be passed on to pyplot
    """
    plt.figure(figsize=figsize)
    cnt = 1
    for mode, C in cm_dict.items():
        ax = plt.subplot(2,3,cnt )
        
        plt.sca(ax)
        plt.imshow(C, vmin=0, vmax=1, cmap=plt.get_cmap('Greys'))
        plt.xticks([])
        plt.yticks(np.arange(classes.size), classes)
        plt.gca().tick_params(axis='y', length=0)
        plt.ylim([-.5, classes.size-.5])
        plt.gca().invert_yaxis()

        for i in range(classes.size):
            for j in range(classes.size):
                if C[i,j] >= .05:
                    if C[i,j] > .6:
                        col = 'w'
                    else:
                        col = 'k'
                    plt.text(j,i, '{:2.0f}'.format(100*C[i,j]), fontsize=5, ha='center', va='center', color=col)
                    
        plt.title(titles[mode], y=1.07)
        cnt +=1
    plt.tight_layout()
    
def kNN_plot_cm_tf(cm_dict, classes, titles, clusterNames, figsize):
    """
    Plots the transcriptomic type-family confusion matrices for each feature set.
    
    Arguments:
    - cm_dict: the confusion matrix dictionary
    - classes: the list of transcriptomic family names
    - titles: the title dictionary to be used for each subplot. Make sure you use the same keys as cm_dict
    - clusterNames: the master list of transcriptomic family names
    - figsize: the figure size to be passed on to pyplot
    """
    plt.figure(figsize=figsize)
    cnt = 1
    for mode, C in cm_dict.items():
        ax = plt.subplot(6,1,cnt)
        aboveCutoff = ~np.isnan(C[:,0]) # plot only where there are values in the confusion matrix

        # settings for the axes
        plt.sca(ax)
        plt.imshow(C[aboveCutoff,:].T, vmin=0, vmax=1, cmap=plt.get_cmap('Greys'), aspect='auto')
        plt.xticks(np.arange(np.sum(aboveCutoff))+.15, clusterNames[aboveCutoff], fontsize=5, rotation=90) # +.15 to adjust label positions
        plt.gca().tick_params(axis='both', length=0)
        plt.yticks(np.arange(classes.size), classes)
        plt.ylim([-.5, classes.size-.5])
        plt.xlim([-.5,np.sum(aboveCutoff)-.5])
        plt.gca().invert_yaxis()
        
        plt.title(titles[mode], y=1.07)
        cnt+=1

    plt.tight_layout()

def kNN_plot_cm_tt(cm_dict, titles, clusterNames, figsize):
    """
    Plots the transcriptomic type-transcriptomic type confusion matrices for each feature set.
    
    Arguments:
    - cm_dict: the confusion matrix dictionary
    - titles: the title dictionary to be used for each subplot. Make sure you use the same keys as cm_dict
    - clusterNames: the master list of transcriptomic family names
    - figsize: the figure size to be passed on to pyplot
    """
    plt.figure(figsize=figsize)
    cnt = 1
    for mode, C in cm_dict.items():
        ax = plt.subplot(3,2,cnt)
        aboveCutoff = ~np.isnan(C[:,0]) # plot only where there are values in the confusion matrix

        plt.sca(ax)
        plt.imshow(C[aboveCutoff][:,aboveCutoff], vmin=0, vmax=1, cmap=plt.get_cmap('Greys'), aspect='auto')
        plt.xticks(np.arange(np.sum(aboveCutoff))+.15, clusterNames[aboveCutoff], fontsize=5, rotation=90) # +.15 to adjust label positions
        plt.gca().tick_params(axis='both', length=0)
        plt.yticks(np.arange(np.sum(aboveCutoff))+.15, clusterNames[aboveCutoff], fontsize=5)
        plt.ylim([-.5, clusterNames[aboveCutoff].size-.5])
        plt.xlim([-.5,np.sum(aboveCutoff)-.5])
        plt.gca().invert_yaxis()
                    
        plt.title(titles[mode], y=1.07)
        cnt +=1
    plt.tight_layout()
    
def evaluate_ami_fms(kNN_dict, label_dict, titles, class_list):
    """
    A function to calculate the adjusted mutual information and Fowlkes-Mallows score of
    the given kNN assignments. Can be used for both family assignments and ttype assignments.
    
    Arguments:
    - kNN_dict: a dictionary that contains the k nearest neighbors of each cell
    - label_dict: a dictionary that has the ground truth labels
    - titles: the dictionary used to give a title to the printed text. Make sure it has the same keys as label_dict
    - class_list: for family assignments, the list of family names. Number of ttypes for ttype assignment
    
    Returns:
    - pred: a dictionary of the predictions based on the k nearest neighbors
    - AMI: the adjusted mutual information score
    - FMS: the Fowlkes-Mallows score
    """
    pred_dict = {}
    AMI_dict = {}
    FMS_dict = {}
    
    if type(class_list)== int:
        class_list = np.arange(class_list)

    for mode in label_dict.keys():
        print(f"--------------------------{titles[mode]}--------------------------")
        pred = []
        labels = label_dict[mode]
        neighbors=kNN_dict[mode]

        for cell in neighbors:
            # in case the nearest neighbors contain nan or a family other than the ones in the list, remove them
            neighbor_labels = [label for label in labels[cell] if label in class_list]
            t, vote = np.unique(neighbor_labels, return_counts=True) # get the "votes" for the nearest neighbor assignment
            pred.append(t[np.argmax(vote)]) # assign the result of the majority vote
        pred = np.array(pred)

        AMI = adjusted_mutual_info_score(pred, labels)
        FMS = fowlkes_mallows_score(pred, labels)
        print("Adjusted Mutual Info:", AMI)
        print("Fowlkes-Mallows Score:", FMS,"\n")

        pred_dict[mode] = pred
        AMI_dict[mode] = AMI
        FMS_dict[mode] = FMS
    
    return pred, AMI, FMS

    