import numpy as np
from sklearn.decomposition import PCA

def get_transcriptomic_features(m1, ttypes):
    """
    Gets the transcriptomic features processed the same way as in the article's 
    confusion matrices, and a Boolean matrix that gets cells that are valid for analysis.
    """
    # like the other sets used in the confusion matrix visualization in Scala's article,
    # the transcriptomic features must be in the state just before it was processed by t-SNE
    # for the transcriptomic features, this means the exon and intron counts are combined,
    # in log2 scale, and reduced from 42,466 features to 50 by PCA

    exons = m1.exonCounts.copy()
    introns = m1.intronCounts.copy()
    exons = np.array(exons.todense())
    introns = np.array(introns.todense())

    # keep only cells that have transcriptomic types assigned to them
    keepcells = (ttypes['type']!='') & (m1.exclude=='')
    exons = exons[keepcells,:]
    introns = introns[keepcells,:]

    # normalize by exon/intron lengths, combine, and put into log scale
    # for this process I referenced rnaseqTools.map_to_tsne
    exons = exons / (m1.exonLengths/1000)
    introns = introns / ((m1.intronLengths+.001)/1000)
    exon_introns = np.log2(exons + introns +1)

    # do PCA. For this I referenced how Yao et al.'s UMI counts were processed in allen-data-preprocess-mod.ipynb
    exon_introns = exon_introns - exon_introns.mean(axis=0)
    U,s,V = np.linalg.svd(exon_introns, full_matrices=False)
    U[:, np.sum(V,axis=1)<0] *= -1
    exon_introns = np.dot(U, np.diag(s))
    exon_introns = exon_introns[:, np.argsort(s)[::-1]][:,:50]

    # creating the feature dictionary and filter dictionary
    tTsneFeatures = np.zeros((m1.cells.size, exon_introns.shape[1])) * np.nan
    tTsneFeatures[keepcells,:] = exon_introns
    
    return tTsneFeatures, keepcells

def get_ephys_features(m1, ttypes):
    """
    Gets the electrophysiological features processed the same way as in the article's 
    confusion matrices, and a Boolean matrix that gets cells that are valid for analysis.
    
    The difference between the original and the output of this function is the cell selection criteria.
    The article selects all cells that have all 17 ephys features, but this one has an additional condition:
    cells must have all 17 ephys features AND have ttype assigned AND not be one of the cells that are excluded from analysis
    """
    features_exclude = ['Afterdepolarization (mV)', 'AP Fano factor', 'ISI Fano factor', 
                        'Latency @ +20pA current (ms)', 'Wildness', 'Spike frequency adaptation',
                        'Sag area (mV*s)', 'Sag time (s)', 'Burstiness',
                        'AP amplitude average adaptation index', 'ISI average adaptation index',
                        'Rebound number of APs'] # Scala et al. omits features that are redundant or mostly has value 0
    features_log =     ['AP coefficient of variation', 'ISI coefficient of variation', 
                        'ISI adaptation index', 'Latency (ms)'] # features that need to be in natural log scale

    # 1. put features that need to be in log scale in log scale
    # 2. omit features that need to be omitted
    # 3. keep only the cells that have all of the remaining features
    # 4. standardize the values
    X = m1.ephys.copy()
    for e in features_log:
        X[:, m1.ephysNames==e] = np.log(X[:, m1.ephysNames==e])
    X = X[:, ~np.isin(m1.ephysNames, features_exclude)]

    # adding additional conditions to original: cell must have a ttype and must not be excluded from analysis
    keepcells = ~np.isnan(np.sum(X, axis=1)) & (ttypes['type']!='') & (m1.exclude=='') 
    X = X[keepcells, :]

    X = X - X.mean(axis=0)
    X = X / X.std(axis=0)

    ephysTsneData = np.zeros((m1.cells.size, X.shape[1])) * np.nan
    ephysTsneData[keepcells,:] = PCA().fit_transform(X) # doing PCA but keeping all dimensions and projecting into new space
    ephysTsneData[keepcells,:] /= np.std(ephysTsneData[keepcells,0]) # the article somehoe only scales with first component's std

    return ephysTsneData, keepcells

def get_morph_features(m1, ttypes):
    """
    Gets the morphometric features processed the same way as in the article's 
    confusion matrices, and a Boolean matrix that gets cells that are valid for analysis.
    
    The difference between the original and the output of this function is the cell selection criteria 
    - though it did not make a difference.
    The article does not exclude the cells that did not have transcriptomic types assigned, but this one does.
    """
    # getting Boolean array to select cells that can be used for morphometric analysis
    keepcells = (np.sum(~np.isnan(m1.morphometrics), axis=1) > 0) # must have all morphometric features
    keepcells[np.isin(m1.cells, ['20180820_sample_1', '20180921_sample_3'])] = False # must not be one of the cells unsuitable for analysis

    inhCells = np.isin(ttypes['family'], ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg'])
    excCells = np.isin(ttypes['family'], ['CT', 'IT', 'NP', 'ET'])
    keepcells &= (inhCells | excCells) # need to be either excitatory or inhibitory neurons

    keepcells &= (ttypes['type']!='') & (m1.exclude == '') # must have ttype assigned and must not be one of the cells that don't have valid features

    # Boolean array to get features for inhibitory/excitatory neurons
    inhFeatures = np.sum(~np.isnan(m1.morphometrics[inhCells & keepcells,:]),axis=0)>0
    excFeatures = np.sum(~np.isnan(m1.morphometrics[excCells & keepcells,:]),axis=0)>0

    inhChunk = m1.morphometrics[inhCells & keepcells,:][:, inhFeatures] # numpy array with all inhibitory cells and features
    excChunk = m1.morphometrics[excCells & keepcells,:][:, excFeatures] # numpy array with all excitatory cells and features

    # standardize all features
    inhChunk = inhChunk - inhChunk.mean(axis=0)
    inhChunk = inhChunk / inhChunk.std(axis=0)
    excChunk = excChunk - excChunk.mean(axis=0)
    excChunk = excChunk / excChunk.std(axis=0)

    # do PCA on the inhibitory/excitatory features, keep 20 dimensions
    # and standardize by the first principal component's standard deviation
    inhPC = PCA(n_components=20).fit_transform(inhChunk)
    inhPC /= np.std(inhPC[:,0])
    excPC = PCA(n_components=20).fit_transform(excChunk)
    excPC /= np.std(excPC[:,0])
    excPC += .25 #to prevent overlap between populations

    # do the same for the z-profiles
    inhZprof = m1.zProfiles[inhCells & keepcells,:]
    excZprof = m1.zProfiles[excCells & keepcells,:]

    inhZPC = PCA(n_components=5).fit_transform(inhZprof)[:,1:]
    inhZPC /= np.std(inhZPC[:,0])
    excZPC = PCA(n_components=5).fit_transform(excZprof)[:,1:]
    excZPC /= np.std(excZPC[:,0])
    excZPC += .25

    morphTsneData = np.zeros((m1.cells.size, inhPC.shape[1]*2 + inhZPC.shape[1]*2)) + 0
    morphTsneData[inhCells & keepcells,  0:inhPC.shape[1]] = inhPC 
    morphTsneData[excCells & keepcells, inhPC.shape[1]:inhPC.shape[1]*2] = excPC
    morphTsneData[inhCells & keepcells, inhPC.shape[1]*2:inhPC.shape[1]*2+inhZPC.shape[1]] = inhZPC
    morphTsneData[excCells & keepcells, inhPC.shape[1]*2+inhZPC.shape[1]:inhPC.shape[1]*2+inhZPC.shape[1]*2] = excZPC
    
    return morphTsneData, keepcells
 
def combine2features(featureset1, featureset2, keepcells1, keepcells2):
    """
    Returns a combination of 2 features and a Boolean matrix that gets cells that are valid for analysis.
    """
    combinedFeatures = np.concatenate((featureset1, featureset2), axis=1)
    keepcells = keepcells1 & keepcells2 & ~np.isnan(np.sum(combinedFeatures,axis=1))
    return combinedFeatures, keepcells
    
def combine3features(featureset1, featureset2, featureset3, keepcells1, keepcells2, keepcells3):
    """
    Returns a combination of all features and a Boolean matrix that gets cells that are valid for analysis.
    """
    combinedFeatures = np.concatenate((featureset1, featureset2, featureset3), axis=1)
    keepcells = keepcells1 & keepcells2 & keepcells3 & ~np.isnan(np.sum(combinedFeatures,axis=1))
    return combinedFeatures, keepcells

def get_feature_dict(m1, ttypes):
    feature_matrices = {}
    cell_filters = {}
    
    feature_matrix, cell_filter = get_transcriptomic_features(m1, ttypes)
    feature_matrices["t"] = feature_matrix
    cell_filters["t"] = cell_filter
    
    feature_matrix, cell_filter = get_ephys_features(m1, ttypes)
    feature_matrices["e"] = feature_matrix
    cell_filters["e"] = cell_filter
    
    feature_matrix, cell_filter = get_morph_features(m1, ttypes)
    feature_matrices["m"] = feature_matrix
    cell_filters["m"] = cell_filter
    
    feature_matrix, cell_filter = combine2features(feature_matrices["t"], feature_matrices["e"],cell_filters["t"], cell_filters["e"])
    feature_matrices["te"] = feature_matrix
    cell_filters["te"] = cell_filter
    
    feature_matrix, cell_filter = combine2features(feature_matrices["t"], feature_matrices["m"],cell_filters["t"], cell_filters["m"])
    feature_matrices["tm"] = feature_matrix
    cell_filters["tm"] = cell_filter
    
    feature_matrix, cell_filter = combine2features(feature_matrices["e"], feature_matrices["m"],cell_filters["e"], cell_filters["m"])
    feature_matrices["em"] = feature_matrix
    cell_filters["em"] = cell_filter
    
    feature_matrix, cell_filter = combine3features(feature_matrices["t"], feature_matrices["e"], feature_matrices["m"], cell_filters["t"], cell_filters["e"], cell_filters["m"])
    feature_matrices["tem"] = feature_matrix
    cell_filters["tem"] = cell_filter
    
    return feature_matrices, cell_filters