import numpy as np

def get_transcriptomic_features(m1, ttypes):
    """
    Gets the transcriptomic features just before they were put through t-SNE,
    and a Boolean matrix that gets cells that are valid for analysis.
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