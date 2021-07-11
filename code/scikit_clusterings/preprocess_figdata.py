import pickle
import numpy as np
from scipy import sparse

def common_gene_idx(gene_list1, gene_list2):
    """
    A function to get indices of common genes for gene_list1and gene_list2
    """
    
    # get a list of genes which can be found in both gene lists and sort them in alphabetical order
    gg = sorted(list(set(gene_list1) & set(gene_list2)))
    print('Using a common set of ' + str(len(gg)) + ' genes.')
    
    # the indices in gene_list1 that correspond to the common genes
    common_genes1 = [np.where(gene_list1==g)[0][0] for g in gg] 
    # the indices in gene_list2 that correspond to the common genes
    common_genes2 = [np.where(gene_list2==g)[0][0] for g in gg] 
    
    return common_genes1, common_genes2

def get_dense(*sparse_arrays):
    """
    Get numpy matrices for the arguments.
    The arguments must be either sparse matrices or numpy matrices.
    
    Returns tuple containing numpy versions of passed arguments.
    Don't forget to unpack if you're only passing one argument!
    """
    dense_arrays = []
    for sparse_array in sparse_arrays:
        if sparse.issparse(sparse_array):
            dense_arrays.append(sparse_array.todense())
        elif type(sparse_array) == np.matrix:
            dense_arrays.append(sparse_array)
        else:
            print("Values passed to get_dense must be a numpy or sparse matrix")
            return None
        
    return tuple(dense_arrays)

def normalize_counts(counts, length, isIntron=False):
    """
    Normalize exon/intron/UMI counts by corresponding exon/intron lengths (normalization by 
    kilobase).
    UMI counts are normalized by exon length, and a very small number is added to intron 
    lengths to avoid division by zero.
    """
    if isIntron == True:
        return counts / ((length+.001)/1000)
    else:
        return counts / (length/1000)
        

# this function was created from the first half of the map_to_tsne function in rnaseqTools.py
def preprocess_figure_data(reference_umicnt, reference_genes, new_exoncnt, new_introncnt,
                   new_genes, exonlen, intronlen, UMI_fname, exint_fname):
    """
    This function is for preprocessing the reference data and new data in the same way as the article 
    by Scala st al. It assumes that the reference data is UMI counts and the new data is separated 
    into exon counts and intron counts.
    It first normalizes the counts by exon lengths and intron lengths. The UMI counts are normalized
    by exon lengths only. Then the normalized UMI counts and exonic + intronic expression levels for 
    common genes and all genes are written as pickle files in the given path.
    
    Arguments:
    - reference_umicnt: the reference UMI counts
    - reference_genes: the list of gene names corresponding to the UMI counts
    - new_exoncnt: the exon counts for the new data
    - new_introncnt: the intron counts for the new data
    - new_genes: the list of gene names corresponding to the exon and intron counts
    - exonlen: exon lengths
    - intronlen: intron lengths
    - UMI_fname: file path + name for the output UMI counts
    - exint_fname: file path + name for the output exonic + intronic expression levels
    
    Writes out:
    1. <UMI_fname>: a pickle file of normalized UMI counts for common genes
    2. <exint_fname>: a pickle file of normalized exonic + intronic expression levels for common genes
    """
    # get indices in reference gene list and new gene list that correspond to the common genes
    ref_common_gidx, new_common_gidx = common_gene_idx(reference_genes, new_genes)
    
    # if counts are sparse matrices, convert to numpy matrices
    reference_umicnt, new_exoncnt, new_introncnt = get_dense(reference_umicnt, new_exoncnt, new_introncnt)
    
    # get the umi/exon/intron counts for the common genes
    com_ref_umi = reference_umicnt[:, ref_common_gidx]
    com_new_exon = new_exoncnt[:,new_common_gidx]
    com_new_intron = new_introncnt[:,new_common_gidx]
    
    # get the exon/intron lengths for the common genes
    com_exonlen = exonlen[new_common_gidx]
    com_intronlen = intronlen[new_common_gidx]
    
    # get normalized UMI counts
    ncom_ref_umi = normalize_counts(com_ref_umi, com_exonlen)
    ncom_ref_umi = np.log2(ncom_ref_umi + 1)
    
    # get normalized exon+intron expression levels
    ncom_exint = normalize_counts(com_new_exon, com_exonlen) + \
    normalize_counts(com_new_intron, com_intronlen, isIntron=True)
    ncom_exint = np.log2(ncom_exint + 1)
    
    # write the normalized counts and expression levels
    pickle.dump(ncom_ref_umi, open(UMI_fname, 'wb'))
    pickle.dump(ncom_exint, open(exint_fname, 'wb'))
