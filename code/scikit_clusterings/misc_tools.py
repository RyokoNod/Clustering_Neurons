import numpy as np
from sklearn import preprocessing
from sklearn.metrics import adjusted_mutual_info_score,fowlkes_mallows_score

def le_family_names(m1data):
    """
    This function label encodes the broad cell families of the given subgroup of Yao et al.'s data.
    The dataset handed over MUST be in subgroups already: viplamp, pvsst, exc, or neurons.
    
    Returns:
    - family_names: numpy array with family assignment for cells in m1data
    - family_codes: numpy array of numeric labels corresponding to family_names
    """
    le = preprocessing.LabelEncoder()
    family_names = [cname.split()[0] for cname in m1data["clusterNames"][m1data["clusters"]]]
    family_codes = le.fit_transform(family_names)

    print("Family_labels:")
    for label in le.classes_:
        print(f"{label}: {le.transform(label.reshape(-1))}")
        
    return family_names, family_codes
    

def ami_and_fmscore(true_labels, predicted_labels, silent=False):
    """
    This function will print the adjusted mutual information and Fowlkes-Mallows score
    while returning the two values.
    Setting silent=True will prevent the messages from printing.
    """
    # unadjusted mutual information score measures if two cluster assignements agree with each other
    # the unadjusted score gives us false results when cluster sizes are small, so using adjusted version
    # value=1: perfect, value around 0: same as random assignment (can be negative)
    AMI = adjusted_mutual_info_score(true_labels, predicted_labels)
    if silent==False:
        print("Adjusted Mutual Info: {}".format(AMI))
        print("(0: bad, 1: perfect)\n")

    # the Fowlkes-Mallows score geometric mean between of the precision and recall
    # value=1: perfect, value=0: terrible
    FMS = fowlkes_mallows_score(true_labels, predicted_labels)
    if silent==False:
        print("Fowlkes-Mallows Score:  {}".format(FMS))
        print("(0: bad, 1: perfect)\n")
    
    return AMI, FMS