import numpy as np
from sklearn import preprocessing

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