# this file stores the selection_cuts for the CYGNO data analysis, in a dictionary
import numpy as np
import pandas as pd

positive_infinity = np.inf
negative_infinity = -np.inf

sc_tgausssigma_factor = 0.152
fiducial_cuts_pedro_dict =     {"sc_xmin":[255, positive_infinity ],
                                "sc_xmax":[negative_infinity, 2000],
                                "sc_ymin":[300, positive_infinity],
                                "sc_ymax":[negative_infinity, 2000],}
fiducial_cuts_flaminia_dict =  {"sc_xmin":[400, positive_infinity ],
                                "sc_xmax":[negative_infinity, 1900],
                                "sc_ymin":[400, positive_infinity],
                                "sc_ymax":[negative_infinity, 1900],}

quality_cut_flaminia_dict = {       "sc_rms" :[6, positive_infinity],
                             "sc_tgausssigma":[0.5/sc_tgausssigma_factor, positive_infinity]}

def create_mask(df, cuts_dict):
    """
    Creates a boolean mask for a DataFrame based on dictionary cuts.
    
    Args:
        df: pandas DataFrame
        cuts_dict: Dictionary with column names as keys and [min, max] pairs as values
        
    Returns:
        Boolean mask where True indicates rows that pass all cuts
    """
    mask = pd.Series(True, index=df.index)
    
    for column, (min_val, max_val) in cuts_dict.items():
        if column in df.columns:
            col_mask = (df[column] >= min_val) & (df[column] <= max_val)
            mask = mask & col_mask
    
    return mask


print(f'loaded cuts dicts')