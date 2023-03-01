#checking for validity
import pandas as pd
import numpy as np

def check(ar):
    if isinstance(ar,pd.DataFrame):
        ar=ar.to_numpy().T
        
    CRe=ar[0]
    DoubleCCRe=ar[1]
    TripleCC=ar[2]
    Bracket=ar[3]
    Benzene=ar[4]
    CycleRe=ar[5]
    SingleCO=ar[6]
    DoubleCO=ar[7]

    if CRe < 1:
        return False
    elif 