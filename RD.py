#checking for validity
import pandas as pd
import numpy as np

def check(ar):
    if isinstance(ar,pd.DataFrame) == True:
        ar=ar.to_numpy().T

    C=ar[0]
    Double=ar[1]
    Triple=ar[2]
    Bracket=ar[3]
    Cyclic=ar[4]
    if C < 1:
        return False
    elif C == 1 and (Double > 0 or Triple > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 2 and (Double > 1 or Triple > 1 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 2 and Double == 1 and (Triple > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 3 and (Double > 2 or Triple > 1 or Bracket > 0 or Cyclic > 1):
        return False
    elif C == 3 and Cyclic == 1 and (Double > 2 or Triple > 1 or Bracket > 0):
        return False
    elif C == 3 and Double <= 2 and (Cyclic > 1 or Triple > 1 or Bracket > 0):
        return False
    elif C == 4 and (Double > 2 or Triple > 1 or Bracket > 0 or  Cyclic > 1):
        return False
    elif C == 4 and Cyclic == 1 and (Double > 2 or Triple > 1 or Bracket > 0):
        return False
    elif C == 4 and Bracket == 1 and (Double > 2 or Triple > 1 or Cyclic > 0):
        return False
    elif C == 4 and Double == 1 and (Triple > 1 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 4 and Double == 2 and (Triple > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 4 and Triple == 1 and (Double > 1 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 4 and Triple == 2 and (Double > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 5 and (Double > 2 or Triple > 2 or Bracket > 1 or Cyclic > 1):
        return False
    elif C == 5 and Cyclic == 2 and (Double > 0 or Triple > 0 or Bracket > 0):
        return False
    elif C == 5 and Bracket == 2 and (Double > 0 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 5 and Double == 2 and Bracket == 1 and (Triple > 0 or Cyclic > 0):
        return False
    elif C == 5 and Double == 2 and Cyclic == 1 and (Triple > 0 or Bracket > 0):
        return False
    elif C == 5 and Double == 2 and (Triple > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 5 and Double == 2 and (Triple > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 5 and Cyclic == 1 and (Double > 1 or Triple > 0 or Bracket > 0):
        return False
    elif C == 5 and Bracket == 1 and (Double > 1 or Triple > 1 or Cyclic > 0):
        return False
    elif C == 5 and Bracket == 1 and (Double > 0 or Triple > 0 or Cyclic > 1):
        return False
    elif C == 5 and Triple == 1 and (Double > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 6 and (Double > 3 or Triple > 2 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 6 and Cyclic == 2 and (Double > 0 or Triple > 0 or Bracket > 0):
        return False
    elif C == 6 and Bracket == 2 and (Double > 2 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 6 and Double == 2 and Bracket <= 2 and (Triple > 0 or Cyclic > 0):
        return False
    elif C == 6 and Double == 2 and Cyclic == 1 and (Triple > 0 or Bracket > 0):
        return False
    elif C == 6 and Double == 3 and (Triple > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 6 and Double == 2 and (Triple > 1 or Bracket > 0 or Cyclic > 1):
        return False
    elif C == 6 and Cyclic == 1 and (Double > 2 or Triple > 0 or Bracket > 1):
        return False
    elif C == 6 and Bracket == 1 and (Double > 2 or Triple > 1 or Cyclic > 0):
        return False
    elif C == 6 and Bracket == 1 and (Double > 1 or Triple > 0 or Cyclic > 1):
        return False
    elif C == 6 and Triple == 1 and (Double > 2 or Bracket > 2 or Cyclic > 0):
        return False
    elif C == 7 and (Double > 3 or Triple > 2 or Bracket > 3 or Cyclic > 2):
        return False
    elif C == 7 and Bracket == 3 and (Double > 1 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 7 and Bracket == 3 and (Double > 1 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 7 and Double == 3 and Bracket <= 2 and (Triple > 0 or Cyclic > 0):
        return False
    elif C == 7 and Double == 3 and Cyclic <= 2 and (Triple > 0 or Bracket > 0):
        return False
    elif C == 7 and Double == 4 and (Triple > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 7 and Double == 2 and (Triple > 1 or Bracket > 0 or Cyclic > 2):
        return False
    elif C == 7 and Double == 2 and Bracket <= 2 and (Triple > 0 or Cyclic > 0):
        return False
    elif C == 7 and Double == 2 and Cyclic <= 2 and (Triple > 0 or Bracket > 0):
        return False
    elif C == 7 and Bracket == 2 and (Double > 2 or Triple > 1 or Cyclic > 0):
        return False
    elif C == 7 and Bracket == 1 and (Double > 3 or Triple > 1 or Cyclic > 1):
        return False
    elif C == 7 and Cyclic == 2 and (Double > 2 or Triple > 0 or Bracket > 0):
        return False
    elif C == 7 and Cyclic == 1 and (Double > 3 or Triple > 0 or Bracket > 1):
        return False
    elif C == 7 and Triple == 2 and (Double > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 7 and Triple == 1 and (Double > 2 or Bracket > 1 or Cyclic > 1):
        return False
    elif C == 8 and (Double > 4 or Triple > 2 or Bracket > 4 or Cyclic > 2):
        return False
    elif C == 8 and Bracket == 4 and (Double > 0 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 8 and Bracket == 3 and (Double > 1 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 8 and Bracket == 2 and (Double > 2 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 8 and Bracket == 1 and (Double > 2 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 8 and Cyclic == 2 and (Double > 2 or Triple > 0 or Bracket > 0):
        return False
    elif C == 8 and Cyclic == 1 and Double == 0 and (Triple > 0 or Bracket > 2):
        return False
    elif C == 8 and Cyclic == 1 and Bracket == 0 and (Triple > 0 or Double > 4):
        return False
    elif C == 8 and Double == 2 and (Triple > 1 or Bracket > 2 or Cyclic > 1):
        return False
    elif C == 8 and Double == 2 and Triple == 1 and (Bracket > 0 or Cyclic > 0):
        return False
    elif C == 8 and Double == 2 and Bracket <= 2 and (Triple > 1 or Cyclic > 1):
        return False
    elif C == 8 and Double == 1 and (Triple > 1 or Bracket > 3 or Cyclic > 1):
        return False
    elif C == 8 and Double == 1 and Triple == 1 and (Bracket > 0 or Cyclic > 0):
        return False
    elif C == 8 and Double == 1 and Bracket <= 3 and (Triple > 1 or Cyclic > 1):
        return False
    elif C == 8 and Triple == 2 and (Double > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 8 and Triple == 1 and (Double > 2 or Bracket > 1 or Cyclic > 1):
        return False
    elif C == 9 and (Double > 4 or Triple > 2 or Bracket > 4 or Cyclic > 3):
        return False
    elif C == 9 and Bracket == 4 and (Double > 0 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 9 and Bracket == 3 and (Double > 1 or Triple > 0 or Cyclic > 1):
        return False
    elif C == 9 and Bracket == 2 and (Double > 2 or Triple > 0 or Cyclic > 1):
        return False
    elif C == 9 and Bracket == 1 and (Double > 2 or Triple > 0 or Cyclic > 2):
        return False
    elif C == 9 and Cyclic == 2 and (Double > 2 or Triple > 0 or Bracket > 0):
        return False
    elif C == 9 and Cyclic == 1 and Double == 0 and (Triple > 0 or Bracket > 2):
        return False
    elif C == 9 and Cyclic == 1 and Bracket == 0 and (Triple > 0 or Double > 4):
        return False
    elif C == 9 and Double == 4 and (Triple > 0 or Bracket > 0 or Cyclic > 2):
        return False
    elif C == 9 and Double == 3 and (Triple > 0 or Bracket > 0 or Cyclic > 2):
        return False
    elif C == 9 and Double == 2 and (Triple > 0 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 9 and Double == 2 and Triple == 1 and (Bracket > 0 or Cyclic > 0):
        return False
    elif C == 9 and Double == 2 and Bracket <= 2 and (Triple > 1 or Cyclic > 1):
        return False
    elif C == 9 and Double == 2 and Cyclic <= 2 and (Triple > 1 or Bracket > 0):
        return False
    elif C == 9 and Double == 1 and (Triple > 0 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 9 and Triple == 2 and (Double > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 10 and (Double > 5 or Triple > 2 or Bracket > 5 or Cyclic > 2):
        return False
    elif C == 10 and Bracket == 5 and (Double > 0 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 10 and Bracket == 4 and (Double > 0 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 10 and Bracket == 3 and (Double > 0 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 10 and Bracket == 2 and (Double > 3 or Triple > 0 or Cyclic > 2):
        return False
    elif C == 10 and Bracket == 1 and (Double > 4 or Triple > 0 or Cyclic > 2):
        return False
    elif C == 10 and Cyclic == 2 and (Double > 5 or Triple > 0 or Bracket > 2):
        return False
    elif C == 10 and Cyclic == 1 and (Double > 5 or Triple > 0 or Bracket > 2):
        return False
    elif C == 10 and Double == 5 and (Triple > 0 or Bracket > 0 or Cyclic > 2):
        return False
    elif C == 10 and Double == 4 and (Triple > 0 or Bracket > 1 or Cyclic > 2):
        return False
    elif C == 10 and Double == 3 and (Triple > 0 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 10 and Double == 2 and (Triple > 0 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 10 and Double == 1 and (Triple > 0 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 10 and Triple == 2 and (Double > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 10 and Triple == 1 and (Double > 0 or Bracket > 0 or Cyclic > 1):
        return False
    elif C == 11 and (Double > 5 or Triple > 2 or Bracket > 5 or Cyclic > 3): 
        return False
    elif C == 11 and Bracket == 5 and (Double > 0 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 11 and Bracket == 4 and (Double > 0 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 11 and Bracket == 3 and (Double > 0 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 11 and Bracket == 2 and (Double > 3 or Triple > 0 or Cyclic > 1):
        return False
    elif C == 11 and Bracket == 1 and (Double > 3 or Triple > 0 or Cyclic > 2):
        return False
    elif C == 11 and Cyclic == 2 and (Double > 5 or Triple > 0 or Bracket > 2):
        return False
    elif C == 11 and Cyclic == 1 and (Double > 5 or Triple > 0 or Bracket > 2):
        return False
    elif C == 11 and Double == 5 and (Triple > 0 or Bracket > 0 or Cyclic > 2):
        return False
    elif C == 11 and Double == 4 and (Triple > 0 or Bracket > 1 or Cyclic > 2):
        return False
    elif C == 11 and Double == 3 and (Triple > 0 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 11 and Double == 2 and (Triple > 0 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 11 and Double == 1 and (Triple > 0 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 11 and Triple == 2 and (Double > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 11 and Triple == 1 and (Double > 0 or Bracket > 0 or Cyclic > 1):
        return False
    elif C == 12 and (Double > 6 or Triple > 2 or Bracket > 5 or Cyclic > 3): 
        return False
    elif C == 12 and Bracket == 5 and (Double > 1 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 12 and Bracket == 4 and (Double > 1 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 12 and Bracket == 3 and (Double > 2 or Triple > 0 or Cyclic > 0):
        return False
    elif C == 12 and Bracket == 2 and (Double > 5 or Triple > 0 or Cyclic > 2):
        return False
    elif C == 12 and Bracket == 1 and (Double > 6 or Triple > 0 or Cyclic > 3):
        return False
    elif C == 12 and Cyclic == 3 and (Double > 6 or Triple > 0 or Bracket > 1):
        return False
    elif C == 12 and Cyclic == 2 and (Double > 6 or Triple > 0 or Bracket > 1):
        return False
    elif C == 12 and Cyclic == 1 and (Double > 6 or Triple > 0 or Bracket > 1):
        return False
    elif C == 12 and Double == 6 and (Triple > 0 or Bracket > 0 or Cyclic > 3):
        return False
    elif C == 12 and Double == 5 and (Triple > 0 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 12 and Double == 4 and (Triple > 0 or Bracket > 2 or Cyclic > 2):
        return False
    elif C == 12 and Double == 3 and (Triple > 0 or Bracket > 0 or Cyclic > 2):
        return False
    elif C == 12 and Double == 2 and (Triple > 0 or Bracket > 0 or Cyclic > 2):
        return False
    elif C == 12 and Double == 1 and (Triple > 0 or Bracket > 5 or Cyclic > 2):
        return False
    elif C == 12 and Triple == 2 and (Double > 0 or Bracket > 0 or Cyclic > 0):
        return False
    elif C == 12 and Triple == 1 and (Double > 0 or Bracket > 0 or Cyclic > 0):
        return False
    else:
        return True