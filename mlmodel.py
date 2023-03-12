import pandas as pd
db = pd.read_excel("DataTb.xlsx")
db

import numpy as np
x = np.array(db[["C","Double","Triple","Bracket","Cyclic"]])
y = np.array(db.Tb)

## Building ML

from sklearn.model_selection import train_test_split, cross_validate
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state =0)

from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor()
DT_cv =cross_validate(DT, x_train, y_train, cv=5, return_train_score=True)
DT.fit(x_train,y_train)

y_train_pred_DT = DT.predict(x_train)
y_test_pred_DT = DT.predict(x_test)
y_pred_DT = DT.predict(x)


DT.score(x_train, y_train)
DT.score(x_test, y_test)

#MODULE

def predict_DT(data):
    print(data)
    DT.predict(data)
    return DT.predict(data)





