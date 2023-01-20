from random import randint as rnd
import random
import pandas as pd
from mlmodel import DT, predict_DT
from RD import check
from error import errorcheck, errorcheck2
import numpy as np

# Define input data
LowerBound = float(input("Lower TEMPARETURE range IS: "))
UpperBound = float(input("Upper TEMPARETURE range IS: "))

# Define maximum iteration limit
max_iterations = 100

# Initialize iteration counter
iteration = 0

# Define a function to check the error of the selected parameters
def errorcheck(data):
    showerr =[]
    for i in data["Predict"]:
        if i < LowerBound:
            pcerror = float((abs(i-LowerBound)/LowerBound)*100)
        elif i > UpperBound:
            pcerror = float((abs(i-UpperBound)/UpperBound)*100)
        else:
            pcerror = 0
        showerr.append(pcerror)
    return showerr

def errorcheck2(data):
    data["Error"] = errorcheck(data)
    data1= data.sort_values("Error")
    return data1
  
# Defint the random
def pop():
    count = 0
    cc =[]
    while count <  10 :
        carbon = [rnd(1, 12), rnd(0, 6), rnd(0, 4), rnd(0, 10), rnd(0, 4)]  #มาแก้ตรงนี้ค่าที่สุ่ม
        checkk = check(carbon)
        if checkk == True:
            cc.append(carbon)
            count = count+1
        else:
            count = count

    return pd.DataFrame(cc)


# Defint the selection
def rank_selection(a):
    rankk =[]
    for i in random_parameters["Predict"]:
        if i < LowerBound:
            rank = float(abs(i - LowerBound))
        elif i > UpperBound:
            rank = float(abs(i - UpperBound))
        else:
            rank = 0
        rankk.append(rank)
    return rankk

def rank_selection2(b):
    b["rank"] = rank_selection(b)
    b1= b.sort_values("rank")
    b1_selected = b1.iloc[0:4]
    return b1_selected

selected = rank_selection2(random)
selected
errorcheck(selected)

# Define the crossover function
def crossover(parent):
    parent=parent.drop(columns=["Predict","rank"])
    p1=parent.iloc[0]
    p2=parent.iloc[1]
    p3=parent.iloc[2]
    p4=parent.iloc[3]
    p1=p1.to_numpy().T
    p2=p2.to_numpy().T
    p3=p3.to_numpy().T
    p4=p4.to_numpy().T
    pt=rnd(1,4)
    print("the cross over point is",pt)
    off1=p1[:pt]
    off2=p2[pt:]
    c1=np.concatenate((off1,off2))
    off1=p2[:pt]
    off2=p1[pt:]
    c2=np.concatenate((off1,off2))
    off1=p3[:pt]
    off2=p4[pt:]
    c3=np.concatenate((off1,off2))
    off1=p4[:pt]
    off2=p3[pt:]
    c4=np.concatenate((off1,off2))
    nxt_gen=np.vstack((c1,c2,c3,c4))
    nxt_gen_pd=pd.DataFrame(nxt_gen, columns=["C","Double", "Triple", "Bracket", "Cyclic"])
    return nxt_gen_pd


# Define the mutation function
def mutate(check2):
    check2=check2.drop(columns=["Predict"])
    rng=rnd(0,10)
    if rng < 3:
        print("No Mutation")
        return
    else:
        rng=rnd(1,10)
        col=rnd(0,3)
        print("the mutate row is",col)
        mut=check2.iloc[col]
        mut=mut.to_numpy().T
        if rng <= 6:
            c=rnd(1,12)
            mut[0]=c
        elif rng == 7:
            mut[1]=rnd(0,mut[0]-1)
        elif rng == 8:
            mut[2]=rnd(0,2)
        elif rng == 9:
            mut[3]=rnd(0,5)
        elif rng == 10:
            mut[4]=rnd(0,3)
    temp=np.arange(5)
    new=np.vstack((mut,temp))
    mut_pd=pd.DataFrame(new, columns=["C","Double", "Triple", "Bracket", "Cyclic"])
    mut_pd=mut_pd.drop([1])
    return mut_pd


# Create an empty dataframe to store the results
results_df = pd.DataFrame()

# Implement the genetic algorithm
while iteration < max_iterations:
    # Get a random set of parameters
    random_parameters = pop()
    random_parameters.columns = ["C","Double", "Triple", "Bracket", "Cyclic"]

    # Predict the output using the random parameters
    random_parameters["Predict"] = predict_DT(random_parameters)

    # Select the best parameters using rank selection
    selected_parameters = rank_selection2(random_parameters)

    # Check the error of the selected parameters
    error = errorcheck(selected_parameters)

    # If the error is zero, append the selected_parameters dataframe to the results_df
    if error == 0:
        results_df = results_df.append(selected_parameters)
    # If the error is non-zero, try crossover and mutation
    else:
        check2 = crossover(selected_parameters)
        check2["Predict"] = predict_DT(check2)
        error = errorcheck(check2)

        if error != 0:
            check3 = mutate(check2)
            check3["Predict"] = predict_DT(check3)
            error = errorcheck(check3)

            if error != 0:
                continue
            else:
                if check(check3) == True:
                    results_df = results_df.append(check3)
                    break
        else:
            if check(check2) == True:
                results_df = results_df.append(check2)
                break
    iteration += 1

