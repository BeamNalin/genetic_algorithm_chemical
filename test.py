from random import randint as rnd
import random
import pandas as pd
from mlmodel import DT, predict_DT
from RD import check
from error import errorcheck, errorcheck2
import numpy as np


###### Before use test.py -> run mlmodel.py -> run RD.py ########

#input T ที่ต้องการ
LowerBound = float(input("Lower TEMPARETURE range IS: "))
UpperBound = float(input("Upper TEMPARETURE range IS: "))

#add percentage error 
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

#rank percentage error from low to high
def errorcheck2(data):
    data["Error"] = errorcheck(data)
    data1= data.sort_values("Error")
    return data1

# Random
def pop():
    count = 0
    cc =[]
    while count <  1000 :
        carbon = [rnd(1, 12), rnd(0, 6), rnd(0, 4), rnd(0, 10), rnd(0, 4)]  #มาแก้ตรงนี้ค่าที่สุ่ม
        checkk = check(carbon)
        if checkk == True:
            cc.append(carbon)
            count = count+1
        else:
            count = count

    return pd.DataFrame(cc)

# random =pop()
# random.columns = ["C","Double", "Triple", "Bracket", "Cyclic"]
# random

#Predict -> ต้องเชื่อมกับตัว ML -> เดี๋ยวต่อกันอีกที
# random["Predict"] = predict_DT(random)
# print("Random dataset")
# print(random)


#### random ตรงนี้มี predict แล้ว
# ----------------------------------------------------------------------------------------#

#Selection
def rank_selection(a):
    rankk =[]
    for i in a["Predict"]:
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
    b1_selected = b1.iloc[0:128]
    return b1_selected


# print("Selection")
# selected = rank_selection2(random)
# selected
# print(errorcheck2(selected))

## crossover

def crossover(parent):
    parent = parent.drop(columns=["Predict", "rank"])
    num_parents = parent.shape[0]
    offspring = []
    for i in range(num_parents):
        for j in range(i+1, num_parents):
            p1 = parent.iloc[i].to_numpy()
            p2 = parent.iloc[j].to_numpy()
            pt = np.random.randint(1, p1.shape[0])
            print(f"the crossover point for parents {i+1} and {j+1} is {pt}")
            off1 = np.concatenate((p1[:pt], p2[pt:]))
            off2 = np.concatenate((p2[:pt], p1[pt:]))
            offspring.append(off1)
            offspring.append(off2)
    offspring_pd = pd.DataFrame(offspring, columns=["C", "Double", "Triple", "Bracket", "Cyclic"])
    return offspring_pd

# check1=rank_selection2(random)
# print(check1)
# check2=crossover(check1)
# check2["Predict"]=predict_DT(check2)

# print(check2)


## mutate

def mutate(check2):
    check2=check2.drop(columns=["Predict","rank"])
    rng=rnd(0,10)
    if rng < 3:
        print("No Mutation")
        return check2
    else:
        rng=rnd(1,10)
        col=rnd(0,3)
        print("the mutate row is",col)
        mut=check2.iloc[col]
        mut=mut.to_numpy().T
        if rng > 0:
            c=rnd(mut[0]-3,mut[0]+3)
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
 
# print("Mutate") 
# check3=mutate(check2)
# check3["Predict"]=predict_DT(check3)
# error = errorcheck(check3)
# print(check3,error)

#wrtie csv file
# random.to_csv("random4.csv")
# selected.to_csv("selected4.csv")
# check2.to_csv("crossover4.csv")
# check3.to_csv("mutate4.csv")



# automaic code

# loop = 0
# countloop = []
# while loop < 100:
#     loop +=1
#     random =pop()
#     random.columns = ["C","Double", "Triple", "Bracket", "Cyclic"]
#     random["Predict"] = predict_DT(random)
#     selected = rank_selection2(random)
#     error = errorcheck(selected)
#     print("Selection")
#     print(selected)
#     print("and error is")
#     print(error)
#     if error[0] > 0:
#         check2=crossover(check1)
#         check2["Predict"]=predict_DT(check2)
#         error = errorcheck(check2)
#         print("Crossover")
#         print(check2)
#         print("and error is")
#         print(error)
#         if error[0] > 0:
#             check3=mutate(check2)
#             check3["Predict"]=predict_DT(check3)
#             error = errorcheck(check3)
#             print("Mutate")
#             print(check3)
#             print("and error is")
#             print(error)
#             if error[0] > 0:
#                 continue
#             else:
#                 if check(check3) == True:
#                     print("The SMILES Solution is")
#                     print(check3)
#                     print("The iteration is",loop)
#                     break
#                 else:
#                     continue
#         else:
#             if check(check2.iloc[0]) == True:
#                 print("The SMILES Solution is")
#                 print(check2.iloc[0])
#                 print("The iteration is",loop)
#                 break
#             else:
#                 continue
#     else:
#         if check(selected.iloc[0]) == True:
#             print("The SMILES Solution is")
#             print(selected.iloc[0])
#             print("The iteration is",loop)
#             break
#         else:
#             continue

#version 2
dataset = pop()
dataset.columns = ["C","Double", "Triple", "Bracket", "Cyclic"]
iteration = 100
for loop in range(iteration):
    print("loop", loop+1)

    dataset = dataset[["C", "Double", "Triple", "Bracket", "Cyclic"]]
    dataset["Predict"] = predict_DT(dataset)
    selected = rank_selection2(dataset)
    error = errorcheck(selected)
    
    print("Selection")
    print(selected)
    print("and error is")
    print(error)
    
    if error[0] > 5:
        check2 = crossover(selected)
        check2["Predict"] = predict_DT(check2)
        check2 = rank_selection2(check2)
        error = errorcheck(check2)
        
        print("Crossover")
        print(check2)
        print("and error is")
        print(error)
        
        if error[0] > 5:
            check3 = mutate(selected.copy())
            check3["Predict"] = predict_DT(check3)
            print("Mutate")
            print(check3)
            print("and error is")
            print(error)
            
            check2 = pd.concat([check2, check3])
            
            if error[0] > 5:
                dataset=check2
                continue   
            else:
                if check(check3):
                    print("The SMILES Solution is")
                    print(check3)
                    print("The iteration is", loop+1)
                    break
                else:
                    dataset = check2
                    continue
        else:
            if check(check2.iloc[0]):
                print("The SMILES Solution is")
                print(check2.iloc[0])
                print("The iteration is", loop+1)
                break
            else:
                dataset = check2
                continue
    else:
        print("passed")
        if check(selected.iloc[0]):
            print("The SMILES Solution is")
            print(selected.iloc[0])
            print("The iteration is", loop+1)
            break
        else:
            dataset = selected
            continue
else:
    print("Maximum number of iterations reached.")



