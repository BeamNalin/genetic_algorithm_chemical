from random import randint as rnd
import random
import pandas as pd
from mlmodel import DT, predict_DT
from mlmodel_Hansen_dispersion import model, predict_DT_Hansen_dis
from mlmodel_Hansen_Hbondn import model, predict_DT_Hansen_Hbond
from mlmodel_Hansen_polarity import model, predict_DT_Hansen_Polarity
from mlmodel_Hildebrand import model, predict_DT_Hildebrand
from mlmodel_LogP import model, predict_DT_logP
from mlmodel_LogS import model, predict_DT_LogS
from RD import check
from error import errorcheck, errorcheck2
import numpy as np

###### Before use test2.py -> run all mlmodel.py -> run RD2.py ########
#input range ที่ต้องการ
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
    while count <  10 :
        carbon = [rnd(1, 12), rnd(0, 6), rnd(0, 4), rnd(0, 10),rnd(0,2) , rnd(0, 4),rnd(0, 10), rnd(0, 3)]  #มาแก้ตรงนี้ค่าที่สุ่ม
        checkk = check(carbon)
        if checkk == True:
            cc.append(carbon)
            count = count+1
        else:
            count = count

    return pd.DataFrame(cc)

random =pop()
random.columns = ["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"]
random

#Predict -> ต้องเชื่อมกับตัว ML -> เดี๋ยวต่อกันอีกที
random["Predict"] = predict_DT_LogS(random)
print("Random dataset")
print(random)


#### random ตรงนี้มี predict แล้ว
# ----------------------------------------------------------------------------------------#

#Selection
def rank_selection(a):
    rankk =[]
    for i in random["Predict"]:
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

print("Selection")
selected = rank_selection2(random)
selected
print(errorcheck2(selected))

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
    offspring_pd = pd.DataFrame(offspring, columns=["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"])
    return offspring_pd


check1=rank_selection2(random)
print(check1)
check2=crossover(check1)
print(check2)
check2["Predict"]=predict_DT_LogS(check2)

## mutate

def mutate(check2):
    check2=check2.drop(columns=["Predict"])
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
        if col == 0:
            c=rnd(1,12)
            mut[0]=c
        elif col == 1:
            mut[1]=rnd(0,mut[0]-1)
        elif col == 2:
            mut[2]=rnd(0,2)
        elif col == 3:
            mut[3]=rnd(0,5)
        elif col == 4:
            mut[4]=rnd(0,3)
        elif col == 5:
            mut[5]=rnd(0,10)
        elif col == 6:
            mut[6]=rnd(0,3)
    temp=np.arange(7)
    new=np.vstack((mut,temp))
    mut_pd=pd.DataFrame(new, columns=["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"])
    mut_pd=mut_pd.drop([1])
    return mut_pd
 
print("Mutate") 
check3=mutate(check2)
check3["Predict"]=predict_DT_LogS(check3)
print(check3)



#distance check after mutate
for i in check3["Predict"]:
        if i < LowerBound:
            check3["rank"] = float(abs(i-LowerBound))
        elif i > UpperBound:
            check3["rank"] = float(abs(i-UpperBound))
        else:
            check3["rank"] = 0
check3


#wrtie csv file
random.to_csv("random4.csv")
selected.to_csv("selected4.csv")
check2.to_csv("crossover4.csv")
check3.to_csv("mutate4.csv")



# automaic code



loop = 0
countloop = []
while loop < 100:
    loop +=1
    random =pop()
    random.columns = ["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"]
    random["Predict"] = predict_DT_LogS(random)
    selected = rank_selection2(random)
    error = errorcheck(selected)
    print("Selection")
    print(selected)
    print("and error is")
    print(error)
    if error[0] > 0:
        check2=crossover(check1)
        check2["Predict"]=predict_DT_LogS(check2)
        error = errorcheck(check2)
        print("Crossover")
        print(check2)
        print("and error is")
        print(error)
        if error[0] > 0:
            check3=mutate(check2)
            check3["Predict"]=predict_DT_LogS(check3)
            error = errorcheck(check3)
            print("Mutate")
            print(check3)
            print("and error is")
            print(error)
            if error[0] > 0:
                continue
            else:
                if check(check3) == True:
                    print("The SMILES Solution is")
                    print(check3)
                    print("The iteration is",loop)
                    break
                else:
                    continue
        else:
            if check(check2.iloc[0]) == True:
                print("The SMILES Solution is")
                print(check2.iloc[0])
                print("The iteration is",loop)
                break
            else:
                continue
    else:
        if check(selected.iloc[0]) == True:
            print("The SMILES Solution is")
            print(selected.iloc[0])
            print("The iteration is",loop)
            break
        else:
            continue