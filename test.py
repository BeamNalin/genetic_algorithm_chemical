from random import randint as rnd
import random
import pandas as pd
from mlmodel import DT, predict_DT
from RD import check
from error import errorcheck, errorcheck2
import numpy as np
import time as tm

###### Before use test.py -> run mlmodel.py -> run RD.py ########

#input T ที่ต้องการ
# LowerBound = float(input("Lower TEMPARETURE range IS: "))
# UpperBound = float(input("Upper TEMPARETURE range IS: "))

LowerBound = 250
UpperBound = 255

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
    b1_selected = b1.iloc[0:100]
    return b1_selected


# print("Selection")
# selected = rank_selection2(random)
# selected
# print(errorcheck2(selected))

## crossover

def crossover(parent):
    # parent = parent.drop(columns=["Predict", "rank"])
    parent = parent[["C", "Double", "Triple", "Bracket", "Cyclic"]]
    num_parents = parent.shape[0]
    offspring = []
    for i in range(num_parents):
        for j in range(i+1, num_parents):
            p1 = parent.iloc[i].to_numpy()
            p2 = parent.iloc[j].to_numpy()
            pt = np.random.randint(1, p1.shape[0])
            # print(f"the crossover point for parents {i+1} and {j+1} is {pt}")
            off1 = np.concatenate((p1[:pt], p2[pt:]))
            off2 = np.concatenate((p2[:pt], p1[pt:]))
            if check(off1):
                offspring.append(off1)
            if check(off2):
                offspring.append(off2)
    offspring_pd = pd.DataFrame(offspring, columns=["C", "Double", "Triple", "Bracket", "Cyclic"])
    return offspring_pd

# check1=rank_selection2(random)
# print(check1)
# check2=crossover(check1)
# check2["Predict"]=predict_DT(check2)

# print(check2)


## mutate

def mutate_old(check2):
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
 
def mutate(selected):
    selected = selected[["C", "Double", "Triple", "Bracket", "Cyclic"]]
    mutated_rows = pd.DataFrame(columns=["C", "Double", "Triple", "Bracket", "Cyclic"])
    for index in range(100):
        rng = rnd(0, 10)
        mut = selected.iloc[index].to_numpy()
        mut = mut.T
        if rng < 3:
            if check(mut):
                mutated_rows = mutated_rows.append(selected.iloc[index], ignore_index=True)
        else:
            rng = rnd(1, 10)
            if rng > 0:
                c = rnd(mut[0] - 3, mut[0] + 3)
                if c < 0:
                    c=1
            elif rng == 7:
                mut[1] = rnd(0, mut[0] - 1)
            elif rng == 8:
                mut[2] = rnd(0, 2)
            elif rng == 9:
                mut[3] = rnd(0, 5)
            elif rng == 10:
                mut[4] = rnd(0, 3)
            temp = np.arange(5)
            new = np.vstack((mut, temp))
            mutated_row = pd.DataFrame(new, columns=["C", "Double", "Triple", "Bracket", "Cyclic"])
            mutated_row = mutated_row.drop([1])
            if check(mutated_row):
                mutated_rows = mutated_rows.append(mutated_row, ignore_index=True)
    return mutated_rows
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
loop=0
dataset = pop()
dataset.columns = ["C","Double", "Triple", "Bracket", "Cyclic"]
all_selected = pd.DataFrame()
all_crossover = pd.DataFrame()
all_mutate = pd.DataFrame()


dataset.to_csv("random_Tb.csv")
iteration = 100
minerr = 0
start = tm.time()
for loop in range(iteration):
    print("loop", loop+1)
    iter_pd = pd.DataFrame(["iterration",(loop+1)])
    dataset = dataset[["C", "Double", "Triple", "Bracket", "Cyclic"]]
    dataset["Predict"] = predict_DT(dataset)
    # selected = rank_selection2(dataset)
    error = errorcheck(dataset)
    dataset["Error"] = error
    selected = dataset.sort_values('Error').iloc[:100]
    all_selected = pd.concat([all_selected,iter_pd], axis=0)
    all_selected = pd.concat([all_selected,selected], axis=0) # append to main dataframe
    print("Selection")
    print(selected)
    print("and error is")
    print(selected["Error"].iloc[:5])
    
    if selected["Error"].iloc[0] > minerr:
        check2 = crossover(selected)
        check2["Predict"] = predict_DT(check2)
        # check2 = rank_selection2(check2)
        error = errorcheck(check2)
        check2["Error"] = error
        check2 = pd.concat([check2,selected], axis=0)
        check2 = check2.sort_values('Error')
        all_crossover = pd.concat([all_crossover,iter_pd], axis=0)
        all_crossover = pd.concat([all_crossover,check2], axis=0) # append to main dataframe
        print("Crossover")
        print(check2)
        print("and error is")
        print(check2["Error"].iloc[:5])
        
        if check2["Error"].iloc[0] > minerr:
            check3 = mutate(check2.copy())
            check3["Predict"] = predict_DT(check3)
            error = errorcheck(check3)
            check3["Error"] = error

            all_mutate = pd.concat([all_mutate, iter_pd], axis=0) # append to main dataframe
            all_mutate = pd.concat([all_mutate, check3], axis=0) # append to main dataframe
            print("Mutate")
            print(check3)
            print("and error is")
            print(check3["Error"].iloc[:5])
            
            check2 = pd.concat([check2, check3])
            
            if check2["Error"].iloc[0] > minerr:
                dataset=check2
                continue   
            else:
                if check(check3.iloc[0]):
                    print("The SMILES Solution is")
                    print(check3.loc[check3['Error'] <= minerr])
                    print("The iteration is", loop+1)
                    break
                else:
                    dataset = check2
                    continue
        else:
            if check(check2.iloc[0]):
                print("The SMILES Solution is")
                print(check2.loc[check2["Error"] <= minerr])
                print("The iteration is", loop+1)
                break
            else:
                dataset = check2
                continue
    else:
        print("passed")
        if check(selected.iloc[0]):
            print("The SMILES Solution is")
            print(selected.loc[selected["Error"] <= minerr])
            print("The iteration is", loop+1)
            break
        else:
            dataset = selected
            continue
else:
    print("Maximum number of iterations reached.")
    if check(dataset.iloc[0]) == True:
            print("The closest SMILES Solution is")
            print(selected.iloc[0])
            print("The iteration is",loop+1)
    else:
        print("No result was found")

end = tm.time()
print("Elapse Time:",end-start)
all_selected.to_csv("selected_tb.csv")
all_crossover.to_csv("crossover_tb.csv")
all_mutate.to_csv("mutate_tb.csv")

