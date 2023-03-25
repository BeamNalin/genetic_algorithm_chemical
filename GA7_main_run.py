from random import randint as rnd
import random
import pandas as pd
from mlmodel_Hansen_dispersion import model, predict_DT_Hansen_dis
from mlmodel_Hansen_Hbondn import model, predict_DT_Hansen_Hbond
from mlmodel_Hansen_polarity import model, predict_DT_Hansen_Polarity
from mlmodel_Hildebrand import model, predict_DT_Hildebrand
from mlmodel_LogP import model, predict_DT_logP
from mlmodel_LogS import model, predict_DT_LogS
from RD2 import check
from error import errorcheck, errorcheck2
import numpy as np

###### Before use test2.py -> run all mlmodel.py -> run RD2.py ########
#input Hansen_dis
Hansen_dis_LowerBound = 10
Hansen_dis_UpperBound = 20

#input Hansen_Hbond
Hansen_Hbond_LowerBound = 1
Hansen_Hbond_UpperBound = 3

#input Hansen_Polarity
Hansen_Polarity_LowerBound = 3
Hansen_Polarity_UpperBound = 5

#input LogS
LogS_LowerBound = -3
LogS_UpperBound = -1


#input LogP
LogP_LowerBound = -3
LogP_UpperBound = -1

#error weigh
LogP_weight =  0.2
LogS_weight = 0.2
Hansen_Polarity_weight= 0.2
Hansen_Hbond_weight = 0.2
Hansen_dis_weight = 0.2
#------------------------------------------------------#
parameter_bounds = {
    "LogS": [LogS_LowerBound, LogS_UpperBound],
    "LogP": [LogP_LowerBound, LogP_UpperBound],
    "Hansen_Polarity": [Hansen_Polarity_LowerBound, Hansen_Polarity_UpperBound],
    "Hansen_dis": [Hansen_dis_LowerBound, Hansen_dis_UpperBound],
    "Hansen_Hbond": [Hansen_Hbond_LowerBound, Hansen_Hbond_UpperBound],
}
### Note ####
# ในทั้งหมด 6 ตัวเรา concern ค่าไหนที่สุดว่าอยากให้ถูกต้องที่สุด เพราะเวลาเรา rank selection จะได้ order ได้ว่าให้เรียงจากค่านอยสุดของตัวแปลไหนก่อน

#---------------------------------------------------------#

#add percentage error of 6 input
def errorcheck_old(data):
    showerr = []
    for i in range(len(data)):
        error = []
        if data.iloc[i]["Predict_LogS"] < LogS_LowerBound:
            error += (abs(data.iloc[i]["Predict_LogS"] - LogS_LowerBound) / LogS_LowerBound) * 100
        elif data.iloc[i]["Predict_LogS"] > LogS_UpperBound:
            error += (abs(data.iloc[i]["Predict_LogS"] - LogS_UpperBound) / LogS_UpperBound) * 100

        
        if data.iloc[i]["Predict_LogP"] < LogP_LowerBound:
            error += (abs(data.iloc[i]["Predict_LogP"] - LogP_LowerBound) / LogP_LowerBound) * 100
        elif data.iloc[i]["Predict_LogP"] > LogP_UpperBound:
            error += (abs(data.iloc[i]["Predict_LogP"] - LogP_UpperBound) / LogP_UpperBound) * 100
            
        if data.iloc[i]["Predict_Hansen_dis"] < Hansen_dis_LowerBound:
            error += (abs(data.iloc[i]["Predict_Hansen_dis"] - Hansen_dis_LowerBound) / Hansen_dis_LowerBound) * 100
        elif data.iloc[i]["Predict_Hansen_dis"] > Hansen_dis_UpperBound:
            error += (abs(data.iloc[i]["Predict_Hansen_dis"] - Hansen_dis_UpperBound) / Hansen_dis_UpperBound) * 100
            
        if data.iloc[i]["Predict_Hansen_H_bond"] < Hansen_Hbond_LowerBound:
            error += (abs(data.iloc[i]["Predict_Hansen_H_bond"] - Hansen_Hbond_LowerBound) / Hansen_Hbond_LowerBound) * 100
        elif data.iloc[i]["Predict_Hansen_H_bond"] > Hansen_Hbond_UpperBound:
            error += (abs(data.iloc[i]["Predict_Hansen_H_bond"] - Hansen_Hbond_UpperBound) / Hansen_Hbond_UpperBound) * 100
            
        if data.iloc[i]["Predict_Hansen_Polarity"] < Hansen_Polarity_LowerBound:
            error += (abs(data.iloc[i]["Predict_Hansen_Polarity"] - Hansen_Polarity_LowerBound) / Hansen_Polarity_LowerBound) * 100
        elif data.iloc[i]["Predict_Hansen_Polarity"] > Hansen_Polarity_UpperBound:
            error += (abs(data.iloc[i]["Predict_Hansen_Polarity"] - Hansen_Polarity_UpperBound) / Hansen_Polarity_UpperBound) * 100

        error[0] *= LogS_weight
        error[1] *= LogP_weight
        error[2] *= Hansen_dis_weight
        error[3] *= Hansen_Hbond_weight
        error[4] *= Hansen_Polarity_weight
        showerr.append(error)
        print(showerr)
    return showerr

def errorcheck(data):
    showerr = []
    for i in range(len(data)):
        error = [0] * 5  # Initialize errors with 0
        if data.iloc[i]["Predict_LogS"] < LogS_LowerBound:
            error[0] += (abs(data.iloc[i]["Predict_LogS"] - LogS_LowerBound) / LogS_LowerBound) * 100
        elif data.iloc[i]["Predict_LogS"] > LogS_UpperBound:
            error[0] += (abs(data.iloc[i]["Predict_LogS"] - LogS_UpperBound) / LogS_UpperBound) * 100

        if data.iloc[i]["Predict_LogP"] < LogP_LowerBound:
            error[1] += (abs(data.iloc[i]["Predict_LogP"] - LogP_LowerBound) / LogP_LowerBound) * 100
        elif data.iloc[i]["Predict_LogP"] > LogP_UpperBound:
            error[1] += (abs(data.iloc[i]["Predict_LogP"] - LogP_UpperBound) / LogP_UpperBound) * 100

        if data.iloc[i]["Predict_Hansen_dis"] < Hansen_dis_LowerBound:
            error[2] += (abs(data.iloc[i]["Predict_Hansen_dis"] - Hansen_dis_LowerBound) / Hansen_dis_LowerBound) * 100
        elif data.iloc[i]["Predict_Hansen_dis"] > Hansen_dis_UpperBound:
            error[2] += (abs(data.iloc[i]["Predict_Hansen_dis"] - Hansen_dis_UpperBound) / Hansen_dis_UpperBound) * 100

        if data.iloc[i]["Predict_Hansen_H_bond"] < Hansen_Hbond_LowerBound:
            error[3] += (abs(data.iloc[i]["Predict_Hansen_H_bond"] - Hansen_Hbond_LowerBound) / Hansen_Hbond_LowerBound) * 100
        elif data.iloc[i]["Predict_Hansen_H_bond"] > Hansen_Hbond_UpperBound:
            error[3] += (abs(data.iloc[i]["Predict_Hansen_H_bond"] - Hansen_Hbond_UpperBound) / Hansen_Hbond_UpperBound) * 100

        if data.iloc[i]["Predict_Hansen_Polarity"] < Hansen_Polarity_LowerBound:
            error[4] += (abs(data.iloc[i]["Predict_Hansen_Polarity"] - Hansen_Polarity_LowerBound) / Hansen_Polarity_LowerBound) * 100
        elif data.iloc[i]["Predict_Hansen_Polarity"] > Hansen_Polarity_UpperBound:
            error[4] += (abs(data.iloc[i]["Predict_Hansen_Polarity"] - Hansen_Polarity_UpperBound) / Hansen_Polarity_UpperBound) * 100

        error[0] *= LogS_weight
        error[1] *= LogP_weight
        error[2] *= Hansen_dis_weight
        error[3] *= Hansen_Hbond_weight
        error[4] *= Hansen_Polarity_weight
        showerr.append(error)
        print(showerr)
    return showerr

#-----------------------------------------------------------------#

# Random
def pop():
    count = 0
    cc =[]
    while count <  1000 :
        carbon = [rnd(1, 12), rnd(0, 6), rnd(0, 4), rnd(0, 10),rnd(0,2) , rnd(0, 4),rnd(0, 10), rnd(0, 3)]  #มาแก้ตรงนี้ค่าที่สุ่ม
        checkk = check(carbon)
        if checkk == True:
            cc.append(carbon)
            count = count+1
        else:
            count = count

    return pd.DataFrame(cc)

# ----------------------------------------------------------------------------------------#
# 2.LogS 3.LogP 4. Hansen_dis 5.Hansen_polarity 6.Hansen_Hbond

#Selection
def rank_selection(population):
    ranks = []
    for i in range(len(population)):
        rank = 0
        for prop, lb, ub in [('Predict_Hansen_dis', Hansen_dis_LowerBound, Hansen_dis_UpperBound),
                             ('Predict_LogP', LogP_LowerBound, LogP_UpperBound), 
                             ('Predict_LogS', LogS_LowerBound, LogS_UpperBound),
                             ('Predict_Hansen_H_bond', Hansen_Hbond_LowerBound, Hansen_Hbond_UpperBound), 
                             ('Predict_Hansen_Polarity', Hansen_Polarity_LowerBound, Hansen_Polarity_UpperBound)]:
            if population.iloc[i][prop] < lb:
                rank += abs(population.iloc[i][prop] - lb)
            elif population.iloc[i][prop] > ub:
                rank += abs(population.iloc[i][prop] - ub)
        ranks.append(rank)
    return ranks

def rank_selection2(population):
    selected_columns = ["Predict_Hansen_dis", "Predict_LogP", "Predict_LogS", "Predict_Hansen_H_bond", "Predict_Hansen_Polarity"]
    population['rank'] = rank_selection(population[selected_columns])
    selected = population.sort_values('rank').iloc[:128]
    return selected

#----------------------------------------------------------------#
## crossover

def crossover(parent):
    parent = parent.drop(columns=["Predict_Hansen_dis","Predict_LogP","Predict_LogS","Predict_Hansen_H_bond","Predict_Hansen_Polarity", "rank"])
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
    offspring_pd = pd.DataFrame(offspring, columns=['CRe', 'DoubleCCRe', 'TripleCC', 'Bracket', 'Benzene', 'CycleRe', 'SingleCO', 'DoubleCO'])
    return offspring_pd

#------------------ mutate-------------------#

def mutate(check22):
    check22= check22[["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"]]
    rng=rnd(0,10)
    if rng < 3:
        print("No Mutation")
        return check2
    else:
        rng=rnd(1,10)
        col=rnd(0,8)
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
    temp=np.arange(8)
    new=np.vstack((mut,temp))
    mut_pd=pd.DataFrame(new, columns=["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"])
    mut_pd=mut_pd.drop([1])
    return mut_pd
 
def errorcheck(data):
    showerr = []
    for i in range(len(data)):
        errors = []
        if data.iloc[i]["Predict_LogS"] < LogS_LowerBound:
            errors.append((abs(data.iloc[i]["Predict_LogS"] - LogS_LowerBound) / LogS_LowerBound) * 100 * LogS_weight)
        elif data.iloc[i]["Predict_LogS"] > LogS_UpperBound:
            errors.append((abs(data.iloc[i]["Predict_LogS"] - LogS_UpperBound) / LogS_UpperBound) * 100 * LogS_weight)
        else:
            errors.append(0)
        
        if data.iloc[i]["Predict_LogP"] < LogP_LowerBound:
            errors.append((abs(data.iloc[i]["Predict_LogP"] - LogP_LowerBound) / LogP_LowerBound) * 100 * LogP_weight)
        elif data.iloc[i]["Predict_LogP"] > LogP_UpperBound:
            errors.append((abs(data.iloc[i]["Predict_LogP"] - LogP_UpperBound) / LogP_UpperBound) * 100 * LogP_weight)
        else:
            errors.append(0)
            
        if data.iloc[i]["Predict_Hansen_dis"] < Hansen_dis_LowerBound:
            errors.append((abs(data.iloc[i]["Predict_Hansen_dis"] - Hansen_dis_LowerBound) / Hansen_dis_LowerBound) * 100 * Hansen_dis_weight)
        elif data.iloc[i]["Predict_Hansen_dis"] > Hansen_dis_UpperBound:
            errors.append((abs(data.iloc[i]["Predict_Hansen_dis"] - Hansen_dis_UpperBound) / Hansen_dis_UpperBound) * 100 * Hansen_dis_weight)
        else:
            errors.append(0)
            
        if data.iloc[i]["Predict_Hansen_H_bond"] < Hansen_Hbond_LowerBound:
            errors.append((abs(data.iloc[i]["Predict_Hansen_H_bond"] - Hansen_Hbond_LowerBound) / Hansen_Hbond_LowerBound) * 100 * Hansen_Hbond_weight)
        elif data.iloc[i]["Predict_Hansen_H_bond"] > Hansen_Hbond_UpperBound:
            errors.append((abs(data.iloc[i]["Predict_Hansen_H_bond"] - Hansen_Hbond_UpperBound) / Hansen_Hbond_UpperBound) * 100 * Hansen_Hbond_weight)
        else:
            errors.append(0)
            
        if data.iloc[i]["Predict_Hansen_Polarity"] < Hansen_Polarity_LowerBound:
            errors.append((abs(data.iloc[i]["Predict_Hansen_Polarity"] - Hansen_Polarity_LowerBound) / Hansen_Polarity_LowerBound) * 100 * Hansen_Polarity_weight)
        elif data.iloc[i]["Predict_Hansen_Polarity"] > Hansen_Polarity_UpperBound:
            errors.append((abs(data.iloc[i]["Predict_Hansen_Polarity"] - Hansen_Polarity_UpperBound) / Hansen_Polarity_UpperBound) * 100 * Hansen_Polarity_weight)
        else:
            errors.append(0)

        avg_error = abs(sum(errors) / len(errors))
        showerr.append(avg_error)
        
    return showerr

# #wrtie csv file
# random.to_csv("random4.csv")
# selected.to_csv("selected4.csv")
# check2.to_csv("crossover4.csv")
# check3.to_csv("mutate4.csv")

#------------------------- automaic code ----------------------#

loop = 0
countloop = []
dataset = pop()

dataset.columns = ["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"]
iteration = 10
for loop in range(iteration):
    print("loop", loop+1)
    dataset = dataset[["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"]]
    dataset_use=dataset.copy()
    dataset["Predict_LogS"] = predict_DT_LogS(dataset_use)
    dataset["Predict_LogP"] = predict_DT_logP(dataset_use)
    dataset["Predict_Hansen_Polarity"] = predict_DT_Hansen_Polarity(dataset_use)
    dataset["Predict_Hansen_dis"] = predict_DT_Hansen_dis(dataset_use)
    dataset["Predict_Hansen_H_bond"] = predict_DT_Hansen_Hbond(dataset_use)
    selected = rank_selection2(dataset)
    error = errorcheck(selected)
    selected["error"] = error
    selected = selected.sort_values('error')
    print("Selection")
    print(selected)
    print("and error is")
    print(selected["error"])
    if selected["error"].iloc[0] > 0:
        check1=rank_selection2(dataset)
        check2=crossover(check1)
        print(check2)
        check22 = check2.copy()
        check22["Predict_LogS"] = predict_DT_LogS(check2)
        check22["Predict_LogP"] = predict_DT_logP(check2)
        check22["Predict_Hansen_Polarity"] = predict_DT_Hansen_Polarity(check2)
        check22["Predict_Hansen_dis"] = predict_DT_Hansen_dis(check2)
        check22["Predict_Hansen_H_bond"] = predict_DT_Hansen_Hbond(check2)
        error = errorcheck(check22)
        check22["error"] = error
        check22 = check22.sort_values('error')
        print("Crossover")
        print(check2)
        print("and error is")
        print(check22["error"])
        if check22["error"].iloc[0] > 0:
            ### last step that not working ####
            check3=mutate(check22)
            check33=check3.copy()
            check33["Predict_LogS"] = predict_DT_LogS(check3)
            check33["Predict_LogP"] = predict_DT_logP(check3)
            check33["Predict_Hansen_Polarity"] = predict_DT_Hansen_Polarity(check3)
            check33["Predict_Hansen_dis"] = predict_DT_Hansen_dis(check3)
            check33["Predict_Hansen_H_bond"] = predict_DT_Hansen_Hbond(check3)
            error = errorcheck(check33)
            check33["error"] = error
            check33 = check33.sort_values('error')
            print("Mutate")
            print(check33)
            print("and error is")
            print(error)
            check22 = pd.concat([check22, check33])
            if check33["error"].iloc[0] > 0:
                dataset = check22
                continue
            else:
                if check(check3) == True:
                    print("The SMILES Solution is")
                    print(check3)
                    print("The iteration is",loop)
                    break
                else:
                    dataset = check22
                    continue
        else:
            if check(check2.iloc[0]) == True:
                print("The SMILES Solution is")
                print(check2.iloc[0])
                print("The iteration is",loop)
                break
            else:
                dataset = check22
                continue
    else:
        if check(selected.iloc[0]) == True:
            print("The SMILES Solution is")
            print(selected.iloc[0])
            print("The iteration is",loop)
            break
        else:
            dataset = selected
            continue