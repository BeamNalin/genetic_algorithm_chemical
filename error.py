#error module
def errorcheck(data):
    showerr =[]
    for i in data["predict"]:
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