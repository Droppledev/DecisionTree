import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

def loadDataset(filename, load = 1):
    ''' load dataset from csv to pandas dataframe '''
    dataset = pd.read_csv(filename,header=None)
    if load == 1:
        dataset.columns = ["NumTimesPrg", "PlGlcConc", "BloodP",
        "SkinThick", "TwoHourSerIns", "BMI",
        "DiPedFunc", "Age", "HasDiabetes"]
    
    dataset = normalizeDataset(dataset)
    dataset_split = splitDataset(dataset,load)
    
    training_set = dataset_split[0]
    test_set = dataset_split[1]

    return training_set,test_set

def normalizeDataset(df):
    ''' normalize each column of dataset '''
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df

def splitDataset(df, load):
    split = 10
    if load == 1:
        kf = StratifiedKFold(n_splits=split)
        X = df.iloc[:, :-1]
        Y = df.iloc[:,-1]
        result = next(kf.split(X,Y), None)

    elif load == 2:
        kf = KFold(n_splits=split)
        result = next(kf.split(df), None)
    
    print (result)
    return result


def getPivotTable():
    pass

def getGiniIndex():
    pass

def getGiniSplit():
    pass

def getEntropy():
    pass

def getGainSplit():
    pass

def misclassificationError():
    pass

def main():
    dataset = loadDataset('pima-indians-diabetes.csv')
    print(dataset)

main()