import pandas as pd
import numpy as np
from ast import literal_eval
import math


#scp "/DATA/vipin_2011mt22/aa/Dataset/target_multimodal_out.npy" "/DATA/vipin_2011mt22/aa/Dataset/source_multimodal_out.npy" "/DATA/vipin_2011mt22/aa/Dataset/labels.npy" rina_1921cs13@172.16.26.59:"/sda/rina_1921cs13/vipin/data/"


ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/abp_hindi.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/News18_India_Hindi.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/alt_hindi.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/boom_hindi.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/nc_hindi.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/vis_hindi.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/wq_hindi.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/News18_India_Bangla.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/boom_bangla.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/News18_India_Tamil.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/News18_India_Telgu.csv')
#ner_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/fc_tamil.csv')

ner_data = ner_data.iloc[:,:].values


source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/abp_hindi.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Hindi.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/alt_hindi.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/boom_hindi.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/nc_hindi.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/vis_hindi.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/wq_hindi.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Bangla.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/boom_bangla.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Tamil.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Telgu.csv')
#source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Source/fc_tamil.csv')
source_data = source_data.drop(columns = ['Unnamed: 0'])
source_data = source_data.iloc[:,:].values

threshold = 0.1

def isConsistent(instanceNum, sourceNum):
    sourceNum = int(sourceNum/4)+2
    ner_data[instanceNum][sourceNum] = literal_eval(ner_data[instanceNum][sourceNum])
    req = threshold*ner_data[instanceNum][sourceNum]['Target_entities']
    req = int(math.ceil(req))
    if ner_data[instanceNum][sourceNum]['Common_entities']<req:
        return False
    else:
        return True

dataset = []
notTaken = []

for i in range(len(ner_data)):
    numSources = 0
    temp = []
    temp.append(source_data[i][0])
    temp.append(0)
    for j in range(2,18,4):
        try:
            tp = type(literal_eval(str(source_data[i][j+2])))
        except:
            tp = type(literal_eval(str(source_data[0][4])))
        
#        if tp is not dict:
#            continue
        if isConsistent(i,j)==False:
            continue
        numSources += 1
        for k in range(4):
            temp.append(source_data[i][j+k])
        print(temp)
    if numSources>0:
        temp[1]=numSources
        dataset.append(temp)
    else:
        notTaken.append(i)

data = pd.DataFrame(dataset, columns =['ID', 'numSources', 'Source_url1', 'Source_text1', 'Image_url1', 'Source_reliability1', 
                                       'Source_url2', 'Source_text2', 'Image_url2', 'Source_reliability2', 
                                       'Source_url3', 'Source_text3', 'Image_url3', 'Source_reliability3', 
                                       'Source_url4', 'Source_text4', 'Image_url4', 'Source_reliability4'])




data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/abp_hindi.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/News18_India_Hindi.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/alt_hindi.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/boom_hindi.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/nc_hindi.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/vis_hindi.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/wq_hindi.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/News18_India_Bangla.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/boom_bangla.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/News18_India_Tamil.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/News18_India_Telgu.csv")
#data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/fc_tamil.csv")




target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/abp_hindi.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Hindi.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/alt_hindi.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/boom_hindi.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/nc_hindi.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/vis_hindi.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/wq_hindi.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Bangla.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/boom_bangla.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Tamil.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Telgu.csv")
#target_data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/fc_tamil.csv")


target_data = target_data.drop(columns = ['Unnamed: 0'])
target_data = target_data.drop(notTaken)
target_data = target_data.reset_index()
target_data = target_data.drop(columns = ['index'])

target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/abp_hindi.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/News18_India_Hindi.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/alt_hindi.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/boom_hindi.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/nc_hindi.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/vis_hindi.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/wq_hindi.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/News18_India_Bangla.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/boom_bangla.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/News18_India_Tamil.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/News18_India_Telgu.csv")
#target_data.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/fc_tamil.csv")

