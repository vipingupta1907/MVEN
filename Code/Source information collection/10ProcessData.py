import numpy as np
import pandas as pd
from ast import literal_eval

target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/alt_hindi.csv")
target = target.drop(columns = ['Unnamed: 0'])
target.to_csv("/DATA/vipin_2011mt22/aa/Dataset/process_data/source/alt_hindi.csv")

source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/alt_hindi.csv")
similarity = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/image_similarity/alt_hindi.csv")
similarity = similarity.iloc[:,:].values

for i in range(len(similarity)):   
    for j in range(2,6):        
        similarity[i][j] = literal_eval(str(similarity[i][j]))        
        name = 'Image_url' + str(j-1)        
        source[name][i] = literal_eval(str(source[name][i]))        
        temp = {}        
        if len(similarity[i][j])>0:
            temp['image_name'] = source['ID'][i] + '_' + str(j-1) + '_' + str(similarity[i][j][0]) + '.jpg'
            temp['image_url'] = source[name][i][similarity[i][j][0]]
            source[name][i] = temp
        else:
            source[name][i] = []

source = source.drop(columns = ['Unnamed: 0'])
source.to_csv("/DATA/vipin_2011mt22/aa/Dataset/process_data/source/alt_hindi.csv")
