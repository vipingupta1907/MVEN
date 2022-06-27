# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from PIL import Image
from numpy import asarray

source_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Source/ReCovery.csv')
source_data = source_data.head()

target_data = pd.read_csv('/DATA/vipin_2011mt22/aa/Dataset/Final_Data/Target/ReCovery.csv')
target_data =target_data.head()
image_data = target_data.iloc[:,[1,4]].values

source_text=[]
target_text=[]
IDs=[]
labels=[]
img_array = []

for index, row in source_data.iterrows():
    final_text = str(row.Source_text1) + str(row.Source_text2) + str(row.Source_text3) + str(row.Source_text4)
    source_text.append(final_text)
    IDs.append(row.ID)

for index, row in target_data.iterrows():
    target_text.append(str(row.Target_text))
    labels.append(row.label)  
    
for i in range(len(image_data)):
    url = image_data[i][1]
    print(image_data[i][1])
    print(image_data[i][0])
    try:
        path = '/DATA/vipin_2011mt22/aa/Dataset/TargetImages/' + image_data[i][0] + '.jpg'
        img = Image.open(path)
    except:
        path = '/DATA/vipin_2011mt22/aa/Dataset/TargetImages/' + 'alt_hindi1' + '.jpg'
        img = Image.open(path)
    img_array.append(asarray(img))  

concat_text = np.c_[source_text, target_text ]
concat_img = np.c_[img_array, img_array ]

np.save('/DATA/vipin_2011mt22/aa/Dataset/text.npy' , concat_text)
np.save('/DATA/vipin_2011mt22/aa/Dataset/ID.npy' , IDs)
np.save('/DATA/vipin_2011mt22/aa/Dataset/label.npy' , labels)
np.save('/DATA/vipin_2011mt22/aa/Dataset/image.npy' , concat_img)

######################################################################################


    


    
    