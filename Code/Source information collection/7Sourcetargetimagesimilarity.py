import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "4"  # specify which GPU(s) to be used
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as  np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import vgg16
from tensorflow.keras import models, Model
import pandas as pd
from ast import literal_eval
from scipy import spatial
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import time
from time import sleep

data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/alt_hindi.csv")
data = data.iloc[:,[1,2,5,9,13,17]].values

# Converting list of imgae urls (stored as string) back to list
for i in range(len(data)):
    for j in range(2,6):
        try:
            data[i][j] = literal_eval(str(data[i][j]))
        except:
            data[i][j] = literal_eval(str(data[0][2]))
vgg = VGG16(include_top=True)
model = Model(vgg.input, vgg.layers[-2].output)
def get_embedding(img):
    img = img_to_array(img)
    if img.shape[2]==1:    
        img = np.stack([img,img,img],axis=2)        
        img = img.reshape(img.shape[0],img.shape[1],3)    
    img = img.reshape((1,) + img.shape)    
    return model.predict(img)
def comp(val):
    return val['Similarity']

final_data = []
t0 = time.time()
for i in range(len(data)): 
    temp_data = []
    temp_data.append(data[i][0])
    target_embedding = ''
    try:
        target_img = Image.open("/DATA/vipin_2011mt22/aa/Dataset/TargetImages/" + data[i][0] + '.jpg')
        target_img = target_img.resize((224,224))
        target_embedding = get_embedding(target_img)
    except:
        temp_data.append([])
        temp_data.append([])
        temp_data.append([])
        temp_data.append([])
        final_data.append(temp_data)
        continue
    for j in range(2,6):
        source_data = []
        for k in range(len(data[i][j])):
            print(i,j,k)          
            name = data[i][0] + '_' + str(j-1) + '_' + str(k)          
            try:
                source_img = Image.open("/DATA/vipin_2011mt22/aa/Dataset/SourceImages/alt_hindi/" + name + '.jpg')
                if min(source_img.size[0],source_img.size[1])>50:            
                    source_img = source_img.resize((224, 224))
                    source_embedding = get_embedding(source_img)
                    sim = cosine_similarity(target_embedding,source_embedding)[0][0]                   
                    source_data.append({"Similarity":sim, "Index":k})
                else:
                    continue                
            except:
                continue
        source_data.sort(key=comp, reverse=True)
        temp_data.append(source_data)
    final_data.append(temp_data)
    print(i, 'Time elapsed:',time.time()-t0,'sec')  
print('Average time per query:',(time.time()-t0)/(len(data)),'seconds.')

df = pd.DataFrame(final_data, columns =['ID','Source1','Source2', 'Source3', 'Source4'])
df.to_csv("/DATA/vipin_2011mt22/aa/Dataset/image_similarity/alt_hindi.csv")

# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)