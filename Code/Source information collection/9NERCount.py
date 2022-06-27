import pandas as pd
import numpy as np
import time

#import spacy


##Bengali---------use tfgpu1
import bn_core_news_sm
nlp = bn_core_news_sm.load()
#
##English---------use tfgpu1
##nlp = spacy.load('en_core_web_lg')
#
nlp.max_length = 4000000
def get_ner(doc):
    doc = str(doc)
    if len(doc)>3000000:
        doc = doc[0:3000000]
    doc = nlp(doc)    
    res = set()    
    for ent in doc.ents:
        res.add(ent.text)    
    return res
    
#Hindi-------use nischal
#from HindiNLP.HindiNer import NER
#import re
#detect_ner = NER()
#
#def get_ner(doc):
#    doc = str(doc)
#    if len(doc)>3000000:
#        doc = doc[0:3000000] 
#    sentence = detect_ner.Predict(doc) 
#    try:
#        entitis = re.findall('"([^"]*)"',sentence.split(" → ",1)[1]) 
#    except:
#        entitis = []
#    res = set()    
#    for ent in entitis:
#        res.add(ent)    
#    return res
    
#CUDA_VISIBLE_DEVICES=3 python "/DATA/vipin_2011mt22/aa/Dataset/9NERCount.py"    

target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/abp_hindi.csv")
target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Hindi.csv")
#target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/alt_hindi.csv")
target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/boom_hindi.csv")
target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/nc_hindi.csv")
target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/vis_hindi.csv")
target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/wq_hindi.csv")
target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Bangla.csv")
target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/boom_bangla.csv")
target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Tamil.csv")
#target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Telgu.csv")
target = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/fc_tamil.csv")

source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/abp_hindi.csv")
source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Hindi.csv")
#source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/alt_hindi.csv")
source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/boom_hindi.csv")
source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/nc_hindi.csv")
source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/vis_hindi.csv")
source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/wq_hindi.csv")
source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Bangla.csv")
source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/boom_bangla.csv")
source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Tamil.csv")
#source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Telgu.csv")
source = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/fc_tamil.csv")

source = source.drop(['Unnamed: 0', 'numSources', 
             'Source_url1', 'Image_url1', 'Source_reliability1', 
             'Source_url2', 'Image_url2', 'Source_reliability2',
             'Source_url3', 'Image_url3', 'Source_reliability3',
             'Source_url4', 'Image_url4', 'Source_reliability4',], axis=1)

for i in range(10):
    target_entities = get_ner(target['Target_text'][i])
    print('################', i, '####################')
    print(target_entities)
    for j in range(1,5):
        source_entities = get_ner(source['Source_text'+str(j)][i])
        print(i,j,target_entities.intersection(source_entities))

t0 = time.time()
for i in range(source.shape[0]):
    target_entities = get_ner(target['Target_text'][i])  
    for j in range(1,5):        
        source_entities = get_ner(source['Source_text'+str(j)][i])        
        source['Source_text'+str(j)][i] = {'Common_entities': len(target_entities.intersection(source_entities)),  'Target_entities': len(target_entities)}
    print(i,'Time elapsed: ',time.time()-t0, 'Seconds')

#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/abp_hindi.csv')
#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/News18_India_Hindi.csv')
#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/alt_hindi.csv')
#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/boom_hindi.csv')
#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/nc_hindi.csv')
#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/vis_hindi.csv')
#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/wq_hindi.csv')
#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/News18_India_Bangla.csv')
#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/boom_bangla.csv')
#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/News18_India_Tamil.csv')
#source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/News18_India_Telgu.csv')
source.to_csv('/DATA/vipin_2011mt22/aa/Dataset/NER/fc_tamil.csv')

