import pandas as pd
import numpy as np
import json

target = pd.read_excel("/DATA/vipin_2011mt22/aa/raw data/with label/fc_tamil_l.xlsx")
source = pd.read_excel("/DATA/vipin_2011mt22/aa/translate/with label/fc_tamil_tran_text_l.xlsx")
target_info = target.iloc[:,[2,4,7]].values       #url, claim, label
#target_info = target.iloc[:,[0,2,3]].values       #url, claim, label
source_info = source.iloc[:,:].values

with open("/DATA/vipin_2011mt22/a/CIKM_resource_2021/hindi/MMFND-main/Dataset/MBFC/factuality.json") as f:
    data = json.load(f)

def reliability(dict, key):
    if key in dict.keys():
        return dict[key]
    else:
        return 'UNKNOWN'
    
dict = {}
for p in data:
    pos = 0
    for i in range(len(p)):
        if p[i] == '.':
            pos = i
            break
    dict[p[:pos]] = data[p]
    
def reliability_from_link(link):
    url = str(link)
    if len(url)<4:
        return 'UNKNOWN'
    if url[:5] == 'https':
        url = url[8:]
    else:
        url = url[7:]
    pos = 0
    for i in range(len(url)):
        if url[i] == '.':
            pos = i
            break
    if reliability(dict,url[:pos]) != 'UNKNOWN':
        return reliability(dict,url[:pos])
    url = url[pos+1:]
    pos = len(url)
    for i in range(len(url)):
        if url[i] == '.':
            pos = i
            break
    
    if reliability(dict,url[:pos]) != 'UNKNOWN':
        return reliability(dict,url[:pos])
    else:
        return 'UNKNOWN'

def get_label(org):
    org = str(org)
    if org == 'mixture':
        return org
    if org=='Real' or org=='1' or org == 'TRUE' or org=='True' or org=='true':
        return 'REAL'
    if org=='Fake' or org=='0' or org == 'FALSE' or org=='False' or org=='false':
        return 'FAKE'
    print('New Label Found: ',org)
    return org

source_data = []
target_data = []

def filter(index):
    if source_info[index][1] == 0:
        return
    
    label = get_label(target_info[index][2])
    if label == 'mixture':
        return
    
    temp = []
    temp1 = []
    temp1.append('fc_tamil' + str(source_info[index][0])) # id representing instance from original data
    temp1.append(target_info[index][0]) # target url
#     temp1.append('https://www.reuters.com/') # target url
    temp1.append(target_info[index][1]) # target text
    temp1.append(source_info[index][2]) # image url
    temp1.append(label) # target label
    target_data.append(temp1)
    temp.append('fc_tamil' + str(source_info[index][0])) # id representing instance from original data
    temp.append(source_info[index][1]) # numSources
    
    for i in range(3,15,3):
        temp.append(source_info[index][i])    # Source_url
        temp.append(source_info[index][i+1])  # Source_text
        temp.append(source_info[index][i+2])  # Source_imge_url
        temp.append(reliability_from_link(source_info[index][i])) # Source_reliability
    source_data.append(temp)
    
for i in range(len(source_info)):
    filter(i)

print('Total instances: ' , len(source_info), ', Final instances: ' , len(source_data))

df = pd.DataFrame(source_data, columns =['ID', 'numSources','Source_url1','Source_text1','Image_url1','Source_reliability1','Source_url2','Source_text2','Image_url2','Source_reliability2','Source_url3','Source_text3','Image_url3','Source_reliability3','Source_url4','Source_text4','Image_url4','Source_reliability4'])
df1 = pd.DataFrame(target_data, columns =['ID','Target_url','Target_text', 'Image_url', 'label'])
df.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/fc_tamil.csv")
df1.to_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/fc_tamil.csv")

