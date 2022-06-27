import os
import requests
import urllib.parse
import pandas as pd
import numpy as np
import time
from time import sleep
from ast import literal_eval


data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/abp_hindi.csv")
data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/alt_hindi.csv")
data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/boom_hindi.csv")
data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/nc_hindi.csv")
data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/vis_hindi.csv")
data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/wq_hindi.csv")
data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/boom_bangla.csv")
data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Bangla.csv")
data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Hindi.csv")
data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Tamil.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Source/News18_India_Telgu.csv")


data = data.iloc[:,[1,2,5,9,13,17]].values
data.shape
data[0][2][0] # It will return a sigle charecter (not complete image link) because list of links are stored as string

# Converting list of imgae urls (stored as string) back to list
for i in range(len(data)):
    for j in range(2,6):
        try:
            data[i][j] = literal_eval(str(data[i][j]))
        except:
            data[i][j] = literal_eval(str(data[0][2]))
      
data[0][2][0]

t0 = time.time()
#for i in range(5000,len(data)):
for i in range(0,len(data)):
    for j in range(2,6):
        for k in range(len(data[i][j])):
            url = data[i][j][k]
      
            name = data[i][0] + '_' + str(j-1) + '_' + str(k) + '.jpg'
            if os.path.exists("/DATA/vipin_2011mt22/aa/Dataset/SourceImages/news18_tamil/" + name):
                continue
                
            try:
                urllib.request.urlretrieve(url, "/DATA/vipin_2011mt22/aa/Dataset/SourceImages/news18_tamil/" + name)
                print('Downloaded Image:',i,j,k)
            except:
                print('Could not Download Image:',i,j,k)
#            try:
#                
#                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"} 
#                res = requests.get(url,headers=headers,timeout=4)
#                with open("/DATA/vipin_2011mt22/aa/Dataset/SourceImages/" + name, "wb") as f:
#                    f.write(res.content)
#                print('Downloaded Image',i,j,k)
#            except:
#                print('Unable to Download',i,j,k)
#                sleep(1)        
    print(i, 'Time elapsed:',time.time()-t0,'sec')
print('Average time per query:',(time.time()-t0)/len(data),'seconds.')
