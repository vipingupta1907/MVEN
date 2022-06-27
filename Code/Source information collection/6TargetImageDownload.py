import requests
import urllib.parse
import pandas as pd
import numpy as np
import time, re
from time import sleep
import os.path
from pathlib import Path

#
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/abp_hindi.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/alt_hindi.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/boom_hindi.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/nc_hindi.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/vis_hindi.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/wq_hindi.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/boom_bangla.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Bangla.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Hindi.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Tamil.csv")
#data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/News18_India_Telgu.csv")
data = pd.read_csv("/DATA/vipin_2011mt22/aa/Dataset/Target/fc_tamil.csv")

data = data.iloc[:,[1,4]].values

t0 = time.time()
for i in range(len(data)):
    url = data[i][1]
    #url =url.split('?')[0]
    #print(url)
    name = data[i][0] + '.jpg'
    print(name)
    if Path('/DATA/vipin_2011mt22/aa/Dataset/TargetImages/' + name).exists():
        continue
    try:
        urllib.request.urlretrieve(url, '/DATA/vipin_2011mt22/aa/Dataset/TargetImages/' + name)
        print('Downloaded Image:',i)
    except:
        print('Could not Download Image:',i)
#    try:          
#        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}
#        
#        res = requests.get(url,headers=headers,timeout=7)
#        with open('/DATA/vipin_2011mt22/aa/Dataset/TargetImages/' + name, "abp") as f:
#            f.write(res.content)
#        print('Downloaded Image:',i)
#    except:
#        print('Could not Download Image:',i)
#        sleep(1)
    print(i, 'Time elapsed:',time.time()-t0,'sec')
print('Average time per query:',(time.time()-t0)/len(data),'seconds.')
