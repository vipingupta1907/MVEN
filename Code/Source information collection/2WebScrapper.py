import requests
from bs4 import BeautifulSoup
import urllib.parse
import pandas as pd
import numpy as np
import json
import time
from time import sleep


# # Importing Source Reliability Data
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


# Adding some domains in dict that has to be removed from search result
rem = ['translate', 'pinterest', 'shutterstock', 'linkedin', 'merriam-webster', 'amazon', 'unsplash', 'facebook', 
       'myntra', 'dictionary', 'youtube', 'flipkart', 'developer', 'twitter', 'webcache', 'reddit', 'britannica']

for dom in rem:
    dict[dom] = 'FALSE'

def reliability_from_link(link):
    url = str(link)
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
from random import randint

# # Importing Dataset
data = pd.read_excel("/DATA/vipin_2011mt22/aa/News18_India_Tamil.xlsx")
data.head()
#main_img_url = data.iloc[:,6].values
main_img_url = data.iloc[:,1].values
print(main_img_url)
# # Extraction

def url_encoding(img_url):
    return urllib.parse.quote(img_url)

def is_valid_link(link):
    link = str(link)
    if len(link)<13:
        return False
    elif link[0:4]!='http':
        return False
    elif reliability_from_link(link)=='FALSE':
        return False
    else:
        return True

def comp(val):
    return val['rel']

def get_filtered_links(links):
    res = []
    for link in links:
        if is_valid_link(link):
            res.append(link)
            
    temp = []
    for link in res:
        
        rel = reliability_from_link(link)
        if rel == 'UNKNOWN':
            rel = 3
        elif rel == 'MIXED':
            rel = 2
        else:
            rel = 1
        temp.append({
            'url': link,
            'rel':rel
        })
    temp.sort(key=comp)
    
    for i in range(len(res)):
        res[i] = temp[i]['url']
    
    return res

def get_complete_image_url(host_url,image_link):
    pref=''
    dom=''
    if host_url[0:5] == 'http:':
        pref = 'http://'
        for i in range(7,len(host_url)):
            if(host_url[i]=='/'):
                dom = host_url[7:i]
                break
    else:
        pref = 'https://'
        for i in range(8,len(host_url)):
            if(host_url[i]=='/'):
                dom = host_url[8:i]
                break
    
    if len(image_link)<2:
        return ''
    elif image_link[0:2]=='//':
        return pref + image_link[2:]
    elif image_link[0]=='/':
        return pref + dom + image_link
    elif len(image_link)<13:
        return ''
    elif image_link[0:4]!='http':
        return ''
    else:
        return image_link

def get_info(img_url):
#    if len(str(img_url))<13 or img_url[0:4]!='http':
#        return 0, [['NA','NA',[]],['NA','NA',[]],['NA','NA',[]],['NA','NA',[]]]
#    if img_url[:22] == 'https://t4.rbxcdn.com/':
#        return 0, [['NA','NA',[]],['NA','NA',[]],['NA','NA',[]],['NA','NA',[]]]
    
    
    headers = {'Host': 'www.google.com',
               "Upgrade-Insecure-Requests": "1",
               "DNT": "1",
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0',
               'Accept': '*/*',
               'Accept-Language': 'en-US,en;q=0.5',
               'Accept-Encoding': 'gzip, deflate, br',
               'Referer': 'https://www.google.com/',
               'Origin': 'https://www.google.com',
               'Connection': 'keep-alive','Content-Length': '0',
               'TE': 'Trailers'}

    url = 'https://www.google.com/searchbyimage?&image_url=' +img_url
    url = 'https://www.google.com/searchbyimage?&image_url=' + url_encoding(img_url)
    response = ''

    try:
        response = requests.get(url,headers=headers,timeout=20)
    except:
        sleep(3)
        return 0, [[url,'Not Responding',[]],[url,'Not Responding',[]],[url,'Not Responding',[]],[url,'Not Responding',[]]]
        
    sleep(randint(3,7))
    print("======================================================================================")
    print(response.history)
    print(response.url)
    print("======================================================================================")
    
    
#    img_url = url_encoding(img_url)
#    google_search_url = 'https://www.google.com/searchbyimage?&image_url=' + img_url
#    url = google_search_url
#    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}
#    response = ''
#    try:
#        response = requests.get(url,headers=headers,timeout=20)
#    except:
#        sleep(3)
#        return 0, [[url,'Not Responding',[]],[url,'Not Responding',[]],[url,'Not Responding',[]],[url,'Not Responding',[]]]

    content = BeautifulSoup(response.content, 'html.parser')
    link_containers = content.select('#search a')
    links =[]
    for link_container in link_containers:
        link = ''
        try:
            link = link_container['href']
        except:
            continue
        links.append(link)
    print(links)
        
    links = get_filtered_links(links)
    print(links)
    res = []
    num = 0
    for i in range(len(links)):           
        url = links[i]
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}
        response = ''
        try:
            response = requests.get(url,headers=headers,timeout=20)
        except:
            sleep(3)
            continue

        content = BeautifulSoup(response.content, 'html.parser')
        
        # Extracting text from all paragraphs on current source page
        texts = content.findAll('p')

        extracted_text = '';
        for text in texts:
            extracted_text += text.get_text() + ' '
        
        val = extracted_text
        val = val.split()
        if len(val)<30:
            continue
        
        # Extracting url of images present on source pages
        src_img_url = []
        
        source_images = content.findAll('img')
        
        for source_image in source_images:
            link = ''
            try:
                link = source_image['src']
            except:
                continue
            
            comp_image_url = get_complete_image_url(url,link)
            
            if(comp_image_url!=''):
                src_img_url.append(comp_image_url)
        
        if len(src_img_url)==0:
            continue
        
        # Storing Results for current source
        res.append([url, extracted_text,src_img_url])
        num += 1
        
        if num == 4:
            break
    
    for i in range(num,4):
        res.append(['NA','NA',[]])
    
    return num , res

extracted_info = []
numSources = []
img_url = []

def run(fir,las):
    t0 = time.time()
    for i in range(fir,las):
        num , temp_info = get_info(main_img_url[i])
        numSources.append(num)
        extracted_info.append(temp_info)
        img_url.append(main_img_url[i])
        print(i, 'Time elapsed:',time.time()-t0,'sec')
    print('Average time per query:',(time.time()-t0)/(las-fir),'seconds.')

print(len(main_img_url))

#run(16000,20015)
run(0,1080)
#run(1000,2000)
#run(2000,3000)
#run(3000,4000)
#run(4000,5000)
#run(5000,6000)
#run(6000,6248)
#
#m = 'https://previews.123rf.com/images/visualsvixen/visualsvixen0801/visualsvixen080100106/2317064-illustration-de-raisin-de-fleurs-et-de-lierre-en-forme-de-c-ur-un-%C3%A9l%C3%A9ment-de-design-dossier-ne-conti.jpg'
#num , temp_info = get_info(m)
#numSources.append(num)
#extracted_info.append(temp_info)
#img_url.append(m)



source1 = []
text1 = []
image_url1 = []
source2 = []
text2 = []
image_url2 = []
source3 = []
text3 = []
image_url3 = []
source4 = []
text4 = []
image_url4 = []

for i in range (len(extracted_info)):
    source1.append(extracted_info[i][0][0])
    text1.append(extracted_info[i][0][1])
    image_url1.append(extracted_info[i][0][2])
    source2.append(extracted_info[i][1][0])
    text2.append(extracted_info[i][1][1])
    image_url2.append(extracted_info[i][1][2])
    source3.append(extracted_info[i][2][0])
    text3.append(extracted_info[i][2][1])
    image_url3.append(extracted_info[i][2][2])
    source4.append(extracted_info[i][3][0])
    text4.append(extracted_info[i][3][1])
    image_url4.append(extracted_info[i][3][2])

dictionary = {}

dictionary['numSources'] = numSources
dictionary['img_url'] = img_url

dictionary['source1'] = source1
dictionary['text1'] = text1
dictionary['image_url1'] = image_url1
dictionary['source2'] = source2
dictionary['text2'] = text2
dictionary['image_url2'] = image_url2
dictionary['source3'] = source3
dictionary['text3'] = text3
dictionary['image_url3'] = image_url3
dictionary['source4'] = source4
dictionary['text4'] = text4
dictionary['image_url4'] = image_url4

import numpy as np
np.save('my_file.npy', dictionary) 

# Load
#read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()

df = pd.DataFrame(dictionary)
df = df.applymap(lambda x: str(x).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))
#df.to_excel("/DATA/vipin_2011mt22/aa/boom_bangla_gris.xlsx")
import xlsxwriter
df.to_excel("News18_India_Tamil_gris.xlsx", engine='xlsxwriter')




