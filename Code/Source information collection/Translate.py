#!pip install googletrans==3.1.0a0
from googletrans import Translator
translator = Translator()


import numpy as np
import pandas as pd
df=pd.read_excel("/content/nc_hindi_gris.xlsx")
df.head()

text1=[]
text2=[]
text3=[]
text4=[]
for index, row in df.iterrows():
    text1.append(str(row.text1))
    text2.append(str(row.text2))
    text3.append(str(row.text3))
    text4.append(str(row.text4))

tran_text1=[]
for i in text1:
    try:
        tran_text1.append(translator.translate(i, dest='hi' ).text)
    except:
        tran_text1.append("??")

tran_text2=[]
for i in text2:
    try:
        tran_text2.append(translator.translate(i, dest='hi' ).text)
    except:
        tran_text2.append("??")
    
tran_text3=[]
for i in text3:
    try:
        tran_text3.append(translator.translate(i, dest='hi' ).text)
    except:
        tran_text3.append("??")
    
tran_text4=[]
for i in text4:
    try:
        tran_text4.append(translator.translate(i, dest='hi' ).text)
    except:
        tran_text4.append("??")

df1 =df
df1['tran_text1'] = tran_text1
df1['tran_text2'] = tran_text2
df1['tran_text3'] = tran_text3
df1['tran_text4'] = tran_text4
df1.to_excel("/content/fc_hindi_both.xlsx")

df2=df
df2['text1'] = tran_text1
df2['text2'] = tran_text2
df2['text3'] = tran_text3
df2['text4'] = tran_text4
df2.to_excel("/content/fc_hindi_tran_text.xlsx")
