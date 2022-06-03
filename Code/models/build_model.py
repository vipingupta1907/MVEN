import os
import sys
import numpy as np
from numpy import asarray,zeros
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import transformers
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import timm





import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import re
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import vgg16
from tensorflow.keras import models, Model
from numpy import save
from numpy import load
#from deep_translator import GoogleTranslator
#import langid
import nltk


# In[2]:


nltk.download('punkt')


# In[3]:


# Utility Functions
def get_image(imagepath):
	img = Image.open(imagepath).convert('RGB')
	img = img.resize((224,224))
	img = img_to_array(img)
	if img.shape[2]==1:
		img = np.stack([img,img,img],axis=2)
		img = img.reshape(img.shape[0],img.shape[1],3)
	return img

def cleaned_text(x):
	x = str(x)
#  	val =  [re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() ]  # if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words
	val =  [w for w in x.split() ]
	res = ""
	for i in range(min(300,len(val))):
		res = res + val[i] + ' '
	res = res.split()
	res = ' '.join(res)
	return res

def clean_para(text):
	sent_text = nltk.sent_tokenize(text)
	cleaned_sentences = []
	for i in range(min(len(sent_text), 15)):
		cleaned_sentences.append(cleaned_text(sent_text[i]))
	cleaned_para = '. '.join(cleaned_sentences)
	return cleaned_para


# In[4]:


# get data
def make_pairs():
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairTexts = []
    pairLabels = []
    ID = []
    print('[INFO] Loading and Processing Dataset...')
    source = []

    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/abp_hindi.csv")
    source.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/News18_India_Hindi.csv")
    source.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/alt_hindi.csv")
    source.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/boom_hindi.csv")
    source.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/nc_hindi.csv")
    source.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/vis_hindi.csv")
    source.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/wq_hindi.csv")
    source.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/News18_India_Bangla.csv")
    source.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/boom_bangla.csv")
    source.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/News18_India_Tamil.csv")
    source.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Source/fc_tamil.csv")
    source.append(temp)

    source = pd.concat(source, ignore_index=True, sort=False)
    target = []

    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/abp_hindi.csv")
    target.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/News18_India_Hindi.csv")
    target.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/alt_hindi.csv")
    target.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/boom_hindi.csv")
    target.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/nc_hindi.csv")
    target.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/vis_hindi.csv")
    target.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/wq_hindi.csv")
    target.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/News18_India_Bangla.csv")
    target.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/boom_bangla.csv")
    target.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/News18_India_Tamil.csv")
    target.append(temp)
    temp = pd.read_csv("/sda/rina_1921cs13/vipin/Final_Data/Target/fc_tamil.csv")
    target.append(temp)

    target = pd.concat(target, ignore_index=True, sort=False)
    print('Source columns', source.columns)
    print('Target columns', target.columns)
    source = source.iloc[:,:].values
    target = target.iloc[:,:].values
    return source, target


# In[5]:


source, target = make_pairs()


# In[6]:


source


# In[7]:


print('Source type', type(source))
print('Source shape', source.shape)
print('Target type', type(target))
print('Target Shape', target.shape)


    
# In[13]:


# get data
def get_data(source, target):
    pairImages = []
    pairTexts = []
    pairLabels = []
    ID = []
    identity = []
    ctr = 0
    for i in range(len(source)):
        if i%100==0:
            print('Loading Data...',i)
        target_img = ''

        
        try:
            target_img = get_image("/sda/rina_1921cs13/vipin/TargetImages/" + source[i][1] + '.jpg')
        except:
            target_img = get_image("/sda/rina_1921cs13/vipin/TargetImages/abp0.jpg")
#             continue
        target_txt = cleaned_text(target[i][3])
        dataset = ''
        if source[i][1][0:3]=='alt':
            dataset = 'alt_hindi'
        elif source[i][1][0:3]=='abp':
            dataset = 'abp'
        elif source[i][1][0:18]=='News18_India_Hindi':
            dataset = 'news18_hindi'
        elif source[i][1][0:10]=='boom_hindi':
            dataset = 'boom_hindi'
        elif source[i][1][0:8]=='nc_hindi':
            dataset = 'nc'
        elif source[i][1][0:9]=='vis_hindi':
            dataset = 'vis'
        elif source[i][1][0:8]=='wq_hindi':
            dataset = 'wq'
        elif source[i][1][0:19]=='News18_India_Bangla':
            dataset = 'news18_bangla'
        elif source[i][1][0:11]=='boom_bangla':
            dataset = 'boom_bangla'
        elif source[i][1][0:18]=='News18_India_Tamil':
            dataset = 'news18_tamil'
        elif source[i][1][0:8]=='fc_tamil':
            dataset = 'fc_tamil'
        else:
            print("other")

            
        for j in range(3,3+4*source[i][2],4):  
        
            try:
                source[i][j+2] = literal_eval(source[i][j+2])
            except:
                pass
            try:
                src_txt = clean_para(source[i][j+1])
            except:
              
                src_txt = source[i][j+1]
            try:
                 src_img = get_image('/sda/rina_1921cs13/vipin/SourceImages/' + dataset + '/' + source[i][j+2]['image_name'])
                pairImages.append([target_img, src_img])
                pairTexts.append([target_txt, src_txt])	
                ID.append([source[i][j+2]['image_name']])
         
                identity.append(ctr)			
                if target[i][5]=='FAKE':
                    pairLabels.append([0])
                else:
                    pairLabels.append([1])
            except:
                continue
        ctr +=1 
    # return a 2-tuple of our pairs and labels
    return (np.array(pairImages), np.array(pairTexts), np.array(pairLabels), np.array(ID), np.expand_dims(np.array(identity), 1))
   # return ( np.array(pairTexts), np.array(pairLabels), np.array(ID), np.expand_dims(np.array(identity), 1))
  #  return np.array(pairImages) , np.array(pairTexts)



img_data , txt_data, label, ID, identity = get_data(source, target)
#txt_pair, label, ID, identity = get_data(source, target)
#img_data , txt_data = get_data(source, target)

np.save("/sda/rina_1921cs13/vipin/vipin/data/labels.npy", label)
np.save("/sda/rina_1921cs13/vipin/vipin/data/ids.npy", ID)
np.save("/sda/rina_1921cs13/vipin/vipin/data/identity.npy",identity)


# Data analysis
print(img_data.shape)








####################################################################################################

device = torch.device("cpu") # Force CPU
print("Using device", device)

# Load the data
#img_data = np.load("../data/image_array.npy")
#txt_data = np.load("../data/text_array.npy")
labels_data = np.load("../data/labels.npy")
ids_data = np.load("../data/ids.npy")
# Printing the shapes
print(img_data.shape)
print(txt_data.shape)
print(labels_data.shape)
print(ids_data.shape)

# Reshape image to -> num_images, sources, num_channels, width, heigth
#NOTE: Can convert image data to tensor only in training loop with very less batch size
num_images,  width, height, num_channels = img_data.shape
img_data_reshape = np.reshape(img_data, newshape=(num_images,  num_channels, width, height))
img_data_target = torch.tensor(img_data_reshape) # Don't convert to GPU
img_data_source = torch.tensor(img_data_reshape) # Don't convert to GPU
print('New Target Shape', img_data_target.shape)
print('New Source Shape', img_data_source.shape)

# Utility Models


# Vision Model
class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x

# Vision Model
class ViTBottom(nn.Module):
    def __init__(self, original_model):
        super(ViTBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x


# Text Model
class BERTModel(nn.Module):
    def __init__(self, bert_model="bert-base-multilingual-uncased", freeze_bert=False):
        super(BERTModel, self).__init__()
        self.model_name = bert_model
        #  Instantiating BERT-based model object
        self.config = AutoConfig.from_pretrained(bert_model, output_hidden_states=False)
        self.bert_layer = AutoModel.from_pretrained(bert_model, config = self.config)
        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''
        hidden_state  = self.bert_layer(input_ids, attn_masks, token_type_ids)
        pooler_output = hidden_state[0][:,0]

        return pooler_output        
        
        
# Text Tokenizer
def get_transformer_model(modelname):
    trans_tokenizer = AutoTokenizer.from_pretrained(modelname, do_lower_case = True)
    print(trans_tokenizer)
    return trans_tokenizer

################ Tokenizer ####################
###############################################
def tokenize(model_name, data_list, tokenizer, MAX_LEN):
	print('Tokenizing')
	# add special tokens for BERT to work properly
	if model_name == 'bert-base-multilingual-uncased':
  #if model_name == 'bert-base-uncased':
		sentences = ["[CLS] " + data_list[i] + " [SEP]" for i in range(0,len(data_list))]
	elif model_name == 'roberta-base':
		sentences = ["<s> " + data_list[i] + " </s>" for i in range(0,len(data_list))]
	tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
	# print ("Tokenize the first sentence:")
	# print (tokenized_texts[0])
	# Pad our input tokens
	input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
	                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
	# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
	input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
	# Create attention masks
	attention_masks = []
	# Create a mask of 1s for each token followed by 0s for padding
	for seq in input_ids:
	  seq_mask = [float(i>0) for i in seq]
	  attention_masks.append(seq_mask)

	# Finally convert this into torch tensors
	data_inputs = torch.tensor(input_ids)
	data_masks = torch.tensor(attention_masks)
	return data_inputs, data_masks
 
 
# Prepare text data
trans_model_name = 'bert-base-multilingual-uncased' #'bert-base-uncased' # for BERT
# trans_model_name = 'roberta-base' # for RoBERTa
trans_tokenizer = get_transformer_model(trans_model_name)
MAX_LEN = 100
source_text_inputs, source_text_masks = tokenize(trans_model_name, txt_data[:,1], trans_tokenizer, MAX_LEN) # Data on CPU Need to convert to GPU
target_text_inputs, target_text_masks = tokenize(trans_model_name, txt_data[:,0], trans_tokenizer, MAX_LEN) # Data on CPU Need to convert to GPU


# Need to save these tokenized arrays
np.save("/sda/rina_1921cs13/vipin/vipin/data/tokenized/source_text_mbert.npy", source_text_inputs.detach().cpu().numpy())
np.save("/sda/rina_1921cs13/vipin/vipin/data/tokenized/source_mask_mbert.npy", source_text_masks.detach().cpu().numpy())
np.save("/sda/rina_1921cs13/vipin/vipin/data/tokenized/target_text_mbert.npy", target_text_inputs.detach().cpu().numpy())
np.save("/sda/rina_1921cs13/vipin/vipin/data/tokenized/target_mask_mbert.npy", target_text_masks.detach().cpu().numpy())



        


#TODO: Multimodal (Image+Text) model
class MultimodalHeadResnet(nn.Module):
    def __init__(self):
        super(MultimodalHeadResnet, self).__init__()
        self.vision_base_model = timm.create_model('resnet18', pretrained=True)
        self.vision_model_head = ResNetBottom(self.vision_base_model)
        #self.text_head = BERTModel('bert-base-uncased')
        self.text_head = BERTModel('bert-base-multilingual-uncased')
    def forward(self, img_features, txt_features):
        with torch.no_grad():
            img_out = self.vision_model_head(img_features)
            txt_out = self.text_head(txt_features[0], txt_features[1], token_type_ids=None)
            multimodal_concat = F.normalize(torch.cat((img_out, txt_out), 1), dim=1)
        return multimodal_concat
        
        
#TODO: Multimodal (Image+Text) model
class MultimodalHeadVit(nn.Module):
    def __init__(self, text_trans_name):
        super(MultimodalHeadVit, self).__init__()
        self.pretrained_v = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vision_model_head = ViTBottom(self.pretrained_v)
        self.text_head = BERTModel(text_trans_name)
    def forward(self, img_features, txt_features):
        with torch.no_grad():
            img_out = self.vision_model_head(img_features) # not working - dimension 3
            maxpool = nn.MaxPool2d((img_out.shape[1], 1))
            img_pooled_out = maxpool(img_out).squeeze(1) # 768 dimension
            # print(img_pooled_out.shape)
            txt_out = self.text_head(txt_features[0], txt_features[1], token_type_ids=None)
            multimodal_concat = F.normalize(torch.cat((img_pooled_out, txt_out), 1), dim=1)
        return multimodal_concat
        
        
# Create Multimodal model object # choice 'vbert' or 'rbert'

multimodal_model = MultimodalHeadResnet().to(device) # For Resnet model
#multimodal_model = MultimodalHeadVit(trans_model_name)

#TODO: Multimodal forward pass on the entire dataset (USE CPU)
for i in range(0, len(img_data_source), 32):
    print(i)
    start_range = i
    end_range = i + 32
    if i+32>= len(img_data_source):
        end_range = len(img_data_source)
    source_image_input = img_data_source[start_range:end_range,:,:,:].to(device)
    target_image_input = img_data_target[start_range:end_range,:,:,:].to(device)
    source_text_input, source_text_mask = source_text_inputs[start_range:end_range,:].to(device), source_text_masks[start_range:end_range,:].to(device)
    target_text_input, target_text_mask = target_text_inputs[start_range:end_range,:].to(device), target_text_masks[start_range:end_range,:].to(device)
    target_multimodal_out = multimodal_model(target_image_input, (target_text_input, target_text_mask))
    source_multimodal_out = multimodal_model(source_image_input, (source_text_input, source_text_mask))
    if i==0:
        source_all_out = source_multimodal_out
        target_all_out = target_multimodal_out
    else:
        source_all_out = torch.cat((source_all_out, source_multimodal_out), 0)
        target_all_out = torch.cat((target_all_out, target_multimodal_out), 0)
    print(source_all_out.shape)
    print(target_all_out.shape)
    
print(source_all_out.shape)
print(target_all_out.shape)

# Saving multimodal features
source_all_out = source_all_out.detach().numpy()
target_all_out = target_all_out.detach().numpy()
#np.save('/DATA/vipin_2011mt22/aa/Dataset/source_multimodal_out_vit_roberta.npy', source_all_out)
#np.save('/DATA/vipin_2011mt22/aa/Dataset/target_multimodal_out_vit_roberta.npy', target_all_out)

np.save('/sda/rina_1921cs13/vipin/vipin/data/source_multimodal_out.npy', source_all_out)
np.save('/sda/rina_1921cs13/vipin/vipin/data/target_multimodal_out.npy', target_all_out)






