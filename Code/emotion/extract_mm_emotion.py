
# coding: utf-8

# In[1]:


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
from tqdm import tqdm
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


# In[2]:


if torch.cuda.is_available():
    device = torch.device("cuda")
    os.environ['CUDA_ENVIRONMENT_DEVICES'] = "0"
else:
    device = torch.device("cpu")
print("Using device", device)


# In[3]:


# Loading original data
img_data = np.load("../data/image_array.npy")
labels_data = np.load("../data/labels.npy")


# In[4]:


num_images, sources, width, height, num_channels = img_data.shape
img_data_reshape = np.reshape(img_data, newshape=(num_images, sources, num_channels, width, height))
img_data_target = torch.tensor(img_data_reshape[:,0,:,:,:])
img_data_source = torch.tensor(img_data_reshape[:,1,:,:,:]) 
print('New Target Shape', img_data_target.shape)
print('New Source Shape', img_data_source.shape)


# In[5]:


#TODO: Add Pytorch DataLoader
def get_data_loader(batch_size, data, labels, split_type='train'):
	data = TensorDataset(data, labels)
	if split_type == 'train':
		sampler = RandomSampler(data)
		dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
	elif split_type == 'val':
		sampler = SequentialSampler(data)
		dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
	return data, sampler, dataloader


# In[6]:


batch_size = 128
test_data, test_sampler, test_dataloader = get_data_loader(batch_size, img_data_target, torch.tensor(labels_data, dtype=torch.long), 'val')


# In[7]:


# Import the main model
#TODO: Define Resent-50 model
class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x

class ResnetBased(nn.Module):
    def __init__(self):
        super(ResnetBased, self).__init__()
        self.vision_base_model = timm.create_model('resnet50', pretrained=True)
        self.vision_model_head = ResNetBottom(self.vision_base_model)
        self.project_1 = nn.Linear(2048, 1024, bias=True)
        self.project_2 = nn.Linear(1024, 512, bias=True)
        self.project_3 = nn.Linear(512, 128, bias=True)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()
        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()
        self.drop3 = nn.Dropout()
        self.classification = nn.Linear(128, 2, bias=True)
    def forward(self, img_features):
        with torch.no_grad():
            img_out = self.vision_model_head(img_features)
        emotion_features = self.tanh3(self.project_3(self.tanh2(self.project_2(self.tanh1(self.project_1(img_out))))))
        class_out = self.classification(emotion_features)
        return emotion_features, class_out


# In[8]:


# Get the model
emo_model = ResnetBased().to(device)
# emo_model.load_state_dict(torch.load('saved_models/emo_combine_res50_lr_3e-05_val_loss_0.59487_ep_61.pt')) # CE model
# emo_model.load_state_dict(torch.load('saved_models/emo_combine_res50_lr_3e-05_val_loss_0.65715_ep_53.pt')) # Weighted CE model
emo_model.load_state_dict(torch.load('saved_models/emo_combine_res50_lr_3e-05_val_loss_0.64149_ep_43.pt')) # Combine data


# In[9]:


#TODO: Load 100d pre-trained Glove embeddings
# Loading the pre-trained Glove embeddings
embeddings_dict = {}
with open("/sda/rina_1921cs13/Word_Embedding/glove/glove.6B.200d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


# In[10]:


#TODO: Construction scaffold labels
emotion_real = embeddings_dict['sadness']+embeddings_dict['joy']+embeddings_dict['love']
emotion_fake = embeddings_dict['fear']+embeddings_dict['surprise']+embeddings_dict['anger']


# In[11]:


#TODO: Extract all emotion features
all_emotion_features = []
all_classes = []
for i in range(0, len(img_data_target), 32):
    print(i)
    start_range = i
    if start_range+32 >= len(img_data_target):
        end_range = len(img_data_target)
    else:
        end_range = start_range + 32
    emotion_features, class_out = emo_model(img_data_target[start_range:end_range, :, :, :].to(device))
    class_out = np.argmax(class_out.cpu().detach().numpy(), axis=1)
    all_emotion_features.extend(emotion_features.cpu().detach().numpy())
    all_classes.extend(class_out)
all_emotion_features = np.array(all_emotion_features)
print(all_emotion_features.shape)


# In[22]:


# Add bias to emotion features
new_em_features = []
for i, class_out in enumerate(all_classes):
    if class_out == 1 and labels_data[i] == 1: # and labels_data[i] == 1 # -> scaffolding
        intermediate_features = np.concatenate((all_emotion_features[i], emotion_real))
    elif class_out == 0:
        intermediate_features = np.concatenate((all_emotion_features[i], emotion_fake))
    else:
        intermediate_features = np.concatenate((all_emotion_features[i], np.zeros(shape=(200))))
    new_em_features.append(intermediate_features)
new_em_arr = np.array(new_em_features)
print('Bias emotion array shape', new_em_arr.shape)


# In[23]:


from sklearn.decomposition import PCA
pca = PCA(n_components=128)
new_em_arr_reduce = pca.fit_transform(new_em_arr)
print('New emotion array shape', new_em_arr_reduce.shape)


# In[24]:


# Split to train and test
# NOTE: Splitting data into train and test
train_data, test_data, train_labels, test_labels = train_test_split(new_em_arr_reduce, labels_data, test_size=0.2, random_state=43)


# In[25]:


# Reshape Labels
new_train_labels = np.reshape(train_labels, newshape=(train_labels.shape[0]))
print(new_train_labels.shape)
new_test_labels = np.reshape(test_labels, newshape=(test_labels.shape[0]))
print(new_test_labels.shape)


# In[26]:


# TODO: Fit a logistic regression model
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(max_iter=500)
logisticRegr.fit(train_data, new_train_labels)
predictions = logisticRegr.predict(test_data)


# In[28]:


print('Combined Emotion Classification accuracy is')
print(metrics.accuracy_score(test_labels, predictions)*100)
print(classification_report(test_labels, predictions, target_names = ['fake', 'real']))


# In[29]:


# Saving emotion representations
#np.save('../data/emotion_reprs_new', new_em_arr_reduce)

