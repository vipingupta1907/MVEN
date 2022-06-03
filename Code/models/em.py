# -*- coding: utf-8 -*-

import os
gpu_number = "0" # Choose either 0 or 1
os.environ['CUDA_ENVIRONMENT_DEVICES'] = gpu_number

import os
import sys
import numpy as np
from numpy import asarray,zeros
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.metrics import classification_report
import transformers
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import timm
from losses import SupConLoss
from torchlars import LARS

if torch.cuda.is_available():
    device_name = "cuda:" + gpu_number
    device = torch.device(device_name)
else:
    device = torch.device("cpu")
# device = torch.device("cpu") # Force CPU
print("Using device", device)


base_model_name = 'rbert' 
#dataset_split = 'full' 
epochs = 100
lr = 3e-5 # Less LR

kf = KFold(n_splits=10, shuffle=True)
dataset_split = sys.argv[1]


model_name = 'cvalidate'

if base_model_name =='rbert':
    source_multimodal_data_full = np.load("../data/source_multimodal_out.npy")
    target_multimodal_data_full = np.load("../data/target_multimodal_out.npy")
    labels_data_full = np.load("../data/labels.npy").squeeze(1)
    contrastive_model_path = 'saved_models/contrast_head_rbert_lr_5_ep_1000.pt'
    model_name = 'em_rbert'
elif base_model_name == 'vbert':
    source_multimodal_data_full = np.load("../data/source_mm_vbert.npy")
    target_multimodal_data_full = np.load("../data/target_mm_vbert.npy")
    labels_data_full = np.load("../data/labels.npy").squeeze(1)
    contrastive_model_path = 'saved_models/contrast_head_vbert_lr_5_ep_1000.pt'
    model_name = 'em_vbert'

emotion_arr = np.load('../data/emotion_reprs_new.npy')


if dataset_split == 'full':
    source_multimodal_data = source_multimodal_data_full
    target_multimodal_data = target_multimodal_data_full
    labels_data = labels_data_full
    emotion_data = emotion_arr    
    model_name = model_name + '_full'    
elif dataset_split == 'h':    
    source_multimodal_data = source_multimodal_data_full[0:7163]
    target_multimodal_data = target_multimodal_data_full[0:7163]
    labels_data = labels_data_full[0:7163]
    emotion_data = emotion_arr[0:7163]    
    model_name = model_name + '_Hindi'    
elif dataset_split == 'b':   
    source_multimodal_data = source_multimodal_data_full[7163:8706]
    target_multimodal_data = target_multimodal_data_full[7163:8706]
    labels_data = labels_data_full[7163:8706]
    emotion_data = emotion_arr[7163:8706]    
    model_name = model_name + '_Bangla'  
elif dataset_split == 't':
    source_multimodal_data = source_multimodal_data_full[8706:10473]
    target_multimodal_data = target_multimodal_data_full[8706:10473]
    labels_data = labels_data_full[8706:10473]
    emotion_data = emotion_arr[8706:10473]
    model_name = model_name + '_Tamil'

#TODO: Printing the shape of the model
print('Model name', model_name)
print('Source shape', source_multimodal_data.shape)
print('Target shape', target_multimodal_data.shape)
print('Emotion Shape', emotion_data.shape)
print('Labels shape', labels_data.shape)

#TODO: Get Class weights
from collections import Counter 
weights = Counter(labels_data)
fake_weight = weights[1]/(weights[0]+weights[1])
real_weight = weights[0]/(weights[0]+weights[1])
class_weights = torch.tensor([fake_weight, real_weight], device=device)
print(class_weights)


# NOTE: ALL features
source_multimodal_tensor = torch.tensor(source_multimodal_data)
target_multimodal_tensor = torch.tensor(target_multimodal_data)
emotion_source_tensor = torch.tensor(emotion_data)
labels_tensor = torch.tensor(labels_data, dtype=torch.long)
print("Source shape", source_multimodal_tensor.shape)
print("Target shape", target_multimodal_tensor.shape)
print("Emotion Source shape", emotion_source_tensor.shape)
print("Labels shape", labels_tensor.shape)


#TODO: Add Pytorch DataLoader
def get_data_loader(batch_size, target_input, source_input, emotion_input, labels, split_type='train'):
	data = TensorDataset(target_input, source_input, emotion_input, labels)
	if split_type == 'train':
		sampler = RandomSampler(data)
	elif split_type == 'val':
		sampler = SequentialSampler(data)
	dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
	return data, sampler, dataloader

#TODO: Get K-Fold Split

kf.get_n_splits(source_multimodal_tensor)
print(kf)
# Obtaining split data
for train_index, test_index in kf.split(source_multimodal_tensor):
    print("TRAIN:", train_index.shape, "TEST:", test_index.shape)

# # NOTE: Old features
# train_tar, test_tar, train_src, test_src, train_emotion, test_emotion, train_labels, test_labels = train_test_split(target_multimodal_tensor, source_multimodal_tensor, emotion_source_tensor, labels_tensor, test_size=0.2, random_state=43)

# batch_size = 128
# train_data, train_sampler, train_dataloader = get_data_loader(batch_size, train_tar, train_src, train_emotion, train_labels, 'train')
# test_data, test_sampler, test_dataloader = get_data_loader(batch_size, test_tar, test_src, test_emotion, test_labels, 'val')

# Model imported from the previous network


# NOTE: Emotion based Model
class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x

class EmotionModule(nn.Module):
    def __init__(self):
        super(EmotionModule, self).__init__()
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
        return emotion_features

#TODO: Defin Combine Novelty-Emotion Model for classification
class FinalClassifier(nn.Module):
    def __init__(self, initial_dim):
        super(FinalClassifier, self).__init__()
        self.emotion_module = EmotionModule()
#        self.reduce_1 = nn.Linear(128, 512, bias=True)
#        self.tanh1 = nn.Tanh()
#        self.reduce_2 = nn.Linear(512, 128, bias=True)
#        self.tanh2 = nn.Tanh()
        self.logits_layer = nn.Linear(128, 2)
        

    def forward(self, target_input, source_input, emotion_input):
#        reduce_output_1 = self.tanh1(self.reduce_1(emotion_input.float())) # 512 dimension
#        reduce_output_2 = self.tanh2(self.reduce_2(reduce_output_1.float())) # 128 dimension
#        logits = self.logits_layer(reduce_output_2.float()) # 2 dimension
        logits = self.logits_layer(emotion_input.float()) # 2 dimension
        return logits

# Dry-Run Classification model
initial_dim = source_multimodal_tensor.shape[1]
# initial_dim = diff_arr.shape[1]
class_model = FinalClassifier(initial_dim).to(device) # For old model

# Optimizer and scheduler
def get_optimizer_scheduler(name, model, train_dataloader_len, epochs, lr_set):
	if name == "Adam":
		optimizer = AdamW(model.parameters(),
                  lr = lr_set, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
		)
	elif name == "LARS-SGD":
		base_optimizer = optim.SGD(model.parameters(), lr=lr_set, momentum=0.9)
		optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

	total_steps = train_dataloader_len * epochs

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, 
												num_warmup_steps = total_steps//2, # Default value in run_glue.py
												num_training_steps = total_steps)
	return optimizer, scheduler

# Getting the optimizer and scheduler

iters_to_accumulate = 2
name = "Adam"
# name = "LARS-SGD"
if base_model_name == 'vbert' and dataset_split == 'recov':
    criterion = nn.CrossEntropyLoss(class_weights)
else:
    criterion = nn.CrossEntropyLoss()
# optimizer, scheduler = get_optimizer_scheduler(name, class_model, len(train_dataloader), epochs, lr)

################ Evaluating Loss ######################
#######################################################
def evaluate_loss(net, device, criterion, dataloader):
    net.eval()
    mean_loss = 0
    count = 0
    with torch.no_grad():
        for it, (target_inputs, source_inputs, emotion_inputs, labels) in enumerate(tqdm(dataloader)):
            target_inputs, source_inputs, emotion_inputs, labels = target_inputs.to(device), source_inputs.to(device), emotion_inputs.to(device), labels.to(device)
            logits = net(target_inputs, source_inputs, emotion_inputs)
            mean_loss += criterion(logits.squeeze(-1), labels).item() # initially it was logits.squeeze(-1)
            count += 1
    return mean_loss / count

################ Flat Accuracy Calculation ####################
###############################################################

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
################ Validation Accuracy Calculation ####################
###############################################################

def evaluate_accuracy(model, device, validation_dataloader):
	model.eval()
	# Tracking variables 
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	# Evaluate data for one epoch
	for batch in validation_dataloader:
	    # Add batch to GPU
	    batch = tuple(t.to(device) for t in batch)	    
	    # Unpack the inputs from our dataloader
	    b_t_inputs, b_s_inputs, b_e_inputs, b_labels = batch	    
	    
	    # Telling the model not to compute or store gradients, saving memory and
	    # speeding up validation
	    with torch.no_grad(): 
	    	# Forward pass, calculate logit predictions.
	        # This will return the logits rather than the loss because we have
	        # not provided labels.
	    	logits = model(b_t_inputs, b_s_inputs, b_e_inputs)       

	    # Move logits and labels to CPU
	    logits = logits.detach().cpu().numpy()
	    label_ids = b_labels.to('cpu').numpy()
	    
	    # Calculate the accuracy for this batch of test sentences.
	    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
	    
	    # Accumulate the total accuracy.
	    eval_accuracy += tmp_eval_accuracy

	    # Track the number of batches
	    nb_eval_steps += 1
	accuracy = eval_accuracy/nb_eval_steps
	return accuracy

def train_model(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):
    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []
    # Iterating over all epochs
    for ep in range(epochs):
        net.train()
        running_loss = 0.0
        for it, (target_inputs, source_inputs, emotion_inputs, labels) in enumerate(tqdm(train_loader)):

            # Converting to cuda tensors
            target_inputs, source_inputs, emotion_inputs, labels = target_inputs.to(device), source_inputs.to(device), emotion_inputs.to(device), labels.to(device)
    		
            # Obtaining the logits from the model
            logits = net(target_inputs, source_inputs, emotion_inputs)
            # print(logits.device)

            # Computing loss
            # print(logits.squeeze(-1).shape)
            # print(labels.shape)
            loss = criterion(logits.squeeze(-1), labels)
            loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Calls backward()
            loss.backward()

            if (it + 1) % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                opti.step()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                net.zero_grad()


            running_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0


        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        val_accuracy = evaluate_accuracy(net, device, val_loader)
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))
        print("Epoch {} complete! Validation Accuracy : {}".format(ep+1, val_accuracy))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

#    # Saving the model
#    path_to_model='saved_models/{}_lr_{}_val_loss_{}_ep_{}.pt'.format(model_name, lr, round(best_loss, 5), best_ep)
#    torch.save(net_copy.state_dict(), path_to_model)
#    net.load_state_dict(torch.load(path_to_model)) # Re-Loading the best model
#    print("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()
    return net, model_name

def evaluate(prediction_dataloader, model, path_to_model, load = False):
	# Prediction on test set
	if load:
		print("Loading the weights of the model...")
		model.load_state_dict(torch.load(path_to_model))

	print('Evaluating on the testset')

	# Put model in evaluation mode
	model.eval()

	# Tracking variables 
	predictions , true_labels = [], []

	# Predict 
	for batch in prediction_dataloader:
	  # Add batch to GPU
	  batch = tuple(t.to(device) for t in batch)
	  
	  # Unpack the inputs from our dataloader
	  b_t_inputs, b_s_inputs, b_e_inputs, b_labels = batch
	  
	  # Telling the model not to compute or store gradients, saving memory and 
	  # speeding up prediction
	  with torch.no_grad():
	      # Forward pass, calculate logit predictions
	      logits = model(b_t_inputs, b_s_inputs, b_e_inputs)

	  # Move logits and labels to CPU
	  logits = logits.detach().cpu().numpy()
	  label_ids = b_labels.to('cpu').numpy()

	  pred_flat = np.argmax(logits, axis=1).flatten()
	  labels_flat = label_ids.flatten()
	  
	  # Store predictions and true labels
	  predictions.extend(pred_flat)
	  true_labels.extend(labels_flat)
	return predictions, true_labels

# TODO: Implement K-Fold splits model
batch_size = 128
ctr = 0
all_predictions = []
all_true_labels = []
for train_index, test_index in kf.split(source_multimodal_tensor):
    print("SPLIT", ctr)
    ctr += 1
    print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
    train_tar, test_tar = target_multimodal_tensor[train_index], target_multimodal_tensor[test_index]
    train_src, test_src = source_multimodal_tensor[train_index], source_multimodal_tensor[test_index]
    train_emotion, test_emotion = emotion_source_tensor[train_index], emotion_source_tensor[test_index]
    train_labels, test_labels = labels_tensor[train_index], labels_tensor[test_index]
    train_data, train_sampler, train_dataloader = get_data_loader(batch_size, train_tar, train_src, train_emotion, train_labels, 'train')
    test_data, test_sampler, test_dataloader = get_data_loader(batch_size, test_tar, test_src, test_emotion, test_labels, 'val')
    optimizer, scheduler = get_optimizer_scheduler(name, class_model, len(train_dataloader), epochs, lr)
    model, model_name = train_model(class_model, criterion, optimizer, lr, scheduler, train_dataloader, test_dataloader, epochs, iters_to_accumulate)
    predictions, true_labels = evaluate(test_dataloader, model, path_to_model = '', load = False)
    all_predictions.extend(predictions)
    all_true_labels.extend(true_labels)

# Printing the final results
print(model_name, 'Multi-Modal Classification accuracy is')
print(metrics.accuracy_score(y_true = all_true_labels, y_pred = all_predictions)*100)
print(classification_report(y_true = all_true_labels, y_pred = all_predictions, target_names = ['fake', 'real'], digits=4))
# Converting to csv
# Removed transpose - check if actually required
clsf_report = pd.DataFrame(classification_report(y_true = all_true_labels, y_pred = all_predictions, output_dict=True, target_names = ['fake', 'real'], digits=4))
clsf_report.to_csv(str('saved_models/'+model_name+'.csv'), index= True)
