import os
import sys
import argparse
import logging
import math
import numpy as np
import torch
import torch.utils.data as data
#from networks.DDAM import DDAMNet
from datetime import datetime
from networks.DDAM_ABAW import DDAMNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.utils import Sequence
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import to_categorical
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

FINE_TUNE_LOSS = False
EXTRA_EXPR_TRAIN = False
EMMA_ANNOTATIONS = True
EMMA_EPOCHS = 10

BATCH_SIZE = 64 #32
EPOCHS = 11 #10
EXTRA_EPOCHS = 5

# Function to freeze all layers
def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True

# Function to freeze specific layers
def freeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False

# Function to unfreeze specific layers
def unfreeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True

def freeze_batchnorm_layers(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.eval()

# Define the test_transforms outside the class
IMG_SIZE = 112

test_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

train_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=1, scale=(0.05, 0.05)),
        ])

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, main_folder, mode, batch_size=32, image_size=(260, 260), n_classes_3=8, shuffle=True, device='cpu', transforms=test_transforms, drop_last=False):
        'Initialization'
        self.main_folder = main_folder
        self.mode = mode
        self.batch_size = batch_size
        self.image_size = image_size
        self.n_classes_3 = n_classes_3
        
        self.shuffle = shuffle
        self.device = device
        self.transforms = transforms
        self.drop_last = drop_last

        if self.mode == 'train':
            if EMMA_ANNOTATIONS:
                self.targets_csv = f"./annotations/final_emma_dataset_{EMMA_EPOCHS}_epochs.csv"
            else:
                self.targets_csv = './annotations/transformed_training_set_annotations_3.csv'
        elif self.mode == 'val':
            self.targets_csv = './annotations/transformed_validation_set_annotations.csv'
        else:
            raise ValueError("Invalid mode. Mode must be 'train' or 'val'.")

        # Load the CSV to get the total number of samples
        self.targets_df = pd.read_csv(self.targets_csv)
        self.list_IDs = self.targets_df.index.tolist()

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.drop_last:
            # Exclude the last batch if its size is less than batch_size
            return len(self.list_IDs) // self.batch_size
        else:
            # Include the last batch even if it's smaller than batch_size
            return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        if end_idx > len(self.list_IDs) and self.drop_last:
            raise IndexError("Index out of range for batch generation with drop_last=True.")

        indexes = self.indexes[start_idx:end_idx]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, 3, *self.image_size), dtype=np.float32)
        y = [[] for _ in range(3)]  # Assuming 3 target arrays (valence/arousal, emotions, actions)

        for i, ID in enumerate(list_IDs_temp):
            row = self.targets_df.iloc[ID]
            image_path = os.path.join(self.main_folder, row['image'])
            image = Image.open(image_path).convert('RGB')
            # Apply transformations
            image = self.transforms(image)
            X[i,] = image.numpy()

            y[0].append([row[1], row[2]])  # First target value (valence/arousal)
            target_3_one_hot = to_categorical(row[3], num_classes=self.n_classes_3)
            y[1].append(target_3_one_hot)
            
            y[2].append([row[col_start] for col_start in range(4, len(row))])
            


        X_tensor = torch.tensor(X).to(self.device)
        y_tensor = [torch.tensor(np.array(sublist)).to(self.device) for sublist in y]  # Convert each sublist to tensor

    
        
        return X_tensor, y_tensor
    
def CCC(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)
    covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    y_true_var = torch.var(y_true)
    y_pred_var = torch.var(y_pred)
    ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean)**2 + 1e-8)
    return ccc

def CCC_loss(y_true, y_pred):
    loss = 1-0.5*(CCC(y_true[:,0], y_pred[:,0])+CCC(y_true[:,1], y_pred[:,1]))
    return loss

def f1_metric(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        Positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
        recall = TP / (Positives + 1e-7)  # Adding a small epsilon for numerical stability
        return recall 
    
    def precision_m(y_true, y_pred):
        TP = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        Pred_Positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + 1e-7)  # Adding a small epsilon for numerical stability
        return precision 
    
    # Initialize lists to store precision, recall, and f1 scores per class
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    # Iterate over each class
    for class_idx in range(y_true.shape[1]):
        precision = precision_m(y_true[:, class_idx], y_pred[:, class_idx])
        recall = recall_m(y_true[:, class_idx], y_pred[:, class_idx])
        
        # Calculate F1 score
        f1 = 2 * ((precision * recall) / (precision + recall + 1e-7))
        
        # Append scores to lists
        precision_per_class.append(precision.item())
        recall_per_class.append(recall.item())
        f1_per_class.append(f1.item())
    
    # Overall macro F1 score
    macro_f1 = torch.mean(torch.tensor(f1_per_class))
    
    # Return overall F1 score and F1 score per class
    return macro_f1.item(), f1_per_class

def f1_metric_o(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        Positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
        
        recall = TP / (Positives + 1e-7)  # Adding a small epsilon for numerical stability
        return recall 
    
    def precision_m(y_true, y_pred):
        TP = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        Pred_Positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives + 1e-7)  # Adding a small epsilon for numerical stability
        return precision 
    
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-7))  # Adding a small epsilon for numerical stability
    
    return f1, f1

def f1_score_actions(y_true, y_pred, threshold=0.5):
    # Convert predicted probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).float()
    
    # Initialize lists to store F1 scores for each AU
    F1s = []
    
    # Calculate F1 score for each AU
    for i in range(y_true.shape[1]):
        # Extract true and predicted values for the current AU
        y_true_au = y_true[:, i]
        y_pred_au = y_pred_binary[:, i]
        
        # Calculate True Positives, False Positives, False Negatives
        TP = torch.sum(y_true_au * y_pred_au)
        FP = torch.sum((1 - y_true_au) * y_pred_au)
        FN = torch.sum(y_true_au * (1 - y_pred_au))
        
        # Calculate Precision, Recall, and F1 Score
        precision = TP / (TP + FP + 1e-7)  # Adding epsilon to avoid division by zero
        recall = TP / (TP + FN + 1e-7)  # Adding epsilon to avoid division by zero
        f1 = 2 * precision * recall / (precision + recall + 1e-7)  # Adding epsilon to avoid division by zero
        
        F1s.append(f1.item())
    
    F1s = torch.tensor(F1s)
    F1_mean = torch.mean(F1s)
    
    return F1s, F1_mean.item()


def compute_AU_F1(pred,label):
    pred = np.array(pred)
    label = np.array(label)
    AU_targets = [[] for i in range(12)]
    AU_preds = [[] for i in range(12)]
    F1s = []
    for i in range(pred.shape[0]):
        for j in range(12):
            p = pred[i,j]
            if p>=0.5:
                AU_preds[j].append(1)
            else:
                AU_preds[j].append(0)
            AU_targets[j].append(label[i,j])
    
    for i in range(12):
        F1s.append(f1_score(AU_targets[i], AU_preds[i]))

    F1s = np.array(F1s)
    F1_mean = np.mean(F1s)
    return F1s, F1_mean

eps = sys.float_info.epsilon

class AttentionLoss(nn.Module):
    def __init__(self, ):
        super(AttentionLoss, self).__init__()
    
    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head-1):
                for j in range(i+1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt+1
                    loss = loss+mse
            loss = cnt/(loss + eps)
        else:
            loss = 0
        return loss
    

def compute_EXP_F1(pred, target):
    pred_labels = []
    pred = np.array(pred)
    target = np.array(target)
    
    # Convert one-hot encoded target to class labels
    if len(target.shape) > 1 and target.shape[1] > 1:
        target = np.argmax(target, axis=1)
    
    # Convert predictions to class labels
    for i in range(pred.shape[0]):
        l = np.argmax(pred[i])
        pred_labels.append(l)
        
    # Compute F1 scores
    F1s = f1_score(target, pred_labels, average=None)
    macro_f1 = np.mean(F1s)
    return F1s, macro_f1


def compute_AU_F1(pred,label):
    pred = np.array(pred)
    label = np.array(label)
    AU_targets = [[] for i in range(12)]
    AU_preds = [[] for i in range(12)]
    F1s = []
    for i in range(pred.shape[0]):
        for j in range(12):
            p = pred[i,j]
            if p>=0.5:
                AU_preds[j].append(1)
            else:
                AU_preds[j].append(0)
            AU_targets[j].append(label[i,j])
    
    for i in range(12):
        F1s.append(f1_score(AU_targets[i], AU_preds[i]))

    F1s = np.array(F1s)
    F1_mean = np.mean(F1s)
    return F1s, F1_mean

def adjust_weights(va_loss, expr_loss, au_loss):
    # Normalize losses and adjust weights (example)
    total_loss = va_loss + expr_loss + au_loss
    return va_loss / total_loss, expr_loss / total_loss, au_loss / total_loss

# Define the train and evaluation functions
def train_model(model, train_loader, optimizer, criterion_val_arousal=None, criterion_emotions=None, criterion_actions=None, criterion_at=None, device=None, challenges=('val_arousal', 'emotions', 'actions'), weights={'val_arousal': 1.0, 'emotions': 1.0, 'actions': 1.0}):
    model.train()
    running_loss = 0.0

    if FINE_TUNE_LOSS:
        task_losses = {'val_arousal': 0.0, 'emotions': 0.0, 'actions': 0.0}

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels_val_arousal = labels[0].to(device) if 'val_arousal' in challenges else None
        labels_emotions = labels[1].to(device) if 'emotions' in challenges else None
        labels_actions = labels[2].to(device) if 'actions' in challenges else None

        optimizer.zero_grad()
        outputs = model(inputs)
        val_arousal = outputs[0] if 'val_arousal' in challenges else None
        emotions = outputs[1] if 'emotions' in challenges else None
        actions = outputs[2] if 'actions' in challenges else None
        heads = outputs[-1] if criterion_at else None

        loss = 0.0
        if 'val_arousal' in challenges:
            loss_val_arousal = criterion_val_arousal(val_arousal, labels_val_arousal)

            if FINE_TUNE_LOSS:
                task_losses['val_arousal'] += loss_val_arousal.item()
                loss += weights['val_arousal'] * loss_val_arousal
            else:
                loss += loss_val_arousal

        if 'emotions' in challenges:
            emotions_argmax = torch.argmax(labels_emotions, dim=1)
            loss_emotions = criterion_emotions(emotions, emotions_argmax)

            if FINE_TUNE_LOSS:
                task_losses['emotions'] += loss_emotions.item()
                loss += weights['emotions'] * loss_emotions
            else:
                loss += loss_emotions

        if 'actions' in challenges:
            loss_actions = criterion_actions(actions.float(), labels_actions.float())
            
            if FINE_TUNE_LOSS:
                task_losses['actions'] += loss_actions.item()
                loss += weights['actions'] * loss_actions
            else:
                loss += loss_actions
        if criterion_at:
            loss += 0.1 * criterion_at(heads)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if FINE_TUNE_LOSS:
        # Update the weights to fine-tune the loss
        total_loss = sum(task_losses.values())
        if total_loss > 0:
            for task in task_losses:
                weights[task] = task_losses[task] / total_loss

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def evaluate_model(model, test_loader, criterion_val_arousal=None, criterion_emotions=None, criterion_actions=None, criterion_at=None, device=None, challenges=('val_arousal', 'emotions', 'actions')):
    model.eval()
    val_arousal_preds, emotions_preds, actions_preds = [], [], []
    val_arousal_labels, emotions_labels, actions_labels = [], [], []
    running_val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels_val_arousal = labels[0].to(device) if 'val_arousal' in challenges else None
            labels_emotions = labels[1].to(device) if 'emotions' in challenges else None
            labels_actions = labels[2].to(device) if 'actions' in challenges else None

            outputs = model(inputs)
            val_arousal = outputs[0] if 'val_arousal' in challenges else None
            emotions = outputs[1] if 'emotions' in challenges else None
            actions = outputs[2] if 'actions' in challenges else None
            heads = outputs[-1] if criterion_at else None
            
            loss = 0.0
            if 'val_arousal' in challenges:
                loss_val_arousal = criterion_val_arousal(val_arousal, labels_val_arousal)
                loss += loss_val_arousal
                val_arousal_preds.append(val_arousal.cpu())
                val_arousal_labels.append(labels_val_arousal.cpu())
            if 'emotions' in challenges:
                emotions_argmax = torch.argmax(labels_emotions, dim=1)
                loss_emotions = criterion_emotions(emotions, emotions_argmax)
                loss += loss_emotions
                emotions_preds.append(emotions.cpu())
                emotions_labels.append(labels_emotions.cpu())
            if 'actions' in challenges:
                loss_actions = criterion_actions(actions.float(), labels_actions.float())
                loss += loss_actions
                actions_preds.append(actions.cpu())
                actions_labels.append(labels_actions.cpu())
            if criterion_at:
                loss += 0.1 * criterion_at(heads)

            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(test_loader)

    ccc_val_arousal = ccc_valence = ccc_arousal = None
    f1_emotions = f1_emotion_mean = None
    f1_actions = f1_mean = None

    if 'val_arousal' in challenges:
        val_arousal_preds = torch.cat(val_arousal_preds)
        val_arousal_labels = torch.cat(val_arousal_labels)
        ccc_valence = CCC(val_arousal_labels[:, 0].cpu(), val_arousal_preds[:, 0].cpu()).item()
        ccc_arousal = CCC(val_arousal_labels[:, 1].cpu(), val_arousal_preds[:, 1].cpu()).item()
        ccc_val_arousal = (ccc_valence + ccc_arousal) / 2
    if 'emotions' in challenges:
        emotions_preds = torch.cat(emotions_preds)
        emotions_labels = torch.cat(emotions_labels)
        f1_emotions, f1_emotion_mean = compute_EXP_F1(emotions_preds, emotions_labels)
    if 'actions' in challenges:
        actions_preds = torch.cat(actions_preds)
        actions_labels = torch.cat(actions_labels)
        f1_actions, f1_mean = f1_score_actions(actions_labels, actions_preds, threshold=0.5)

    return ccc_val_arousal, ccc_valence, ccc_arousal, f1_emotions, f1_actions, avg_val_loss, f1_mean, f1_emotion_mean, actions_preds, actions_labels


#######################################################################################################################################
print('--------------------------------------------------------------------------------------')
print('------- train.py execution start ', datetime.now())

challenges=('val_arousal', 'emotions', 'actions')
model_path = './checkpoints_ver2.0/affecnet8_epoch25_acc0.6469.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataloaders (placeholders, replace with actual DataLoader instances)
# Initialize the generator
train_loader = DataGenerator("../ABAW_7th/cropped_aligned", mode="train", batch_size=BATCH_SIZE, image_size=(112, 112), shuffle=True, device='cuda', transforms=train_transforms, drop_last = True)
test_loader = DataGenerator("../ABAW_7th/cropped_aligned",mode="val",batch_size=BATCH_SIZE, image_size=(112,112), shuffle=False, device='cuda', transforms=test_transforms, drop_last = True)

# Define loss functions
criterion_val_arousal = CCC_loss
criterion_emotions = nn.CrossEntropyLoss()
criterion_actions = nn.BCELoss()
criterion_at = AttentionLoss()

# Multitask Model Training
model = DDAMNet(num_class=8, num_head=2, pretrained=False, train_val_arousal=True, train_emotions=True, train_actions=True)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
freeze_all_layers(model)
layers_to_unfreeze = ['custom_classifier', "Linear","cat_head","features"]
unfreeze_layers(model, layers_to_unfreeze)
#freeze_batchnorm_layers(model)
model.to(device)

if EMMA_ANNOTATIONS:
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0)
    warm_up_with_cosine_lr = lambda \
            epoch: epoch / 5 if epoch <= 5 else 0.5 * (math.cos(
        (epoch - 5) / (6 - 5) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
else:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)

best_P_Score = float('-inf')
best_model_state = None

# Weights for loss fine-tuning (ignored unless FINE_TUNE_LOSS is True)
weights = { 'val_arousal': 1.0, 'emotions': 1.0, 'actions': 1.0 }

print('------- Init of training ', datetime.now())

for epoch in range(EPOCHS):
    train_loss = train_model(model, train_loader, optimizer, criterion_val_arousal, criterion_emotions, criterion_actions, criterion_at, device, challenges=('val_arousal', 'emotions', 'actions'), weights=weights)
    results = evaluate_model(model, test_loader, criterion_val_arousal, criterion_emotions, criterion_actions, criterion_at, device, challenges=('val_arousal', 'emotions', 'actions'))

    P_score = results[0] + (results[7] or 0) + (results[6] or 0)
    val_loss = results[5]

    if P_score > best_P_Score:
        best_P_Score = P_score
        best_model_state = model.state_dict()

    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")
    print(f"P_SCORE: {results[0] + (results[7] or 0) + (results[6] or 0)}")
    if 'val_arousal' in challenges:
        print(f"Validation CCC (Valence-Arousal): {results[0]}, Valence: {results[1]}, Arousal: {results[2]}")
    if 'emotions' in challenges:
        print(f"F1 Score_ABAW (Emotions): {results[7]}, F1 Score (Emotions per class): {results[3]}")
    if 'actions' in challenges:
        print(f"F1 Score Mean (Actions): {results[6]}, F1 Score (Actions): {results[4]}")

    scheduler.step()

    torch.save(best_model_state, 'best_multitask_model_att.pth')

print('------- Training over: ', datetime.now())
print('------- best_P_Score: ', best_P_Score)

pred_action_units = results[8]
true_action_units = results[9]
f1_val_arousal = results[0]

if EXTRA_EXPR_TRAIN:
    print('-------Now training EXPR a little bit more')

    freeze_all_layers(model)
    layers_to_unfreeze = ["custom_classifier.emotions"]
    unfreeze_layers(model, layers_to_unfreeze)

    for epoch in range(EXTRA_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, None, criterion_emotions, None, criterion_at, device, challenges=('emotions'), weights=weights)
        results = evaluate_model(model, test_loader, None, criterion_emotions, None, criterion_at, device, challenges=('emotions'))

        print(f"Extra EXPR epoch {epoch + 1}/{EXTRA_EPOCHS}, Training Loss: {train_loss}")
        print(f"Validation Loss: {val_loss}")
        print(f"P_SCORE: {results[7] or 0}")
        if 'emotions' in challenges:
            print(f"F1 Score_ABAW (Emotions): {results[7]}, F1 Score (Emotions per class): {results[3]}")

        if EMMA_ANNOTATIONS:
            scheduler.step()

        torch.save(best_model_state, 'best_multitask_model_att.pth')

    print('------- Extra training over: ', datetime.now())
    print('------- best_P_Score: ', best_P_Score)

# Define the range of thresholds to search
thresholds = np.arange(0.1, 0.9, 0.01)

# Initialize the best thresholds and corresponding best F1 scores
best_thresholds = np.zeros(pred_action_units.shape[1])
best_f1_scores = np.zeros(pred_action_units.shape[1])

# Iterate over each index
for i in range(pred_action_units.shape[1]):
    best_f1 = 0
    best_thresh = 0
    for threshold in thresholds:
        # Apply threshold
        pred_binary = (pred_action_units[:, i] >= threshold).int()
        
        # Calculate F1 score
        f1 = f1_score(true_action_units[:, i], pred_binary, average="macro")
        
        # Check if this is the best F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = threshold
    
    # Store the best threshold and F1 score
    best_thresholds[i] = best_thresh
    best_f1_scores[i] = best_f1

# Output the best thresholds for each index
print("Best Thresholds:", best_thresholds)
print("Best F1 Scores:", best_f1_scores)

# Apply the best thresholds to get the final binary predictions
pred_action_units_binary_optimal = np.zeros(pred_action_units.shape)
for i in range(pred_action_units.shape[1]):
    pred_action_units_binary_optimal[:, i] = (pred_action_units[:, i] >= best_thresholds[i]).int()

# Calculate the final macro-average F1 score with the best thresholds
final_f1_action_units = f1_score(true_action_units, pred_action_units_binary_optimal, average="macro")
print("Final Macro-Average F1 Score:", final_f1_action_units)

performance_measure = f1_val_arousal + results[7] + final_f1_action_units

print(f"Performance Measure (P): {performance_measure}")

print('--------------------------------------------------------------------------------------')
