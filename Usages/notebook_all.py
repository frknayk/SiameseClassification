import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate
# from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Loading and Inspecting the Data
data_path = 'data/Covid19_Kaggle_CNN/'
img_path = data_path + 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset'
train_img_dir = img_path + '/train'
test_img_dir = img_path + '/test'
img_dir = os.listdir(img_path)
df_meta = pd.read_csv(data_path+'Chest_xray_Corona_Metadata.csv')
df_meta_summary = pd.read_csv(data_path+'Chest_xray_Corona_dataset_Summary.csv')

# Replace null data points to 'unknown'
df_meta.fillna('unknown', inplace=True)

train_data = df_meta[df_meta['Dataset_type']=='TRAIN']
test_data = df_meta[df_meta['Dataset_type']=='TEST']

# Define output classes
train_data.loc[train_data['Label'].eq('Normal'), 'class'] = 'healthy'
train_data.loc[(train_data['class'].ne('healthy') & train_data['Label_1_Virus_category'].eq('bacteria')), 'class'] = 'bacteria'
train_data.loc[(train_data['class'].ne('healthy') & train_data['class'].ne('bacteria') & train_data['Label_2_Virus_category'].eq('COVID-19')), 'class'] = 'COVID-19'
train_data.loc[(train_data['class'].ne('healthy') & train_data['class'].ne('bacteria') & train_data['class'].ne('COVID-19')), 'class'] = 'other'

target_dict = {'healthy' : 0,
               'bacteria' : 1,
               'COVID-19' : 2,
               'other' : 3}
train_data['target'] = train_data['class'].map(target_dict)

# Display X-Ray Images
def plot_images(path,class_str,numdisplay):
    fig, ax = plt.subplots(numdisplay,2, figsize=(15,2.5*numdisplay))
    for row,file in enumerate(path):
        image = plt.imread(file)
#         print(image.shape)
        ax[row,0].imshow(image, cmap=plt.cm.bone)
        ax[row,1].hist(image.ravel(), 256, [0,256])
        ax[row,0].axis('off')
        if row == 0:
            ax[row,0].set_title('Images')
            ax[row,1].set_title('Histograms')
    fig.suptitle('Class='+class_str,size=16)
    plt.show()

def display_class_images(img_path,dataset,train_or_test_str,classlabel,numdisplay):
    path = dataset[dataset['class']==classlabel]['X_ray_image_name'].values
    sample_path = path[:numdisplay]
    img_dir = img_path+"/"+train_or_test_str
    sample_path = list(map(lambda x: os.path.join(img_dir,x), sample_path))
    plot_images(sample_path,classlabel,numdisplay)
# display_class_images(img_path,train_data,"train","healthy",4)

class CustomDataSet(Dataset):
    def __init__(self, main_dir,meta_data, transform):
        self.main_dir = main_dir
        self.meta_data = meta_data
        self.transform = transform
        self.total_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        meta_data = self.meta_data.iloc[idx] 
        filename = meta_data['X_ray_image_name']
        try:
            file_idx = self.total_imgs.index(filename)
        except:
            print("Data not found!")
            return None        
        img_loc = os.path.join(self.main_dir, self.total_imgs[file_idx])
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((128,128))
        tensor_image = self.transform(image)
        tensor_label = torch.tensor(meta_data['target'].item())
        return tensor_image, tensor_label

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = filter (lambda x:x is not None, batch)
    return default_collate(list(batch))

batch_size=32

# std_dev and mean of images per channel
stats = ((0.0093, 0.0093, 0.0092),(0.4827, 0.4828, 0.4828)) 

train_tfms = tt.Compose([tt.RandomCrop(128, padding=8, padding_mode='edge'), tt.ToTensor(), tt.Normalize(*stats, inplace = True)])
test_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats, inplace = True)])

train_ds, test_ds = train_test_split(train_data, test_size=0.25,random_state= 1, shuffle = True)
train_ds, test_ds = train_ds.reset_index(drop=True), test_ds.reset_index(drop=True)

# 
train_dataset = CustomDataSet(train_img_dir, train_ds, transform=train_tfms)    
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,                            
                          num_workers=0, pin_memory=True, collate_fn=my_collate)
test_dataset = CustomDataSet(train_img_dir, test_ds, transform=test_tfms)
test_loader = DataLoader(test_dataset , batch_size=2*batch_size, shuffle=False, 
                         num_workers=0, pin_memory=True, collate_fn=my_collate)

def show_batch(dl):
    for images,labels in dl:
        print(images.shape, labels.shape)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=8).permute(1,2,0))
        break
# show_batch(train_loader)
# plt.show()


import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(16), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Setting up GPU device
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device() # cuda
train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)

@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# More sophisticated fit function, with the following features:
# Learning rate scheduling, weight decay, gradient clipping

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

model = to_device(ResNet9(3, 4), device)

# Train model
num_epochs = 1
opt_func = torch.optim.Adam
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
history = []
#history = fit(num_epochs, lr, model, train_loader, test_loader, opt_func)
history += fit_one_cycle(num_epochs, max_lr, model, train_loader, test_loader, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')

@torch.no_grad()
def get_all_preds_and_targets(model, loader):
    all_preds = torch.tensor([])
    all_targets = torch.tensor([])
    for batch in loader:
        images, labels = batch

        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        all_preds = torch.cat((all_preds, preds),dim=0)
        all_targets = torch.cat((all_targets, labels),dim=0)
    return all_preds, all_targets

plot_accuracies(history)
plot_losses(history)
plot_lrs(history)
plt.show()

device = torch.device('cpu')
test_loader = DeviceDataLoader(test_loader, device)
model = to_device(model, device)

with torch.no_grad():
    predictions, targets = get_all_preds_and_targets(model, test_loader)

sns.countplot(predictions.numpy())
plt.show()

#let's print a classification report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(targets, predictions))
con_mat = confusion_matrix(targets, predictions)
con_mat = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
plt.figure(figsize = (10,10))
plt.title('CONFUSION MATRIX')
sns.heatmap(con_mat, cmap='coolwarm',
            yticklabels=['Healthy', 'Bacteria','COVID-19','other virus'],
            xticklabels=['Healthy', 'Bacteria','COVID-19','other virus'],
            annot=True)