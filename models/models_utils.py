import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        # Generate predictions 
        out = self(images)
        # Calculate loss
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        # Generate predictions 
        out = self(images)
        # Calculate loss
        loss = F.cross_entropy(out, labels)
        # Calculate accuracy
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_step_siamese(self, batch):
        img0, img1 , label = batch
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        out = self(img0,img1)
        
        # Calculate loss
        criterion = torch.nn.BCEWithLogitsLoss(size_average=True)
        loss = criterion(out,label)

        ############# Calculate accuracy #############
        # Find indices where False (means images from the same class)
        # They must have acc=1 with minimum distance, otherwise acc=0 
        acc_list = []
        # acc_list_false = []
        for idx,sample in enumerate(label):
            if int( torch_to_scalar(sample) ) == 0:
                dist = torch_to_scalar(out[idx])
                acc_list.append( 1/np.exp(-dist) )
        # acc_false = np.mean(acc_list_false)
        # Find indices where True (means images from the different class)
        # They must have acc=0 with minimum distance, otherwise acc=1 
        # acc_list_true = []
        for idx,sample in enumerate(label):
            if int( torch_to_scalar(sample) ) == 0:
                dist = torch_to_scalar(out[idx])
                acc_list.append( 1 - np.exp(-dist) )
        # acc_true = np.mean(acc_list_true)

        # Total acc is sum of two
        # acc = acc_false + acc_true
        acc = np.mean(acc_list)
        # return {'val_loss': loss.detach(), 'val_acc': acc}
        return loss.detach(),acc

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # Combine losses
        epoch_loss = torch.stack(batch_losses).mean() 
        batch_accs = [x['val_acc'] for x in outputs]
        # Combine accuracies
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))

def accuracy_siamese(outputs, labels):
    """0 means : same classes"""
    return torch.tensor(torch.sum(outputs != labels).item()/len(outputs))

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

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

@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)

@torch.no_grad()
def evaluate_siamese(model, test_loader):
    model.eval()
    loss_list = []
    acc_list = []
    print("Test started ")
    for epoch, batch in enumerate( tqdm(test_loader) ):
        loss,acc = model.validation_step_siamese(batch) 
        loss_list.append(loss.tolist())
        acc_list.append(acc)
    print("Test ended")
    val_loss = np.mean(loss_list)
    print("Val loss calculated : ",val_loss)
    val_acc = np.mean(acc_list)
    print("Val acc calculated : ",val_acc)
    return val_loss, val_acc

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def torch_to_scalar(tensor):
    return tensor.tolist()[0]