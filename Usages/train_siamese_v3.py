
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from dataset_makers.dataset_siamese import DatasetMaker
from models.siamese_v3 import SiameseNetwork, ContrastiveLoss
from models.models_utils import to_device, get_default_device, DeviceDataLoader
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter

TRAN_FROM_CHECKPOINT = True

grad_clip = True

device = get_default_device()
dataset = DatasetMaker(batch_size=64,batch_size_test=8)
train_loader, _ = dataset.create_datasets()
train_loader = DeviceDataLoader(train_loader, device)
training_model = to_device(SiameseNetwork(3, 4),device)

################### PLOT ###################
import matplotlib.pyplot as plt
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


STARTING_INDEX = None
current_date = None
model_name = "Siamese_v3"
writer = None

if not TRAN_FROM_CHECKPOINT:
    ################### Create log folders ###################
    from runner import get_datetime, create_training_folder
    current_date = get_datetime()    
    if len(model_name) > 0:
        current_date = model_name + '_' +  current_date
    create_training_folder("Logs/networks/"+current_date)
    writer = SummaryWriter(log_dir='Logs/runs/'+current_date)
    ################### Starting index of training ###################
    STARTING_INDEX = 0

################### TRAIN ###################
net = SiameseNetwork(3,3).cuda()

if TRAN_FROM_CHECKPOINT:
    STARTING_INDEX = 164
    current_date = "/home/anton/coding/msc/DeepLearningHW/CoronaHack_Chest/Logs/networks/Siamese_v3_2021_2_1_11_20_46/"
    best_path = current_date + '/'+"network_" + str(STARTING_INDEX) + ".pth"
    net.load_state_dict(torch.load(best_path))
    print("Trained network is loaded : {}".format(best_path))
    writer = SummaryWriter(log_dir='Logs/runs/Siamese_v3_2021_2_1_11_20_46_cont')

criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
counter = []
loss_history = [] 
iteration_number= 0
for epoch in range(STARTING_INDEX+1,STARTING_INDEX+200):
    start_time = time.time()
    epoch_loss_list = []
    for i, data in enumerate( tqdm(train_loader),0 ):
        img0, img1 , label = data
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        # Gradient clipping
        if grad_clip: 
            nn.utils.clip_grad_value_(net.parameters(), grad_clip)
        optimizer.step()
        epoch_loss_list.append(loss_contrastive.item())
    end_time = time.time()
    dt = end_time - start_time
    epoch_mean_loss = np.mean(epoch_loss_list)
    writer.add_scalar("loss", epoch_mean_loss, epoch)
    loss_history.append(epoch_mean_loss)
    print("Epoch number {}\n Current loss {}\n Took {} seconds\n".format(epoch,epoch_mean_loss,dt))

    # Save network
    if not TRAN_FROM_CHECKPOINT:
        path = "Logs/networks/"+current_date+"/network_"+str(epoch)+'.pth'
        torch.save(net.state_dict(), path)
    else:
        path = current_date+"/network_"+str(epoch)+'.pth'
        torch.save(net.state_dict(), path)
