import os
from posix import ST_NOEXEC
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid
from dataset_makers.dataset_siamese import DatasetMakerInference, class_decode
from models.siamese_v3 import SiameseNetwork, ContrastiveLoss
from models.models_utils import to_device, get_default_device, DeviceDataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

def imshow(images,labels,euclidean_distance,img0_labels,img1_labels):
    npimg = images.numpy()
    # plt.axis("off")
    plt.figure(figsize=(10,10))
    # plt.figure()

    for idx,l in enumerate(labels.tolist()):
        l_int = int(l[0])
        if l_int == 0:
            plt.text(idx*130, 400, 'SAME',fontweight='bold')
        else:
            plt.text(idx*130, 400, 'DIFF',fontweight='bold')

    for idx,l in enumerate(euclidean_distance.tolist()):
        euc_text = "{0:3f}".format(l)
        plt.text(idx*135, 500, euc_text,fontweight='bold')
    plt.text(8*135-20, 500, "Euclidean_Dist",fontweight='bold',color='r')

    for idx,l in enumerate(img0_labels.tolist()):
        class_text = class_decode(l)
        plt.text(idx*130, 600, class_text,fontweight='bold')
    plt.text(8*130-20, 600, "Class",fontweight='bold',color='b')

    plt.tight_layout()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imshow_one_vs_all(images,euclidean_distance,img0_labels,img1_labels,mean_distances):

    fig = plt.figure()
    npimg = images.numpy()
    npimg_final = np.transpose(npimg, (1, 2, 0)) 

    ############ IMAGES ###########
    ax1 = fig.add_subplot(3,1,1)
    ax1.imshow(npimg_final)
    
    ############ TABLE - 1 ###########
    ax2 = fig.add_subplot(3,1,2)
    data=[]
    data_img0 = []
    for idx,l in enumerate(img0_labels.tolist()):
        data_img0.append(class_decode(l))
    data.append(data_img0)
    data_euc = []
    for idx,l in enumerate(euclidean_distance.tolist()):
        euc_text = "{0:3f}".format(l)
        data_euc.append(euc_text)
    data.append(data_euc)
    column_labels=[]
    for idx,l in enumerate(img1_labels.tolist()):
        column_labels.append( class_decode(l) )
    row_labels = ['OriginalClass','EuclideanDistances']
    ax2.axis('tight')
    ax2.axis('off')
    ax2.table(cellText=data,colLabels=column_labels,rowLabels=row_labels)

    ############ TABLE - 1 ###########
    ax3 = fig.add_subplot(3,1,3)
    data=[ ]
    column_labels=[]
    for idx,key in enumerate(mean_distances):
        column_labels.append( key )
    val_data = []
    for idx,key in enumerate(mean_distances):
        val_data.append(mean_distances[key])
    data.append(val_data)

    row_labels = ['MeanDist']
    ax3.axis('tight')
    ax3.axis('off')
    ax3.table(cellText=data,colLabels=column_labels,rowLabels=row_labels)

    plt.show()


device = get_default_device()
# dataset = DatasetMaker(batch_size=1,batch_size_test=8)
dataset = DatasetMakerInference(batch_size=1,batch_size_test=8)
_, test_loader = dataset.create_datasets()
test_loader = DeviceDataLoader(test_loader, device)
training_model = to_device(SiameseNetwork(3, 4),device)


################### Test ###################
net = SiameseNetwork(3,3).cuda()
best_path = "/home/anton/coding/msc/DeepLearningHW/CoronaHack_Chest/Logs/networks/Siamese_v3_2021_2_1_11_20_46/network_191.pth"
net.load_state_dict(torch.load(best_path))
criterion = ContrastiveLoss()

def compare_different_class(test_loader,num_iter=10):
    dataiter = iter(test_loader)
    for i in range(num_iter):
        x0,x1,labels,img0_labels,img1_labels = next(dataiter)
        concatenated = torch.cat((x0,x1),0)
        output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        concatenated = concatenated.cpu()
        labels = labels.cpu()
        imshow(make_grid(concatenated),
                labels,
                euclidean_distance,
                img0_labels,img1_labels)

def compare_same_class(test_loader,num_iter=10):
    """Compare batch of same class images with batch of other class images"""
    dataiter = iter(test_loader)
    x0,_,_,img0_labels,_ = next(dataiter)
    for i in range(num_iter):
        _,x1,_,_,img1_labels = next(dataiter)
        concatenated = torch.cat((x0,x1),0)
        output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        concatenated = concatenated.cpu()
        imshow_one_vs_all(
            make_grid(concatenated),
            euclidean_distance,
            img0_labels,img1_labels)
# compare_same_class(test_loader)

def get_image_by_class(test_loader,class_name='Virus'):
    dataiter = iter(test_loader)
    x0 = None
    img0_labels = None
    img0_class = None
    while True:
        x0,_,_,img0_labels,_ = next(dataiter)
        img0_class = class_decode(img0_labels)
        if img0_class == class_name:
            break
    # There is batch number of images of same class
    # but we want to compare with only one of them
    # so take first and copy others with that
    x0_new = torch.zeros_like(x0)
    for x in range(x0.shape[0]):
        x0_new[x] = x0[0]
    return x0_new,img0_labels,img0_class

def prediction(euclidean_distance,img1_labels):
    preds_covid = []
    preds_unknown = []
    preds_bacteria = []
    euclidean_distance = euclidean_distance.tolist()
    for i in range(euclidean_distance.__len__()):
        img_class = class_decode(img1_labels[i])
        if img_class == 'Virus':
            preds_covid.append(euclidean_distance[i])
        elif img_class == 'unknown':
            preds_unknown.append(euclidean_distance[i])
        elif img_class == 'bacteria':
            preds_bacteria.append(euclidean_distance[i])
    covid_dis_mean = np.mean(preds_covid)
    unknown_dis_mean = np.mean(preds_unknown)
    bacteria_dis_mean = np.mean(preds_bacteria)
    mean_distances = {
        'covid' : covid_dis_mean,
        'unknown' : unknown_dis_mean,
        'bacteria' : bacteria_dis_mean
    }
    return mean_distances

def compare_same_class_single(test_loader, num_iter=10):
    # Get a covid image from test dataset
    covid_img,img0_labels,img0_class = get_image_by_class(test_loader,'Virus')
    dataiter = iter(test_loader)
    for x in range(num_iter):
        _,x1,_,_,img1_labels = next(dataiter)
        concatenated = torch.cat((covid_img,x1),0)
        output1,output2 = net(Variable(covid_img).cuda(),Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        concatenated = concatenated.cpu()
        mean_distances_dict = prediction(euclidean_distance,img1_labels)
        imshow_one_vs_all(
            make_grid(concatenated),
            euclidean_distance,
            img0_labels,img1_labels,mean_distances_dict)            


for x in range(50):
    compare_same_class_single(test_loader,num_iter=2)