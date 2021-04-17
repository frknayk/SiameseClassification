from models.siamese_v3 import SiameseNetwork
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

stats_global = ((0.0093, 0.0093, 0.0092),(0.4827, 0.4828, 0.4828))

target_dict = {'healthy' : 0,
               'bacteria' : 1,
               'COVID-19' : 2,
               'other' : 3}

def class_encode(img_label):
    class_tensor = torch.zeros((1))
    if img_label == 'unknown' :
        class_tensor[0] = 0
    elif img_label == 'bacteria' :
        class_tensor[0] = 1
    elif img_label == 'Virus' :
        class_tensor[0] = 2
    else:
        class_tensor[0] = -1
    return class_tensor

def class_decode(class_tensor):
    img_label = 'None'
    class_tensor_scalar = int(class_tensor[0])
    if class_tensor_scalar == 0 :
        img_label = 'unknown'
    elif class_tensor_scalar == 1 :
        img_label = 'bacteria'
    elif class_tensor_scalar == 2 :
        img_label = 'Virus'
    else:
        img_label = 'None'
    return img_label

class SiameseNetworkDataset(Dataset):
    def __init__(self,main_dir,meta_data, transform):
        self.main_dir = main_dir
        self.meta_data = meta_data
        self.total_imgs = os.listdir(main_dir)
        self.transform = transform
        self.latest_img0_label = None
        self.latest_img1_label = None
        
    def __getitem__(self,index):
        # Random sample
        img0_label,img0_idx = self.get_random_img()

        if img0_idx is None:
            return None
        img0_dir = self.get_img_loc(img0_idx)

        # Init img1
        img1_label = ""
        img1_dir = ""

        # We need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_label,img1_idx = self.get_random_img()
                if img1_idx is None:
                    continue
                if img0_label == img1_label:
                    img1_dir = self.get_img_loc(img1_idx)
                    break
        else:
            while True:
                #keep looping till a different class image is found
                img1_label, img1_idx = self.get_random_img()
                if img1_idx is None:
                    continue
                if img1_label != img0_label:
                    img1_dir = self.get_img_loc(img1_idx)
                    break

        self.latest_img0_label = img0_label
        self.latest_img1_label = img1_label

        img0 = self.create_img(img0_dir)
        img1 = self.create_img(img1_dir)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_label!=img0_label)],dtype=np.float32))
    
    def __len__(self):
        return len(self.meta_data)

    def create_img(self, img_dir):
        image = Image.open(img_dir).convert("RGB")
        image = image.resize((128,128))
        return self.transform(image)

    def get_img_loc(self, file_idx):
        return os.path.join(self.main_dir, self.total_imgs[file_idx])
        
    def get_random_img(self):
        is_img_found = False
        img0_idx = None
        img0_label = None
        meta_data = self.meta_data.sample()
        filename = meta_data['X_ray_image_name'].values[0]
        img0_label = meta_data['Label_1_Virus_category'].values[0]
        img0_idx = self.find_img(filename)
        return img0_label, img0_idx
    
    def find_img(self,filename):
        try:
            img0_idx = self.total_imgs.index(filename)
            return img0_idx
        except:
            # print("Img {0} not found ".format(filename))
            return None

    def __len__(self):
        return len(self.meta_data)

    def create_img(self, img_dir):
        image = Image.open(img_dir).convert("RGB")
        image = image.resize((128,128))
        return self.transform(image)

    def get_img_loc(self, file_idx):
        return os.path.join(self.main_dir, self.total_imgs[file_idx])
        
    def get_random_img(self):
        is_img_found = False
        img0_idx = None
        img0_label = None
        meta_data = self.meta_data.sample()
        filename = meta_data['X_ray_image_name'].values[0]
        img0_label = meta_data['Label_1_Virus_category'].values[0]
        img0_idx = self.find_img(filename)
        return img0_label, img0_idx
    
    def find_img(self,filename):
        try:
            img0_idx = self.total_imgs.index(filename)
            return img0_idx
        except:
            # print("Img {0} not found ".format(filename))
            return None

class SiameseNetworkDataset_Inference(Dataset):
    def __init__(self,main_dir,meta_data, transform):
        self.main_dir = main_dir
        self.meta_data = meta_data
        self.total_imgs = os.listdir(main_dir)
        self.transform = transform
        self.latest_img0_label = None
        self.latest_img1_label = None

    def __getitem__(self,index):
        # Random sample
        img0_label,img0_idx = self.get_random_img_covid()

        if img0_idx is None:
            return None
        img0_dir = self.get_img_loc(img0_idx)

        # Init img1
        img1_label = ""
        img1_dir = ""

        # We need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_label,img1_idx = self.get_random_img()
                if img1_idx is None:
                    continue
                if img0_label == img1_label:
                    img1_dir = self.get_img_loc(img1_idx)
                    break
        else:
            while True:
                #keep looping till a different class image is found
                img1_label, img1_idx = self.get_random_img()
                if img1_idx is None:
                    continue
                if img1_label != img0_label:
                    img1_dir = self.get_img_loc(img1_idx)
                    break

        img0_label_value = class_encode(img0_label)
        img1_label_value = class_encode(img1_label)

        img0 = self.create_img(img0_dir)
        img1 = self.create_img(img1_dir)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_label!=img0_label)],dtype=np.float32)),img0_label_value,img1_label_value

    def __getitem__old(self,index):
        # Random sample
        img0_label,img0_idx = self.get_random_img()

        if img0_idx is None:
            return None
        img0_dir = self.get_img_loc(img0_idx)

        # Init img1
        img1_label = ""
        img1_dir = ""

        # We need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_label,img1_idx = self.get_random_img()
                if img1_idx is None:
                    continue
                if img0_label == img1_label:
                    img1_dir = self.get_img_loc(img1_idx)
                    break
        else:
            while True:
                #keep looping till a different class image is found
                img1_label, img1_idx = self.get_random_img()
                if img1_idx is None:
                    continue
                if img1_label != img0_label:
                    img1_dir = self.get_img_loc(img1_idx)
                    break

        img0_label_value = class_encode(img0_label)
        img1_label_value = class_encode(img1_label)

        img0 = self.create_img(img0_dir)
        img1 = self.create_img(img1_dir)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_label!=img0_label)],dtype=np.float32)),img0_label_value,img1_label_value
    
    def __len__(self):
        return len(self.meta_data)

    def create_img(self, img_dir):
        image = Image.open(img_dir).convert("RGB")
        image = image.resize((128,128))
        return self.transform(image)

    def get_img_loc(self, file_idx):
        return os.path.join(self.main_dir, self.total_imgs[file_idx])
        
    def get_random_img(self):
        is_img_found = False
        img0_idx = None
        img0_label = None
        meta_data = self.meta_data.sample()
        filename = meta_data['X_ray_image_name'].values[0]
        img0_label = meta_data['Label_1_Virus_category'].values[0]
        img0_idx = self.find_img(filename)
        return img0_label, img0_idx

    def get_random_img_covid(self):
        is_img_found = False
        img0_idx = None
        img0_label = None
        while True:
            meta_data = self.meta_data.sample()
            filename = meta_data['X_ray_image_name'].values[0]
            img0_label = meta_data['Label_1_Virus_category'].values[0]
            img0_idx = self.find_img(filename)
            if img0_label == 'Virus':
                break
        return img0_label, img0_idx

    def find_img(self,filename):
        try:
            img0_idx = self.total_imgs.index(filename)
            return img0_idx
        except:
            # print("Img {0} not found ".format(filename))
            return None

    def __len__(self):
        return len(self.meta_data)

    def create_img(self, img_dir):
        image = Image.open(img_dir).convert("RGB")
        image = image.resize((128,128))
        return self.transform(image)

    def get_img_loc(self, file_idx):
        return os.path.join(self.main_dir, self.total_imgs[file_idx])
        
    def get_random_img(self):
        is_img_found = False
        img0_idx = None
        img0_label = None
        meta_data = self.meta_data.sample()
        filename = meta_data['X_ray_image_name'].values[0]
        img0_label = meta_data['Label_1_Virus_category'].values[0]
        img0_idx = self.find_img(filename)
        return img0_label, img0_idx
    
    def find_img(self,filename):
        try:
            img0_idx = self.total_imgs.index(filename)
            return img0_idx
        except:
            # print("Img {0} not found ".format(filename))
            return None

class DatasetMaker(object):
    def __init__(self,
        data_path='data/Covid19_Kaggle_CNN/',
        train_test_ratio = 0.25,
        shuffle_dataset=True,
        batch_size=64,
        batch_size_test = 64,
        stats=stats_global):

        self.train_img_dir = ""
        self.test_img_dir = ""
        self.df_meta = ""
        self.df_meta_summary = ""
        self.train_data = None
        self.test_data = None
        self.read_data(data_path)
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.stats = stats_global
        self.train_test_ratio = train_test_ratio,
        self.shuffle = shuffle_dataset

    def read_data(self, data_path):
        # Loading and Inspecting the Data
        img_path = data_path + 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset'
        self.train_img_dir = img_path + '/train'
        self.test_img_dir = img_path + '/test'
        self.df_meta = pd.read_csv(data_path+'Chest_xray_Corona_Metadata.csv')
        self.df_meta_summary = pd.read_csv(data_path+'Chest_xray_Corona_dataset_Summary.csv')
        # Replace null data points to 'unknown'
        self.df_meta.fillna('unknown', inplace=True)
        self.train_data = self.df_meta[self.df_meta['Dataset_type']=='TRAIN']
        self.test_data = self.df_meta[self.df_meta['Dataset_type']=='TEST']

    def define_classes(self):
        """
        Current dataset include two categories and one category has 2 labels \n
        To create a dataset with N labels, we need to merge some of categories. \n
        --- 
        1. No need to care about healthy cases since they can't contribute to classify covid cases from other unhealthy cases.
        2. Basically there is only two categories with considerable amount of data : Covid and bacteria. So choose them
        3. Except those two classes, we can merge all others as 'other' class.
        """
        self.train_data.loc[self.train_data['Label'].eq('Normal'), 'class'] = 'healthy'
        self.train_data.loc[(self.train_data['class'].ne('healthy') & self.train_data['Label_1_Virus_category'].eq('bacteria')), 'class'] = 'bacteria'
        self.train_data.loc[(self.train_data['class'].ne('healthy') & self.train_data['class'].ne('bacteria') & self.train_data['Label_2_Virus_category'].eq('COVID-19')), 'class'] = 'COVID-19'
        self.train_data.loc[(self.train_data['class'].ne('healthy') & self.train_data['class'].ne('bacteria') & self.train_data['class'].ne('COVID-19')), 'class'] = 'other'
        self.train_data['target'] = self.train_data['class'].map(target_dict)

    @staticmethod
    def create_vision_transformers(stats):
        """
        Augment data
        =====
        There is only around 40 images of covid cases
        To augment data, let's use torch's vision transformers
        """
        train_tfms = tt.Compose([tt.RandomCrop(128, padding=8, padding_mode='edge'), tt.ToTensor(), tt.Normalize(*stats, inplace = True)])
        test_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats, inplace = True)])
        return train_tfms, test_tfms

    @staticmethod
    def train_test_split(train_data,test_size=0.15,shuffle=True):
        train_ds, test_ds = train_test_split(train_data, test_size=0.25,random_state= 1, shuffle = True)
        train_ds, test_ds = train_ds.reset_index(drop=True), test_ds.reset_index(drop=True)
        return train_ds, test_ds

    def create_datasets(self):
        self.define_classes()
        train_ds, test_ds = self.train_test_split(self.train_data,self.train_test_ratio,self.shuffle)
        train_tfms, test_tfms = self.create_vision_transformers(self.stats)
        train_dataset = SiameseNetworkDataset(self.train_img_dir, train_ds, transform=train_tfms)    
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,                            
                                num_workers=0, pin_memory=True, collate_fn=my_collate)
        test_dataset = SiameseNetworkDataset(self.train_img_dir, test_ds, transform=test_tfms)
        test_loader = DataLoader(test_dataset , batch_size=self.batch_size_test, shuffle=False, 
                                num_workers=0, pin_memory=True, collate_fn=my_collate)
        return train_loader, test_loader

class DatasetMakerInference(object):
    def __init__(self,
        data_path='data/Covid19_Kaggle_CNN/',
        train_test_ratio = 0.25,
        shuffle_dataset=True,
        batch_size=64,
        batch_size_test = 64,
        stats=stats_global):

        self.train_img_dir = ""
        self.test_img_dir = ""
        self.df_meta = ""
        self.df_meta_summary = ""
        self.train_data = None
        self.test_data = None
        self.read_data(data_path)
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.stats = stats_global
        self.train_test_ratio = train_test_ratio,
        self.shuffle = shuffle_dataset

    def read_data(self, data_path):
        # Loading and Inspecting the Data
        img_path = data_path + 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset'
        self.train_img_dir = img_path + '/train'
        self.test_img_dir = img_path + '/test'
        self.df_meta = pd.read_csv(data_path+'Chest_xray_Corona_Metadata.csv')
        self.df_meta_summary = pd.read_csv(data_path+'Chest_xray_Corona_dataset_Summary.csv')
        # Replace null data points to 'unknown'
        self.df_meta.fillna('unknown', inplace=True)
        self.train_data = self.df_meta[self.df_meta['Dataset_type']=='TRAIN']
        self.test_data = self.df_meta[self.df_meta['Dataset_type']=='TEST']

    def define_classes(self):
        """
        Current dataset include two categories and one category has 2 labels \n
        To create a dataset with N labels, we need to merge some of categories. \n
        --- 
        1. No need to care about healthy cases since they can't contribute to classify covid cases from other unhealthy cases.
        2. Basically there is only two categories with considerable amount of data : Covid and bacteria. So choose them
        3. Except those two classes, we can merge all others as 'other' class.
        """
        self.train_data.loc[self.train_data['Label'].eq('Normal'), 'class'] = 'healthy'
        self.train_data.loc[(self.train_data['class'].ne('healthy') & self.train_data['Label_1_Virus_category'].eq('bacteria')), 'class'] = 'bacteria'
        self.train_data.loc[(self.train_data['class'].ne('healthy') & self.train_data['class'].ne('bacteria') & self.train_data['Label_2_Virus_category'].eq('COVID-19')), 'class'] = 'COVID-19'
        self.train_data.loc[(self.train_data['class'].ne('healthy') & self.train_data['class'].ne('bacteria') & self.train_data['class'].ne('COVID-19')), 'class'] = 'other'
        self.train_data['target'] = self.train_data['class'].map(target_dict)

    def define_classes_test(self):
        """
        Current dataset include two categories and one category has 2 labels \n
        To create a dataset with N labels, we need to merge some of categories. \n
        --- 
        1. No need to care about healthy cases since they can't contribute to classify covid cases from other unhealthy cases.
        2. Basically there is only two categories with considerable amount of data : Covid and bacteria. So choose them
        3. Except those two classes, we can merge all others as 'other' class.
        """
        self.test_data.loc[self.test_data['Label'].eq('Normal'), 'class'] = 'healthy'
        self.test_data.loc[(self.test_data['class'].ne('healthy') & self.test_data['Label_1_Virus_category'].eq('bacteria')), 'class'] = 'bacteria'
        self.test_data.loc[(self.test_data['class'].ne('healthy') & self.test_data['class'].ne('bacteria') & self.test_data['Label_2_Virus_category'].eq('COVID-19')), 'class'] = 'COVID-19'
        self.test_data.loc[(self.test_data['class'].ne('healthy') & self.test_data['class'].ne('bacteria') & self.test_data['class'].ne('COVID-19')), 'class'] = 'other'
        self.test_data['target'] = self.test_data['class'].map(target_dict)

    @staticmethod
    def create_vision_transformers(stats):
        """
        Augment data
        =====
        There is only around 40 images of covid cases
        To augment data, let's use torch's vision transformers
        """
        train_tfms = tt.Compose([tt.RandomCrop(128, padding=8, padding_mode='edge'), tt.ToTensor(), tt.Normalize(*stats, inplace = True)])
        test_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats, inplace = True)])
        return train_tfms, test_tfms

    @staticmethod
    def train_test_split(train_data,test_size=0.15,shuffle=True):
        train_ds, test_ds = train_test_split(train_data, test_size=0.25,random_state= 1, shuffle = True)
        train_ds, test_ds = train_ds.reset_index(drop=True), test_ds.reset_index(drop=True)
        return train_ds, test_ds

    def create_datasets(self):
        self.define_classes()
        train_ds, test_ds = self.train_test_split(self.train_data,self.train_test_ratio,self.shuffle)
        train_tfms, test_tfms = self.create_vision_transformers(self.stats)
        train_dataset = SiameseNetworkDataset_Inference(self.train_img_dir, train_ds, transform=train_tfms)    
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,                            
                                num_workers=0, pin_memory=True, collate_fn=my_collate)
        test_dataset = SiameseNetworkDataset_Inference(self.train_img_dir, test_ds, transform=test_tfms)
        test_loader = DataLoader(test_dataset , batch_size=self.batch_size_test, shuffle=False, 
                                num_workers=0, pin_memory=True, collate_fn=my_collate)
        return train_loader, test_loader

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = filter (lambda x:x is not None, batch)
    return default_collate(list(batch))

if __name__ == "__main__":
    dataset = DatasetMaker()
    train_loader, test_loader = dataset.create_datasets()
    dataiter = iter(train_loader)
    for x in range(100):
        example_batch = next(dataiter)
        concatenated = torch.cat((example_batch[0],example_batch[1]),0)

