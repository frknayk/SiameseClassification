import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as tt
from PIL import Image
import pandas as pd

stats_global = ((0.0093, 0.0093, 0.0092),(0.4827, 0.4828, 0.4828))

target_dict = {'healthy' : 0,
               'bacteria' : 1,
               'COVID-19' : 2,
               'other' : 3}


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

class DatasetMaker(object):
    def __init__(self,
        data_path='data/Covid19_Kaggle_CNN/',
        train_test_ratio = 0.25,
        shuffle_dataset=True,
        batch_size=64,
        stats=stats_global):

        self.train_img_dir = ""
        self.test_img_dir = ""
        self.df_meta = ""
        self.df_meta_summary = ""
        self.train_data = None
        self.test_data = None
        self.read_data(data_path)
        self.batch_size = batch_size
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
        train_dataset = CustomDataSet(self.train_img_dir, train_ds, transform=train_tfms)    
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,                            
                                num_workers=0, pin_memory=True, collate_fn=my_collate)
        test_dataset = CustomDataSet(self.train_img_dir, test_ds, transform=test_tfms)
        test_loader = DataLoader(test_dataset , batch_size=2*self.batch_size, shuffle=False, 
                                num_workers=0, pin_memory=True, collate_fn=my_collate)
        return train_loader, test_loader

if __name__ == "__main__":
    from dataset_makers.dataset_maker import DatasetMaker
    from utils.utils_imgs import show_batch, plt

    dataset = DatasetMaker()
    train_loader, test_loader = dataset.create_datasets()
    show_batch(train_loader)
    plt.show()
