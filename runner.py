import os
from models.models_utils import *
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm

# TODO: Make this a repository for any torch project!
def get_datetime():
    today = datetime.now()
    todays_date_full =  str(today.year)+"_"+str(today.month)+"_"+str(today.day)+"_"
    todays_date_full += str(today.hour) + "_" + str(today.minute) + "_" + str(today.second)
    return todays_date_full

def create_training_folder(log_path):
    try:
        os.makedirs(log_path)
    except Exception as e:
        print("Could not created the agents folder : ",e)

class Runner(object):
    """
    Run training/inference
    """
    def __init__(self, 
        model,
        model_name = '',
        mode='train',
        log=True):
        """Constructor

        Args:
            model (torch.nn): Classifier network
            mode (str, optional): Mode of the run. Defaults to 'train'.
        """
        self.model = model
        self.base_path = None
        self.log_enabled = log
        if mode == 'train' and self.log_enabled:
            current_date = get_datetime()
            if len(model_name) > 0:
                current_date = model_name + '_' +  current_date

            self.base_path = current_date
            create_training_folder("Logs/networks/"+current_date)
            self.writer = SummaryWriter(log_dir='Logs/runs/'+current_date)

    @staticmethod
    def get_default_params():
        training_params_default = {
            'epochs' : 50,
            'optimizer' : torch.optim.Adam,
            'lr' : 1e-3,
            'weight_decay' : 1e-4,
            'grad_clip' : 0.1
        }
        return training_params_default

    def save_model(self, path):
        """Loading Model"""
        try:
            torch.save(self.model.state_dict(), path)
        except Exception as e:
            raise "Network could not load : {0}".format(e)

    def load_model(self, path):
        """Loading Model"""
        try:
            self.model.load_state_dict(torch.load(path))
        except Exception as e:
            raise("Network could not load : {0}".format(e))

    def fit(self,
            train_loader,
            test_loader,
            training_params=None):
        """
        Train network for number of 'epochs' times. 
        Works well classical image classification tasks.

        Args:
            train_loader (torch.utils.data.Dataset): Dataset object
            test_loader (torch.utils.data.Dataset): Dataset object
            training_params (dict, optional): Default training parameters. Defaults to None.

        Returns:
            [list]: History of results as list
        """
        if training_params is None:
            training_params = self.get_default_params()

        epochs = training_params['epochs']
        optimizer_function = training_params['optimizer']
        lr = training_params['lr']
        weight_decay = training_params['weight_decay']
        grad_clip = training_params['grad_clip']

        torch.cuda.empty_cache()
        history = []

        # Set up custom optimizer with weight decay
        optimizer = optimizer_function(self.model.parameters(), lr, weight_decay=weight_decay)
        # Set up one-cycle learning rate scheduler
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, 
                                                    steps_per_epoch=len(train_loader))
        
        for epoch in range(epochs):
            # Training Phase 
            self.model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = self.model.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                # Gradient clipping
                if grad_clip: 
                    nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Record & update learning rate
                lrs.append(get_lr(optimizer))
                sched.step()
            
            # Validation phase
            result = evaluate(self.model, test_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            self.model.epoch_end(epoch, result)
            history.append(result)

            # Log            
            self.writer.add_scalar("loss", result['train_loss'], epoch)
            self.writer.add_scalar("val", result['val_loss'], epoch)
            self.writer.add_scalar("acc", result['val_acc'], epoch)
            self.save_model("Logs/networks/"+self.base_path+"/network_"+str(epoch)+'.pth')

        return history

    def fit_siamese(self,
            train_loader,
            test_loader,
            training_params=None):
        """
        Train network for number of 'epochs' times. 
        Works well classical image classification tasks.

        Args:
            train_loader (torch.utils.data.Dataset): Dataset object
            test_loader (torch.utils.data.Dataset): Dataset object
            training_params (dict, optional): Default training parameters. Defaults to None.

        Returns:
            [list]: History of results as list
        """
        if training_params is None:
            training_params = self.get_default_params()

        epochs = training_params['epochs']
        optimizer_function = training_params['optimizer']
        lr = training_params['lr']
        weight_decay = training_params['weight_decay']
        grad_clip = training_params['grad_clip']

        torch.cuda.empty_cache()

        # Set up custom optimizer with weight decay
        optimizer = optimizer_function(self.model.parameters(), lr, weight_decay=weight_decay)
        
        criterion = torch.nn.BCEWithLogitsLoss(size_average=True)

        # Set up one-cycle learning rate scheduler
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, 
                                                    steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            loss_history = []
            val_loss_history = []
            val_acc_history = []
            lrs = []
            self.model.train()
            
            for step, data in enumerate( tqdm(train_loader) ):
                img0, img1 , label = data
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                output = self.model(img0,img1)
                loss_contrastive = criterion(output,label)

                loss_contrastive.backward()

                # Gradient clipping
                if grad_clip: 
                    nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Record & update learning rate
                lrs.append(get_lr(optimizer))
                sched.step()

                epoch_loss = loss_contrastive.item()
                loss_history.append(epoch_loss)

                if step == 45:
                    break

            epoch_loss_mean = np.mean(loss_history)
            print("Epoch:{} || Loss {}\n".format(epoch,epoch_loss_mean ))

            torch.cuda.empty_cache()
            
            if epoch % 10 == 0:
                self.model.eval()
                val_loss_history = []
                for step, data in enumerate( tqdm(test_loader) ):
                    img0, img1 , label = data
                    img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                    output = self.model(img0,img1)
                    loss_contrastive = criterion(output,label)
                    epoch_loss = loss_contrastive.item()
                    val_loss_history.append(epoch_loss)
                val_loss_mean = np.mean(val_loss_history)
                print("|||| Validation Loss {}\n".format(val_loss_mean))
                if self.log_enabled:
                    self.writer.add_scalar("val_loss", val_loss_mean, epoch)

            if self.log_enabled:
                self.writer.add_scalar("loss", epoch_loss_mean, epoch)
                if epoch % 10 == 0:
                    self.save_model("Logs/networks/"+self.base_path+"/network_"+str(epoch)+'.pth')

        # Evaluate and calculate accuracy of trained agent
        val_loss, val_acc = evaluate_siamese(self.model, test_loader)
        print("Validation accuracy : {}\nValidation loss : {}\n".format(
            str(val_loss), str(val_acc) ))

        if self.log_enabled:
            self.save_model("Logs/networks/"+self.base_path+"/network_final.pth")

    def inference(self,test_loader,trained_path):
        """
        Train network for number of 'epochs' times. 
        Works well classical image classification tasks.

        Args:
            train_loader (torch.utils.data.Dataset): Dataset object
            test_loader (torch.utils.data.Dataset): Dataset object
            training_params (dict, optional): Default training parameters. Defaults to None.

        Returns:
            [list]: History of results as list
        """
        training_params = self.get_default_params()
        epochs = training_params['epochs']
        
        self.load_model(trained_path)

        torch.cuda.empty_cache()
        
        criterion = torch.nn.BCEWithLogitsLoss(size_average=True)

        for epoch in range(epochs):
            loss_history = []
            torch.cuda.empty_cache()
            self.model.eval()
            for step, data in enumerate( tqdm(test_loader) ):
                img0, img1 , label = data
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                output = self.model(img0,img1)
                loss_contrastive = criterion(output,label)
                epoch_loss = loss_contrastive.item()
                loss_history.append(epoch_loss)

            epoch_loss_mean = np.mean(loss_history)
            print("Epoch:{} || Loss {}\n".format(epoch,epoch_loss_mean ))

        # Evaluate and calculate accuracy of trained agent
        val_loss, val_acc = evaluate_siamese(self.model, test_loader)
        print("Validation accuracy : {}\nValidation loss : {}\n".format(
            str(val_loss), str(val_acc) ))

    def fit_siamese_2(self,
            train_loader,
            test_loader,
            training_params=None):
        """
        Train network for number of 'epochs' times. 
        Works well classical image classification tasks.

        Args:
            train_loader (torch.utils.data.Dataset): Dataset object
            test_loader (torch.utils.data.Dataset): Dataset object
            training_params (dict, optional): Default training parameters. Defaults to None.

        Returns:
            [list]: History of results as list
        """
        if training_params is None:
            training_params = self.get_default_params()

        epochs = training_params['epochs']
        optimizer_function = training_params['optimizer']
        lr = training_params['lr']
        weight_decay = training_params['weight_decay']
        grad_clip = training_params['grad_clip']

        torch.cuda.empty_cache()

        # Set up custom optimizer with weight decay
        optimizer = optimizer_function(self.model.parameters(), lr, weight_decay=weight_decay)
        
        criterion = torch.nn.BCEWithLogitsLoss(size_average=True)

        # Set up one-cycle learning rate scheduler
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, 
                                                    steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            loss_history = []
            val_loss_history = []
            val_acc_history = []
            self.model.train()
            
            for step, data in enumerate( tqdm(train_loader) ):
                img0, img1 , label = data
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                output = self.model(img0,img1)
                loss_contrastive = criterion(output,label)

                loss_contrastive.backward()

                # Gradient clipping
                if grad_clip: 
                    nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Record & update learning rate
                sched.step()

                epoch_loss = loss_contrastive.item()
                loss_history.append(epoch_loss)

                if step%50:
                    break

            epoch_loss_mean = np.mean(loss_history)
            print("Epoch:{} || Loss {}\n".format(epoch,epoch_loss_mean ))

            self.model.eval()
            val_loss_history = []
            for step, data in enumerate( tqdm(test_loader) ):
                img0, img1 , label = data
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                output = self.model(img0,img1)
                loss_contrastive = criterion(output,label)
                val_loss_history.append(loss_contrastive.item())
            val_loss_mean = np.mean(val_loss_history)
            print("||||||||||| Validation Loss {}\n".format(val_loss_mean))
