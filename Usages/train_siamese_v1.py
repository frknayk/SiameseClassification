from termcolor import colored
from torch.autograd.grad_mode import F 
from dataset_makers.dataset_siamese import DatasetMaker
from runner import Runner
from models.siamese_v1 import Siamese_v1
from models.models_utils import to_device, get_default_device, DeviceDataLoader

device = get_default_device()
dataset = DatasetMaker(batch_size=64,batch_size_test=8)
train_loader, test_loader = dataset.create_datasets()
print(colored('Train and test loaders are created !', 'green', attrs=['reverse', 'blink']) ) 
train_loader = DeviceDataLoader(train_loader, device)
print(colored('Train loader sent to GPU', 'green', attrs=['blink']) ) 
test_loader = DeviceDataLoader(test_loader, device)
print(colored('Test loader sent to GPU', 'green', attrs=['blink']) ) 
training_model = to_device(Siamese_v1(3, 4),device)

runner = Runner(model=training_model,model_name='Siamese_v1_val_Deneme',log=True)
training_params = runner.get_default_params()
training_params['epochs'] = 150
runner.fit_siamese(
    train_loader=train_loader,
    test_loader=test_loader,
    training_params = training_params)
