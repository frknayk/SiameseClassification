from termcolor import colored
from torch.autograd.grad_mode import F 
from dataset_makers.dataset_siamese import DatasetMaker
from runner import Runner
from models.siamese_v2 import Siamese_v2
from models.models_utils import to_device, get_default_device, DeviceDataLoader

device = get_default_device()
dataset = DatasetMaker(batch_size=16,batch_size_test=8)
train_loader, test_loader = dataset.create_datasets()
train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)
training_model = to_device(Siamese_v2(3, 1),device)
runner = Runner(model=training_model,model_name='Siamese_v2',log=False)
training_params = runner.get_default_params()
training_params['epochs'] = 50
runner.fit_siamese_2(
    train_loader=train_loader,
    test_loader=test_loader,
    training_params = training_params)
