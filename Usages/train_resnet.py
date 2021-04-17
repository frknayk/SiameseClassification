from dataset_makers.dataset_maker import DatasetMaker
from runner import Runner
from models.resnet_9 import ResNet9
from models.models_utils import to_device, get_default_device, DeviceDataLoader

device = get_default_device()
dataset = DatasetMaker()
train_loader, test_loader = dataset.create_datasets()
train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)
training_model = to_device(ResNet9(3, 4),device)
runner = Runner(model=training_model,model_name='ResNet9')
training_params = runner.get_default_params()
runner.fit(
    train_loader=train_loader,
    test_loader=test_loader,
    training_params = training_params)