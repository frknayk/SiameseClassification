# Corona X-Ray Chest Images Classification with Siamese Network

## About
- In this project, I aimed to solve classification of Covid-19 x-ray chest images which is really rare in the dataset.
 (Dataset used in the project can be found here : https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset.
- Since the siamese networks are well-known with their ability rank differences or similarities, I realized they are capable of 
detecting very similar chest images even if they are noisy. Since the current networks only perform detection, true class of an image 
can only be detected with performing random detections with images from a known class. This only happens during inference, so purpose of 
training is training siamese networks to perform well on ranking similiarities.

## Usage
- All training/testing codes are under the ```/Usages``` folder.
- To train and test baseline model please run ```train_resnet.py```
- To train proposed method please run ```train_siamese_v3.py```
- Test proposed method and see performance of the trained network via ```test_siamese_v3.py```
    Please do not forget to change the path of the trained network regarding your own computer

- Please check the ```/Logs``` folder when the training continue or done. All weights are logged here together with learning curves.
  To see the learning curve run under the ```/Logs``` folder ```tensorboard --logdir=runs``` 

## Project Structure
- ```/models``` folder containst neural network architectures and their helpers
- ```/Usages``` training/testing codes
- ```/utils```  contains plotting or showing images with numerous of options
- ```/dataset_makers``` contains pytorch dataset loader objects for each method
- ```/data``` Data of the Covid X-ray images

## Results
<img width=640px height=480px src="Results\siamese_res_1.png" alt="SiameseDetection-1">

## Contact with me
To ask any question about the method, just contact with me via ```furkanayik@outlook.com```