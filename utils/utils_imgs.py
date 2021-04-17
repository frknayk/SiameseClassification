import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def plot_images(path,class_str,numdisplay):
    """Display X-Ray images"""
    fig, ax = plt.subplots(numdisplay,2, figsize=(15,2.5*numdisplay))
    for row,file in enumerate(path):
        image = plt.imread(file)
        ax[row,0].imshow(image, cmap=plt.cm.bone)
        ax[row,1].hist(image.ravel(), 256, [0,256])
        ax[row,0].axis('off')
        if row == 0:
            ax[row,0].set_title('Images')
            ax[row,1].set_title('Histograms')
    fig.suptitle('Class='+class_str,size=16)
    plt.show()

def display_class_images(img_path,dataset,train_or_test_str,classlabel,numdisplay):
    path = dataset[dataset['class']==classlabel]['X_ray_image_name'].values
    sample_path = path[:numdisplay]
    img_dir = os.listdir(img_path)+"/"+train_or_test_str
    sample_path = list(map(lambda x: os.path.join(img_dir,x), sample_path))
    plot_images(sample_path,classlabel,numdisplay)

def show_batch(dl):
    for images,labels in dl:
        print(images.shape, labels.shape)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=8).permute(1,2,0))
        break

def show_batch_all(dl,plt_active=False):
    for images,labels in dl:
        print(images.shape, labels.shape)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=8).permute(1,2,0))
        if plt_active:
            plt.show()
        print("Continue(Y or y) or Quit(Q or q)")
        ans = input()
        if ans.upper() == 'Q':
            break 

def imshow_simple(images):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images,nrow=8).permute(1,2,0))
    plt.show()

def imshow_multi2(image1,image2,labels):
    # create figure (fig), and array of axes (ax)
    f, axarr = plt.subplots(2)
    axarr[0].imshow(make_grid(image1,nrow=8).permute(1,2,0))
    axarr[1].imshow(make_grid(image2,nrow=8).permute(1,2,0))
    axarr[0].set_title(labels.numpy().tolist())  
    plt.tight_layout()
    plt.show()