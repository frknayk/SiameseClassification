import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
sns.set()


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')

def report_classification(targets, predictions):
    con_mat = confusion_matrix(targets, predictions)
    con_mat = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize = (10,10))
    plt.title('CONFUSION MATRIX')
    sns.heatmap(con_mat, cmap='coolwarm',
                yticklabels=['Healthy', 'Bacteria','COVID-19','other virus'],
                xticklabels=['Healthy', 'Bacteria','COVID-19','other virus'],
                annot=True)
    print(classification_report(targets, predictions))    
