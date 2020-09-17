__author__ = "Sereda"
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import torch
import numpy as np




def plot_ecg_to_fig(ecg, png_name=None):
    numleads = len(ecg)
    fig, axs = plt.subplots(numleads, 1, figsize=(8, 4), sharex=True, sharey=True)
    axs = axs.ravel()

    for i in range(numleads):
        axs[i].plot(ecg[i])

    if png_name is not None:
        fig.savefig(png_name)
    return fig


def save_models(filename, folder, generator, discriminator):
    os.makedirs( folder, exist_ok=True)
    PATH = folder + "/" + filename + '.tar'
    torch.save({
        'D_state_dict': discriminator.state_dict(),
        'G_state_dict': generator.state_dict()

    }, PATH)
    print("Models were saved")


def save_training_curves(G_losses, D_losses, InfoLosses, folder):
    path = folder + "/training_curves.png"
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.plot(InfoLosses, label="Discriminator")
    plt.xlabel('epoches')
    plt.ylabel('loss per epoch')
    plt.title('GAN errors on train set')
    plt.savefig(path)
    plt.clf()

def savefig_first_lead_one_ax(ecgs, folder,filename, title):
    os.makedirs(folder, exist_ok=True)
    plt.title(title)
    plt.ylabel('first lead signal')
    path = folder + "/"+filename+".png"

    num_ecgs = ecgs.shape[0]
    for i in range(num_ecgs):
        ecg_i = ecgs[i][0]
        plt.plot(ecg_i)

    plt.savefig(path)
    plt.clf()

def savefig_first_lead_several_axs(ecgs, folder,filename, title):
    os.makedirs(folder, exist_ok=True)
    plt.title(title)
    plt.ylabel('first lead signal')
    path = folder + "/"+filename+"_sev.png"

    num_ecgs = ecgs.shape[0]
    fig, axs = plt.subplots(num_ecgs, 1, figsize=(8, 2*num_ecgs), sharex=True, sharey=True)
    axs = axs.ravel()
    for i in range(num_ecgs):
        ecg_i = ecgs[i][0]
        axs[i].plot(ecg_i)

    plt.savefig(path)
    plt.close(fig)


def save_batch_to_images(id, ecgs):
    """
    Save batch of ecgs to html, and the first ecg - also to png.
    :param id: (int or whatever) some unique number for use in filenames
    :param ecgs: (numpy array) with shape (num_ecgs, num_channels, len_of_ecg)
    :return:
    """
    html = ''
    num_ecgs = ecgs.shape[0]
    for i in range(num_ecgs - 1):
        ecg = ecgs[i]
        # save one ecg as png
        png_name = None
        if i==0:
            png_name = "images/" + str(id) + '.png'
        fig = plot_ecg_to_fig(ecg, png_name)
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + 'descrition.. <br>'
        plt.close(fig)
    filename = "images/" + str(id) + '.html'
    with open(filename, 'w') as f:
        f.write(html)