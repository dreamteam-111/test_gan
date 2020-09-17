__author__ = "Sereda"
import numpy as np
import json
import easygui
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from InfoGAN.discriminator import Discriminator
from InfoGAN.train import Params

"""
    Сей файл предназначен для того, чтобы своими глазами посмотреть избирательность реакций дискриминатора (или ее отсуствие)
    Иными словами, хотим ответить на вопрос: является ли дискриминатор детектором каких-то характерных ситуаций в сигнале ЭКГ
    Получает обученный дискриминатор, экгшку, и прикладывает дискриминатор ко всем возможным точкам.
    Результат отрисовывается в result_pic_name.
"""

def get_ecg_lead_signal():
    BWR_PATH = "/home/user2/Desktop/"
    FILENAME = "ecg_data_200.json"
    #ALL_LEADS_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    json_file = BWR_PATH + FILENAME
    lead_name = 'i'
    patient_id = '50653582' # выбирать глядя в test.html, кот. формуируется скриптом Dataset/explore_set.py
    with open(json_file, 'r') as f:
        data = json.load(f)
        ecg_node = data[patient_id]
        ecg_lead_signal = ecg_node['Leads'][lead_name]['Signal']
        return ecg_lead_signal
    return None





def get_all_patches(ecg_lead_signal, patch_len):
    signal_len = len(ecg_lead_signal)
    print ("len of signal = " + str(signal_len))
    print("desired len of patch = " + str(patch_len))
    half_patch_len = int(patch_len/2) + 1

    start_point = 0 + half_patch_len
    end_point = signal_len - half_patch_len

    coords_iterator = range(start_point, end_point)
    all_patches = []
    for patch_center in coords_iterator:
        patch_start = patch_center - half_patch_len
        patch_end = patch_start + patch_len
        patch = ecg_lead_signal[patch_start:patch_end]
        all_patches.append(patch)

    numpy_patches = np.array(all_patches)
    numpy_patches = np.expand_dims(numpy_patches, axis=1) # нужно три измерения, второе из них для кол-ва каналов, тут каналов 1
    print("generated batch with shape: " + str(numpy_patches.shape))
    coords = np.fromiter(coords_iterator, int)
    return numpy_patches, coords

def get_trained_discriminator():
    hyperparams_json_file = easygui.fileopenbox("выберите гиперпараметры")
    with open(hyperparams_json_file) as json_file:
        data = json.load(json_file)
    print (data)
    path_to_checkpoint = "./"+ str(data['experiment_folder']) + "/checkpoints/LAST.tar"
    GAN_checkpoint = torch.load(path_to_checkpoint)
    n_classes = data['n_classes']
    code_dim = data['code_dim']
    patch_len = data['patch_len']
    num_channels = data['num_channels']
    model = Discriminator(n_classes, code_dim, patch_len, num_channels)
    model.load_state_dict(GAN_checkpoint['D_state_dict'])
    model.eval()
    return model

def feed_batch_to_discriminator(model, numpy_batch):
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        model.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    ecg_batch = Variable(Tensor(numpy_batch))
    validity, label, latent_code = model(ecg_batch)
    return validity.detach().numpy(), label.detach().numpy(), latent_code.detach().numpy()

def visualise_output_of_discriminator( validity, label, latent_code, ecg_signal, coords, len_of_window):
    print ("got validitiy shape " + str(validity.shape))
    print("got label shape " + str(label.shape))
    print("got latent_code shape " + str(latent_code.shape))

    n_plots = 1+ validity.shape[1] +label.shape[1] + latent_code.shape[1]

    fig, axs = plt.subplots(n_plots, 1, sharex=True, sharey=False)
    axs = axs.ravel()

    axs[0].plot(ecg_signal[0:len_of_window])
    axs[0].set_title("ecg_signal")

    axs[1].plot(coords[0:len_of_window], validity[0:len_of_window,0],"r*")
    axs[1].set_title("true/false")


    axs[2].plot(coords[0:len_of_window], latent_code[0:len_of_window])
    axs[2].set_title("latent code (uniform (-1,1))")

    n_classes = label.shape[1]
    for i in range(0, n_classes):
        ax_i = 3 + i
        axs[ax_i].plot(coords[0:len_of_window], label[0:len_of_window, i],0)
        axs[ax_i].set_title("discrete " + str(i))



    plt.savefig("discr_result.png")


if __name__ == "__main__":
    # берем одно отведение каккой-то из экгшек
    ecg_lead_signal = get_ecg_lead_signal()

    #загужаем предварительно обученный дискриминатор
    discriminator = get_trained_discriminator()

    # делаем нарезку из экгшки
    numpy_patches, coords = get_all_patches(ecg_lead_signal, discriminator.patch_len)

    #скармливаем эту нарезку дискриминатору
    validity, label, latent_code = feed_batch_to_discriminator(discriminator, numpy_patches)

    # отрисовываем его ответ в картинку
    visualise_output_of_discriminator(validity, label, latent_code, ecg_lead_signal, coords, len_of_window=1500)

