import torch
import os

# Mapping labels
codes = {
    'blues':0,
    'classical':1,
    'country':2,
    'disco':3,
    'hiphop':4,
    'jazz':5,
    'metal':6,
    'pop':7,
    'reggae':8,
    'rock':9
}

'''
check whether use the GPU
'''
def check_device():
    # check the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

'''
check whether the folder exists, if not, create one
'''
def check_folder(folder_path):
    if len(folder_path.split('/')) > 2:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    else:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
    return folder_path

