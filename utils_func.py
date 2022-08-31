import torch
import os
from PIL import Image
from tqdm import tqdm



def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')


def make_img_data(path, trans):
    files = os.listdir(path)
    data = [trans(Image.open(path+file)) for file in tqdm(files) if not file.startswith('.')]
    return data    


