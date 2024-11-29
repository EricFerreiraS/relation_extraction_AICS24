import torch
from loader.model_loader import loadmodel
from PIL import Image
from torchvision import transforms as T
import os
import csv
import pandas as pd
import settings
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import numpy as np
import cv2 as cv
import pickle
device = torch.device('cpu') 
from collections import Counter
import fnmatch
import gc

from tqdm import tqdm

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def init_model():
    model = loadmodel(hook_feature)
    model.eval()

    activations_avg=[]
    def hook_fn(m, i, o):
        unit_avg={}
        for grad in o:
            try:
                for i,j in enumerate(grad):
                    ##avg
                    unit_avg[i+1]=j.mean().item()
                activations_avg.append(unit_avg)
            except AttributeError: 
                print ("None found for Gradient")

    print(settings.MODEL)
    if settings.MODEL == 'resnet152' or settings.MODEL == 'resnet50':
        layer = model._modules['layer4']
    elif settings.MODEL == 'densenet161':
        layer = model._modules['features']
    else:
        print('model should be verified')
        raise Exception('Settings Error')
    layer.register_forward_hook(hook_fn)
    return model, activations_avg

model, activations_avg = init_model()
print('Extracting features')
transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.4715, 0.4413, 0.4020],
                                    std=[0.2732, 0.2650, 0.2742])])
path = f'relation_extraction/relationships_images_VG/{settings.DATASET_TRANS}_{settings.MODEL}/'

imgs = os.listdir(path)
batch = 50000
not_used=[]
data_class=[]
image_name=[]
count = 0
last_count = 199999
tot = int(len(fnmatch.filter(os.listdir(path),'*.*')))
for id,img in tqdm(enumerate(imgs)):
    if id < last_count:
        pass
    else:
        try:
            image = Image.open(path+img)
            x = transform(image).unsqueeze(dim=0).to(device)
        except:
            image = Image.open(path+img).convert("RGB")
            x = transform(image).unsqueeze(dim=0).to(device)
        try:
            model(x)
            if settings.DATASET_TRANS != 'cifar10':
                name = str(img.split('.')[0])[:-4]#action40
            else:
                name = str(img.split('.')[0])[:-5]#cifar10
            
            data_class.append(name)
            image_name.append(img)
        except:
            not_used.append(id+1)

        count += 1
        if count == batch or id == tot - 1:
            print(id)

            d_avg = pd.DataFrame(activations_avg)
            n = pd.DataFrame(image_name, columns=['name'])
            dt_avg = d_avg.merge(n,how='inner',left_index=True, right_index=True)

            pos_df=[]
            for i in tqdm(dt_avg.values):
                cl = i[-1:]
                rank_p = (i[:-1].ravel().argsort()[-30:]+1)
                pos_df.append(list(np.append(rank_p,cl)))

            positive_df = pd.DataFrame(pos_df).rename(
                columns={0:'29',1:'28',2:'27',3:'26',4:'25',5:'24',6:'23',7:'22',8:'21',9:'20',
                        10:'19',11:'18',12:'17',13:'16',14:'15',15:'14',16:'13',17:'12',18:'11',19:'10',
                        20:'9',21:'8',22:'7',23:'6',24:'5',25:'4',26:'3',27:'2',28:'1',29:'0',30:'name'}).set_index(['name']).stack()
            positive_df = pd.DataFrame(positive_df).rename(columns={0:'unit'}).reset_index().rename(
                columns={'level_0':'class','level_1':'unit_rank'})

            net_result = pd.read_csv(f'NetDissect-Lite/result/pytorch_{settings.MODEL}_{settings.DATASET}/tally.csv')

            positive_net=positive_df.merge(net_result,on='unit',how='inner')

            positive_net.unit_rank = positive_net.unit_rank.astype(np.int16)
            pos_unique = positive_net.sort_values(['name','unit_rank']).drop_duplicates(['name','label']).groupby(['name']).head(20)

            pos_unique.to_csv(f'relation_extraction/relationships_features/local_positive_unique_features_{settings.DATASET_TRANS}_{settings.MODEL}_{settings.DATASET}.{id}.csv',sep=',',encoding='utf-8', index=False)

            
            del model
            del activations_avg
            del not_used
            del data_class
            del image_name
            del count
            del pos_unique
            del net_result
            del positive_df
            del d_avg
            del n

            gc.collect()
            
            not_used=[]
            data_class=[]
            image_name=[]
            model, activations_avg = init_model()
            count = 0
            print('Finished a epoch')