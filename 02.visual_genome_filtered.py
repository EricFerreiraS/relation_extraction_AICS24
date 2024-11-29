import numpy as np
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm
import os
path='NetDissect-Lite/' #path for the NetDissect results
import settings

#filtering the list of images based on the features presented in netdissect
local_features = pd.read_csv(path+f'result/local_positive_unique_features_{settings.DATASET_TRANS}_{settings.MODEL}_{settings.DATASET}.csv')

if settings.DATASET_TRANS == 'cifar10':
    local_features['class'] = local_features['class'].str.replace('_','')

lf_10 = local_features.groupby('name').head(10).sort_values(['class','unit_rank'])
lf_10_list = pd.DataFrame(lf_10.groupby(['name','class'])['label'].apply(list).reset_index())

con = set()
for i in lf_10_list['label'].values:
    for k in i:
        con.add(k)

concepts = str(list(con)).replace('\'','"')
concepts = concepts.replace('-c','')

rel = json.load(open('relation_extraction/relationships.json','r')) #json from Visual Genome with the relations

image_df = pd.DataFrame(data={'image_id':[],'relation':[],'relation_id':[],'object1_id':[],'object1_name':[],'object1_x':[],'object1_y':[],'object1_h':[],'object1_w':[],'object2_id':[],'object2_name':[],'object2_x':[],'object2_y':[],'object2_h':[],'object2_w':[]})

for img in tqdm(rel):
    img_id = img['image_id']
    for relation in img['relationships']:

        #checking with the concepts belongs to the Netdissection list
        obj1_name, obj2_name = '',''
        if ('names' in relation['object'] and any(item in relation['object']['names'] for item in concepts)):
            obj1_name = relation['object']['names']
        elif ('name' in relation['object'] and relation['object']['name'] in concepts):
            obj1_name = relation['object']['name']

        if ('names' in relation['subject'] and any(item in relation['subject']['names'] for item in concepts)):
            obj2_name = relation['subject']['names']
        elif ('name' in relation['subject'] and relation['subject']['name'] in concepts):
            obj2_name = relation['subject']['name']

        if obj1_name != '' and obj2_name != '':
            
            #relation
            relation_id = relation['relationship_id']
            relation_name = relation['predicate'].lower()

            #object1 data
            obj1_id = relation['object']['object_id']
            obj1_x = relation['object']['x']
            obj1_y = relation['object']['y']
            obj1_w = relation['object']['w']
            obj1_h = relation['object']['h']
            
            #object2 data
            obj2_id = relation['subject']['object_id']
            obj2_x = relation['subject']['x']
            obj2_y = relation['subject']['y']
            obj2_w = relation['subject']['w']
            obj2_h = relation['subject']['h']
            
            new_row = {'image_id':img_id,'relation':relation_name,'relation_id':relation_id,'object1_id':obj1_id,'object1_name':obj1_name,'object1_x':obj1_x,'object1_y':obj1_y,'object1_h':obj1_h,'object1_w':obj1_w,'object2_id':obj2_id,'object2_name':obj2_name,'object2_x':obj2_x,'object2_y':obj2_y,'object2_h':obj2_h,'object2_w':obj2_w}

            new_df = pd.DataFrame([new_row])
            image_df = pd.concat([image_df,new_df],ignore_index=True)
            
image_df.to_csv(f'relation_extraction/visual_genome_filtered_{settings.DATASET_TRANS}_{settings.MODEL}.csv')