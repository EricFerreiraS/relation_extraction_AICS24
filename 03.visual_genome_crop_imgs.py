import numpy as np
import pandas as pd
import json
from PIL import Image
import shutil
import os
from tqdm import tqdm
import os
import settings
#crop the images based on the filtered file

image_df = pd.read_csv(f'relation_extraction/visual_genome_filtered_{settings.DATASET_TRANS}_{settings.MODEL}.csv')

image_id = image_df['image_id'].unique().astype(int)

print('Selecting images')
for img in tqdm(image_id):
    os.makedirs(f'relation_extraction/selected_images_VG/{settings.DATASET_TRANS}_{settings.MODEL}/',exist_ok=True)
    try:
        shutil.copy2(f'VG_100K/{img}.jpg', f'relation_extraction/selected_images_VG/{settings.DATASET_TRANS}_{settings.MODEL}/{img}.jpg')
    except:
        os.chdir('VG_100K/')
        cmd = f'wget https://cs.stanford.edu/people/rak248/VG_100K_2/{img}.jpg'
        os.system(cmd)
        shutil.copy2(f'VG_100K/{img}.jpg', f'relation_extraction/selected_images_VG/{settings.DATASET_TRANS}_{settings.MODEL}/{img}.jpg')


image_df['class'] = image_df['object2_name'] + '_' + image_df['relation'] + '_' + image_df['object1_name']

print('Cropping images')
for index, row in tqdm(image_df.iterrows()):
    if type(row['class']) is float:
        pass
    else:
        image = Image.open(f"relation_extraction/selected_images_VG/{settings.DATASET_TRANS}_{settings.MODEL}/{int(row['image_id'])}.jpg")

        x1_1, y1_1, width_1, height_1 = int(row['object1_x']), int(row['object1_y']), int(row['object1_w']), int(row['object1_h'])
        x1_2, y1_2, width_2, height_2 = int(row['object2_x']), int(row['object2_y']), int(row['object2_w']), int(row['object2_h'])

        cropped_image_1 = image.crop((x1_1, y1_1, x1_1 + width_1, y1_1 + height_1))
        cropped_image_2 = image.crop((x1_2, y1_2, x1_2 + width_2, y1_2 + height_2))

        new_image = Image.new('RGBA', image.size, (0, 0, 0, 0))    
        new_image.paste(cropped_image_1, (x1_1, y1_1))
        new_image.paste(cropped_image_2, (x1_2, y1_2))
        
        img_name = str(int(row['image_id'])) + '_' + row['class'] +'.jpg'
        img_name = img_name.replace("'","")
        img_name = img_name.replace("[","")
        img_name = img_name.replace("]","")
        counter = 1

        os.makedirs(f'relation_extraction/relationships_images_VG/{settings.DATASET_TRANS}_{settings.MODEL}/',exist_ok=True)

        file_path = os.path.join(f"relation_extraction/relationships_images_VG/{settings.DATASET_TRANS}_{settings.MODEL}/", img_name)

        while os.path.isfile(file_path):
            file_name, ext = os.path.splitext(img_name)
            file_name = f"{file_name}_{counter}{ext}"
            file_path = os.path.join(f'relation_extraction/relationships_images_VG/{settings.DATASET_TRANS}_{settings.MODEL}/', file_name)
            counter += 1
        new_image = new_image.convert("RGB")
        new_image = new_image.crop(new_image.getbbox())
        new_image.save(file_path)