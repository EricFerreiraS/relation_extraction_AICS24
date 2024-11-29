import numpy as np
import pandas as pd
import json
from PIL import Image
import glob
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import settings
import os

path = 'relation_extraction/relationships_features/'
csv_files = glob.glob(path + f"local_positive_unique_features_{settings.DATASET_TRANS}_{settings.MODEL}_{settings.DATASET}.*.csv")
df_list = (pd.read_csv(file) for file in csv_files)
df = pd.concat(df_list, ignore_index=True)
df.to_csv(f'relation_extraction/relationships_features/VG_netdissect_{settings.DATASET_TRANS}_{settings.MODEL}.csv')

df['id_image'] = df['name'].str.split('_').str[0]
df['object1'] = df['name'].str.split('_').str[1]
df['object1'] = df['object1'].str.replace('"','')
df['relation'] = df['name'].str.split('_').str[2]
df['relation'] = df['relation'].str.replace('"','')
df['object2'] = df['name'].str.split('_').str[3].str.split('.').str[0]
df['object2'] = df['object2'].str.replace('"','')

cp = pd.read_csv(f'NetDissect-Lite/result/pytorch_{settings.MODEL}_{settings.DATASET}/tally.csv')

# #### Creating a top-10 concept list for each image

df_10 = df.groupby('name').head(10).sort_values(['unit_rank'])
df_10['label'] = df_10['label'].str.split('-c').str[0]
df_10_list = pd.DataFrame(df_10.groupby(['name'])['label'].apply(list).reset_index())

df_10_list['id_image'] = df_10_list['name'].str.split('_').str[0]
df_10_list['object1'] = df_10_list['name'].str.split('_').str[1]
df_10_list['object1'] = df_10_list['object1'].str.replace('"','')
df_10_list['relation'] = df_10_list['name'].str.split('_').str[2]
df_10_list['relation'] = df_10_list['relation'].str.replace('"','')
df_10_list['object2'] = df_10_list['name'].str.split('_').str[3].str.split('.').str[0]
df_10_list['object2'] = df_10_list['object2'].str.replace('"','')

# #### Function to check if the concepts labeled in the relationship were detected in the top-10 concepts from NetDissect

def check_list(val,list_):
    if val in list_:
        return True
    else:
        return False

df_10_list['has_obj1'] = df_10_list.apply(lambda x: check_list(x.object1, x.label), axis=1)
df_10_list['has_obj2'] = df_10_list.apply(lambda x: check_list(x.object2, x.label), axis=1)
df_10_list['has_both'] = df_10_list.apply(lambda x: (x.has_obj1 and x.has_obj2), axis=1)

df_10_list['has_both'].value_counts()

print('% images that have both concepts in the NetDissection: ' + str(round((df_10_list['has_both'].value_counts()[1]/df_10_list.shape[0]) * 100)) + '%')

# #### Selecting only who has both concepts
df_selected = df_10_list[df_10_list['has_both']==True]

# #### Creating a dataframe to get the rating
#create a mirrored dataframe only to change the object positions, then analysing how many relationships were learned
df_10_list_line = df_10_list.copy()
df_selected_line = df_selected.copy()

df_10_list_line = df_10_list_line[['object1','relation','object2']].groupby(df_10_list_line[['object1','relation','object2']].apply(frozenset, axis=1))['object1'].count().reset_index()
df_10_list_line.rename(columns={'object1':'count','index':'set'},inplace=True)

df_selected_line = df_selected_line[['object1','relation','object2']].groupby(df_selected_line[['object1','relation','object2']].apply(frozenset, axis=1))['object1'].count().reset_index()
df_selected_line.rename(columns={'object1':'count','index':'set'},inplace=True)

#merging to get the rate
df_join = pd.merge(df_10_list_line, df_selected_line, how='inner', on=['set'],suffixes=('_tot','_selected'))

df_join['rate'] = df_join['count_selected'] / df_join['count_tot']

print('% relations that have more than 1 occurence: ' + str(round((df_join[df_join['count_tot']>1].shape[0]/df_join.shape[0]) * 100)) + '%')

df_join_mto = df_join[df_join['count_tot']>1]

os.makedirs(f'relation_extraction/{settings.DATASET_TRANS}_{settings.MODEL}/',exist_ok=True)

df_10_list.to_csv(f'relation_extraction/{settings.DATASET_TRANS}_{settings.MODEL}/VG_netdissect_top10.csv')
df_selected.to_csv(f'relation_extraction/{settings.DATASET_TRANS}_{settings.MODEL}/VG_netdissect_top10_selected.csv')
df_join.to_csv(f'relation_extraction/{settings.DATASET_TRANS}_{settings.MODEL}/VG_netdissect_top10_selected_rate.csv')
df_join_mto.to_csv(f'relation_extraction/{settings.DATASET_TRANS}_{settings.MODEL}/VG_netdissect_top10_selected_rate_mto.csv')