import numpy as np
import pandas as pd
from kgtk.functions import kgtk, kypher
import networkx as nx
from pyvis.network import Network
import graph_tool.all as gt
import graph_tool.draw as gt_draw
import sys
from tqdm import tqdm
import os

path='NetDissect-Lite/' #path for the NetDissect results

import settings

global_features = pd.read_csv(path+f'result/global_positive_unique_features_{settings.DATASET_TRANS}_{settings.MODEL}_{settings.DATASET}.csv')
local_features = pd.read_csv(path+f'result/local_positive_unique_features_{settings.DATASET_TRANS}_{settings.MODEL}_{settings.DATASET}.csv')

if settings.DATASET_TRANS == 'cifar10':
    global_features['class'] = global_features['class'].str.replace('_','')
    local_features['class'] = local_features['class'].str.replace('_','')

gf_10 = global_features.groupby('class').head(10).sort_values(['class','unit_rank'])
gf_10_list = pd.DataFrame(gf_10.groupby('class')['label'].apply(list).reset_index())

lf_10 = local_features.groupby('name').head(10).sort_values(['class','unit_rank'])
lf_10_list = pd.DataFrame(lf_10.groupby(['name','class'])['label'].apply(list).reset_index())
lf_10_list_class = pd.DataFrame(lf_10.groupby(['class'])['label'].apply(set).reset_index())

print('Creating the datasets')
con = set()
for i in gf_10_list['label'].values:
    for k in i:
        con.add(k.replace('\'','"').replace('-c',''))
con = str(list(con)).replace("'",'"')

kgtk(f"""
    query -i cskg.tsv -o relation_extraction/graphs/gf_10_list_{settings.DATASET_TRANS}_{settings.MODEL}.tsv
    --match '(n1)-[r]->(n2)'
    --where '(lower(n1.label) in {con} or lower(n2.label) in {con}) and n1 != n2'
""")

con = set()
for i in lf_10_list['label'].values:
    for k in i:
        con.add(k.replace('\'','"').replace('-c',''))
con = str(list(con)).replace("'",'"')
kgtk(f"""
    query -i cskg.tsv -o relation_extraction/graphs/lf_10_list_{settings.DATASET_TRANS}_{settings.MODEL}.tsv
    --match '(n1)-[r]->(n2)'
    --where '(lower(n1.label) in {con} or lower(n2.label) in {con}) and n1 != n2'
""")

con = set()
for i in lf_10_list_class['label'].values:
    for k in i:
        con.add(k.replace('\'','"').replace('-c',''))
con = str(list(con)).replace("'",'"')
kgtk(f"""
    query -i cskg.tsv -o relation_extraction/graphs/lf_10_list_class_{settings.DATASET_TRANS}_{settings.MODEL}.tsv
    --match '(n1)-[r]->(n2)'
    --where '(lower(n1.label) in {con} or lower(n2.label) in {con}) and n1 != n2'
""")

## one graph with all relations
con = set()
for i in gf_10_list['label'].values:
    for k in i:
        con.add(k.replace('\'','"').replace('-c',''))

for i in lf_10_list['label'].values:
    for k in i:
        con.add(k.replace('\'','"').replace('-c',''))

for i in lf_10_list_class['label'].values:
    for k in i:
        con.add(k.replace('\'','"').replace('-c',''))
con = str(list(con)).replace("'",'"')
kgtk(f"""
    query -i cskg.tsv -o relation_extraction/graphs/graph_full_{settings.DATASET_TRANS}_{settings.MODEL}.tsv
    --match '(n1)-[r]->(n2)'
    --where '(lower(n1.label) in {con} or lower(n2.label) in {con}) and n1 != n2'
""")

b = kgtk(f"""
    query -i relation_extraction/graphs/gf_10_list_{settings.DATASET_TRANS}_{settings.MODEL}.tsv
""")

b[(b['node1;label']=='person') & (b['node2;label']=='hand')][['node1;label','node2;label','relation;label']].drop_duplicates()

print('Creating the Graphs - Global')
for i in tqdm(gf_10_list.values):
    l_r = str(list(i[1])).replace('\'','"')
    os.makedirs(f'relation_extraction/graphs/classes/{settings.DATASET_TRANS}_{settings.MODEL}/',exist_ok=True)
    rel = kgtk(f"""
                query -i relation_extraction/graphs/gf_10_list_{settings.DATASET_TRANS}_{settings.MODEL}.tsv -o relation_extraction/graphs/classes/{settings.DATASET_TRANS}_{settings.MODEL}/class_{i[0]}.tsv
                --match '(n1)-[r]->(n2)'
                --where 'lower(n1.label) in {l_r} and lower(n2.label) in {l_r} and n1.label != n2.label'
            """)
    
print('Creating the Graphs - Local')
for i in tqdm(lf_10_list.values):
    l_r = str(list(i[2])).replace('\'','"').replace('-c','')
    os.makedirs(f'relation_extraction/graphs/images/{settings.DATASET_TRANS}_{settings.MODEL}/{i[1]}/',exist_ok=True)
    rel = kgtk(f"""
                query -i relation_extraction/graphs/lf_10_list_{settings.DATASET_TRANS}_{settings.MODEL}.tsv -o relation_extraction/graphs/images/{settings.DATASET_TRANS}_{settings.MODEL}/{i[1]}/{i[0]}.tsv
                --match '(n1)-[r]->(n2)'
                --where 'lower(n1.label) in {l_r} and lower(n2.label) in {l_r} and n1.label != n2.label'
            """)

def draw_network(df):
    net = Network(notebook=True)

    sources = df['node1;label']
    targets = df['node2;label']
    relations = df['relation;label']

    edge_data = zip(sources, targets, relations)

    for e in edge_data:
                    src = e[0]
                    dst = e[1]
                    rel = e[2]

                    net.add_node(src, src, title=src)
                    net.add_node(dst, dst, title=dst)
                    net.add_edge(src, dst, title=rel)
    return net

def show_kb(class_name):
    b = kgtk(f"""
            query -i relation_extraction/graphs/classes/{settings.DATASET_TRANS}_{settings.MODEL}/class_{class_name}.tsv
            """)
    net = draw_network(b)
    G = nx.from_pandas_edgelist(b,source='node1;label',target='node2;label',edge_attr='relation;label')
    return net.save_graph(f'concepts_graph_{class_name}_global_{settings.DATASET_TRANS}.html')