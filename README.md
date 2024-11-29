# Towards Understanding Deep Representations in CNN: from Concepts to Relations Extraction via Knowledge Graphs - AICS'24

## General Aspects
The codes presented here were developed for the AICS'24 conference paper. To execute the methods presented here, the NetDissect code must be executed first. The project link is https://github.com/CSAILVision/NetDissect-Lite. I ran it using the "settings.py" file presented here. Secondly, the code in https://github.com/EricFerreiraS/disentangled_representation-concept_ranking should be executed to get the top 10 global and local concepts.

In this paper we used the Action40 dataset - http://vision.stanford.edu/Datasets/40actions.html - and CIFAR-10 - https://www.cs.toronto.edu/~kriz/cifar.html - to extract the features. The Visual Genome - https://homes.cs.washington.edu/~ranjay/visualgenome/api.html - and the Commonsense Knowledge Graph datasets - https://drive.google.com/drive/u/1/folders/16347KHSloJJZIbgC9V5gH7_pRx0CzjPQ - are required as well.

In order to use the CSKG, I recommend use the conda enviroment presented here: https://kgtk.readthedocs.io/en/latest/install/ 

## Reproducibility
The structure for the reproducibility is:
- Download the Action40 dataset - http://vision.stanford.edu/Datasets/40actions.html
- Download the CIFAR-10 dataset - https://www.cs.toronto.edu/~kriz/cifar.html
- Download the Visual Genome dataset - https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
- Download the CSKG dataset - https://drive.google.com/drive/u/1/folders/16347KHSloJJZIbgC9V5gH7_pRx0CzjPQ (cskg.tsv.gz file)
- Run NetDissect (https://github.com/CSAILVision/NetDissect-Lite) using the "settings.py"
- Run Concept extraction from https://github.com/EricFerreiraS/disentangled_representation-concept_ranking
- Run the codes, following the numeric order. 

## Reference
```
```