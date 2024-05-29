#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import anndata as ad
from sklearn.decomposition import NMF

#This script will filter out rods misannotated as cones. These rods were likely misannotated because they were assigned their cell type in the global UMAP space, instead of the zoomed in subset that we are doing. 

#Here we grab our file output from Pearson_NMF, since this will have our NMF scores
path ="kasiadata/PearsonNormalized_ConeSubset.h5ad"
KHadata = sc.read_h5ad(path)

#NMF2 in this iteration had rod genes as its highest loads. If you are rerunning the Pearson_NMF, its likely a different NMF will have the rod signature, since the order of patterns retreived can be different every run. 
Rod_ID = []
for i in list(KHadata.obs['NMF2'].values):
    #I am setting a threshold of 1.8, though this number may be different on other NMF runs. 
    if i >= 1.8:
        Rod_ID.append('Rod')
    else:
        Rod_ID.append('Cone')
        
KHadata.obs['Rod_ID']=Rod_ID

#Here we have a plotter that just serves as a sanity check for what you are calling a rod. 
"""
query = "Rod_ID"
sc.pl.umap(KHadata, 
           color = query,
           color_map = "rocket_r", 
           size = 8, 
)

sc.pl.umap(KHadata, 
           color = "NRL",
           color_map = "rocket_r", 
           size = 8, 
)
"""

CONEadata = KHadata[KHadata.obs['Rod_ID']=='Cone']
del CONEadata.obs['Rod_ID']

for i in list(CONEadata.obs):
    if "NMF" in i:
        del CONEadata.obs[i]

del CONEadata.obsm

print(CONEadata)

CONEadata.write_h5ad("kasiadata/ConeSubset_RodFiltered.h5ad")


#%%
    
