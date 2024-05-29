#%%
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import anndata as adata


#opening the data. KHfull_Cellxgene.h5ad should have everything formatted properly,

path = "kasiadata/KHfull_Cellxgene.h5ad"
KHadata = sc.read_h5ad(path)

print(set(list(KHadata.obs['Celltype'].values)))

#%%
#alright this is a simple subsetting script
#Here is all the Photoreceptors.

obs = KHadata.obs
print(obs)

PRyn = []
for i in obs['Celltype']:
    if i == 'Cone':
        PRyn.append(True)
    elif i == 'Rod':
        PRyn.append(True)
    elif i == 'PR Precursors':
        PRyn.append(True)
    else:
        PRyn.append(False)

KHadata.obs['Photoreceptor'] = PRyn

PRs = KHadata[ KHadata.obs["Photoreceptor"]== True]
cones = KHadata[KHadata.obs["Celltype"] == 'Cone']
PRs.write_h5ad( filename= 'kasiadata/PRSubset.h5ad')
cones.write_h5ad( filename= 'kasiadata/ConeSubset.h5ad')

