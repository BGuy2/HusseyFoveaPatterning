#%%

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import sys
import numpy as np
import scipy
import anndata as ad
import scanpy as sc
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
import umap
import statistics

#This is a scipt for making basic plots over a UMAP,making violin plots, and plotting NMF scores.
#I run this script in VS code with the Jupyter plugin, so I can run smaller code blocks(each of the individual plotting scripts) from this larger script.  

#path will be your filename. change to whatever dataset or subset you want to plot. 
path = 'kasiadata/Photoreceptor_Subset.h5ad'
data = sc.read_h5ad(path)

print(data)

#%%
#UMAP plotter
sc.pl.umap(data, 
           color = 'NRL',
           color_map = 'rocket_r',
           #palette={"Control": "tab:cyan","10day": "purple","200day": 'red',},
           size = 15, 
           outline_color = ('black','white'),
           alpha = 0.5,
           #save ='.png',
           )

#%%
#Violin Plots
sc.pl.violin(data,
             keys = 'nCount_RNA',
             groupby='treatment',
             jitter = 0.4,
             save = 'violinplot_countRNA_treatments.png')

#%%
#Scatterplot for other numeric variables in the dataset. 
#Since seaborn doesnt seem to like adata, i just made a dataframe with the count info. 

counts=data.to_df()

ax = sns.scatterplot(
                     x=data.obs['NMF1'], 
                     y=data.obs['NMF2'], 
                     s=3, 
                     alpha=.7,
                     hue = counts['NRL'], 
                     palette='rocket_r',
                     )
sns.move_legend(obj = ax, loc="upper left",bbox_to_anchor=(1, 1))
#plt.show()
plt.savefig("exampletitle.png")




# %%
