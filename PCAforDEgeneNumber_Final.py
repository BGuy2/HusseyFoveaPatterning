#%%
from collections import Counter
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
import statistics as stat
from scipy.stats import zscore
import scProject as scP
import warnings
warnings.filterwarnings("ignore")

#This script is for us to determine how many genes are relevant for distinguishing the S and the LM cones from the Cowan/Roska dataset. 
#to do this we grab the top x DE genes between the S and LM. Here I have 9 different numbers of genes (gene_num). For each of these numbers, we look at the explained variance of their first PC, and plot that. 

path ='RoskaData/2D_Roska_PeripheralUMAP.h5ad'
data = sc.read_h5ad(path)

gene_num = [5,10,20,50,100,200,400,800,1000]
pc1_list= []

for i in gene_num:
    sc.tl.rank_genes_groups(data,
                            groupby='cell_type',
                            method = 'wilcoxon',
                            n_genes= i,
                            #reference='10day'
                            )
    treatmentstow = data.uns['rank_genes_groups']['names']
    top20 = treatmentstow[:20]

    scores = data.uns['rank_genes_groups']['scores']

    s_Identidic = {}
    lm_Identidic = {}
    lm_marks = []
    s_marks = []

    for one, two in treatmentstow:
        lm_marks.append(one)
        s_marks.append(two)
        
    lm_scores = []
    s_scores = []    

    for lm_weight, s_weight in scores:
        lm_scores.append(lm_weight)
        s_weight = s_weight
        s_scores.append(s_weight)

    count = 0
    for k in lm_marks:
        maxi = max(lm_scores)
        lm_Identidic[k]= lm_scores[count] / maxi
        s_Identidic[k]= 0
        count += 1
        
    count = 0 

    for k in s_marks:
        maxi = max(s_scores)
        s_Identidic[k]= s_scores[count] / maxi
        lm_Identidic[k]=0
        count += 1

    #print("Done Sorting")

    listofgenes = lm_marks + s_marks
    #print(listofgenes)
    sc.pp.scale(data)
    sourceDF = data.to_df()
    coneDF = pd.DataFrame()
    for k in listofgenes:
        if k in sourceDF:
            coneDF[k] = list(sourceDF[k].values)

    CGadata = ad.AnnData(coneDF)
    CGadata.obs = data.obs

    sc.pp.pca(CGadata, n_comps = 5)
    exp_var = CGadata.uns['pca']['variance_ratio']
    #print("Explained variance for " + str(i * 2)+" genes")
    #print(exp_var[0])
    
    #retreive pc1 exp variance
    pc1_list.append(exp_var[0])


exp_varDF = pd.DataFrame()
exp_varDF['N_genes'] = gene_num
exp_varDF['var_explained'] = pc1_list

#%%


print("")
print("****************")
print(exp_varDF)

#Here is our plotter

sns.set_theme(rc={'figure.figsize':(10,10)})
with sns.axes_style("white"):
    sns.scatterplot(data = exp_varDF,
                    x = 'N_genes',
                    y = 'var_explained',
                    color = 'black',
                    size = 1,
                    alpha = 0.5
                    )
plt.show()


#%%