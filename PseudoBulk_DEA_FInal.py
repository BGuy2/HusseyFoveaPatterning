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
import scipy.io, scipy.sparse
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
import umap
import statistics as stat
#import scipy.stats 
import scProject as scP
from sklearn.mixture import GaussianMixture
import math
import decoupler as dc
import pydeseq2 as des
from pydeseq2.ds import DeseqStats


#we want to validate that the GMM model could find S and LM cones. the easiest way to do this is be pseudobulking everything that was labelled as S and LM and seeing what changes in expression.

path = "kasiadata/KHCones_GMM_100Genes160Thresh.h5ad"
KHadata = sc.read_h5ad(path)

#basic settings, from the tutorial, and from Pearson_NMF
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
#sc.settings.set_figure_params(dpi=80, facecolor="white")
#basic filtering
#KHadata.var_names_make_unique()
sc.pp.filter_cells(KHadata, min_genes=200)
#sc.pp.filter_cells(KHadata,min_counts=200)
sc.pp.filter_genes(KHadata, min_cells=5)
sc.pp.filter_genes(KHadata, min_counts=10)
# define outliers and further filtering. We have excluded the suggested MT filtering. 
#KHadata.obs['outlier_mt'] = KHadata.obs.pct_counts_mt > 5
#KHadata.obs['outlier_total'] = KHadata.obs["nCount_RNA"] > 15000
KHadata.obs['outlier_ngenes'] = KHadata.obs["n_genes"] > 5000


#print('%u cells with high %% of mitochondrial genes' % (sum(KHadata.obs['outlier_mt'])))
#print('%u cells with large total counts' % (sum(KHadata.obs['outlier_total'])))
print('%u cells with large number of genes' % (sum(KHadata.obs['outlier_ngenes'])))

#KHadata = KHadata[~KHadata.obs['outlier_mt'], :]
#KHadata = KHadata[~KHadata.obs['outlier_total'], :]
KHadata = KHadata[~KHadata.obs['outlier_ngenes'], :]


#%%

# scale and store results in layer
KHadata.layers["scaled"] = sc.pp.scale(KHadata, copy=True).X

#filter to genes we want from the roska dataset
#lets get the Roska data

#getting the Roska Cone genes
path = 'RoskaData/Roska_top500_Celltype.csv'
ros = pd.read_csv(path, index_col=False)
conegenes = pd.DataFrame(ros)
del conegenes['Unnamed: 0']

LM_Roska = list(conegenes['T3_10day'].values)
S_Roska = list(conegenes['T3_200day'].values)

ind = KHadata.var.index.to_numpy()
for i in LM_Roska:
    if i not in ind:
        LM_Roska.remove(i)
        
        
for i in S_Roska:
    if i not in ind:
        S_Roska.remove(i)
Roska_genes = LM_Roska + S_Roska

prune = ['SPRY4-AS1',
         'ATP5B', 
         'ATP5D', 
         'C14orf166']
Roska_genes = [i for i in Roska_genes if i not in prune]

#selecting only genes from the Roska dataset
KHadata = KHadata[:,Roska_genes]
print(KHadata)

#selecting only cones that made it past the threshold
KHadata_noInc = KHadata[KHadata.obs['ConeID'] != 'Inconclusive']

#%%

#here is just a plotter to check your work. Comment out if you just want to run the script. 

"""
counts = KHadata_noInc.to_df()
sns.set_theme(rc={'figure.figsize':(10,10)})
with sns.axes_style("white"):
    sns.scatterplot(data = KHadata_noInc.obs,
                    x = 'Norm_NMF_1',
                    y = 'Norm_NMF_2',
                    hue = counts['IGFBP7'],
                    #color = 'green',
                    size = 1,
                    alpha = 0.5
                    )
plt.show()
"""

#%%

#prepping our pseudobulk
pdata = dc.get_pseudobulk(
    KHadata_noInc,
    sample_col='ConeID',
    groups_col='treatment',
    #layer='counts',
    mode='sum',
    min_cells=0,
    min_counts=0
)

print(pdata.obs['treatment'])


#%%
#trying to do dea
#make the dseq object
inference = des.default_inference.DefaultInference(n_cpus=8)
dds = des.dds.DeseqDataSet(
    adata=pdata,
    design_factors=['treatment', 'ConeID'],
    #ref_level=['disease', 'normal'],
    refit_cooks=True,
    inference=inference,
)

#computing log fold change
dds.deseq2()
#%%
# Extract contrast between 10day and control
stat_res = des.ds.DeseqStats(
    dds,
    contrast=['ConeID', 'LM Cone', 'S Cone'],
    inference=inference,
)

# Compute Wald test
stat_res.summary()

print('uh we are done now')

#%%

# Extract results
results_df = stat_res.results_df

print(results_df)
results_df.to_csv("Pseudobulk.csv")

#%%
#plotting the volcano. isn't it nice?
with sns.axes_style("whitegrid"):
    dc.plot_volcano_df(
        results_df,
        x='log2FoldChange',
        y='padj',
        top=20,
        figsize=(8, 4),
        return_fig = True)

#%%

