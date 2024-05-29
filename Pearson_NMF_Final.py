#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import anndata as ad
from sklearn.decomposition import NMF

#Here we are doing basic processing for our Cone subset. 
# We will use this to find NMF patterns that are enriched for Rod Genes so that we can filter out falsely annotated cones. 

#first I need my data:
KHadata = sc.read_h5ad("kasiadata/ConesSubset.h5ad")

print("")
print("This subset has these dimensions (cell x gene)")
print(KHadata)
print("")

#%%

#basic settings, from scanpy tutorial
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")


#basic filtering
sc.pp.filter_cells(KHadata, min_genes=200)
#sc.pp.filter_cells(KHadata,min_counts=200)
sc.pp.filter_genes(KHadata, min_cells=5)
sc.pp.filter_genes(KHadata, min_counts=3)

# define outliers and further filtering. We have excluded the suggested MT filtering. 
#KHadata.obs['outlier_mt'] = KHadata.obs.pct_counts_mt > 5
#KHadata.obs['outlier_total'] = KHadata.obs["nCount_RNA"] > 15000
KHadata.obs['outlier_ngenes'] = KHadata.obs["n_genes"] > 5000

#This is for plotting outliers, see the code block above
#sns.violinplot(data = KHadata.obs["n_genes"])
#plt.show()

#print('%u cells with high %% of mitochondrial genes' % (sum(KHadata.obs['outlier_mt'])))
#print('%u cells with large total counts' % (sum(KHadata.obs['outlier_total'])))
print('%u cells with large number of genes' % (sum(KHadata.obs['outlier_ngenes'])))

#KHadata = KHadata[~KHadata.obs['outlier_mt'], :]
#KHadata = KHadata[~KHadata.obs['outlier_total'], :]
KHadata = KHadata[~KHadata.obs['outlier_ngenes'], :]

print(KHadata)


#%%
#applying Pearson residuals
sc.experimental.pp.highly_variable_genes(
    KHadata, flavor="pearson_residuals", n_top_genes=2000)


#applying gene selection, maintaining just hvgs
PRadata = KHadata[:, KHadata.var["highly_variable"]]

#normaizing gene selection by their residual score
sc.experimental.pp.normalize_pearson_residuals(PRadata)


hvgs = KHadata.var["highly_variable"]



######
# end pearson residual part, continuing to NMF
######
                 
#%%

#now we start the NMF
#convert X to DF, i dont think I do NMF within anndata
prDF = PRadata.to_df()
genenames = list(prDF.columns.values)

#Adjust if doing Pearson normalized, since NMF doesn't like negative numbers. whodathunkit. 
mins = prDF.min()
minval = abs(mins.min())
prDF = prDF + minval

#number of components we will use for NMF
comp = 20

#naming outputDF columns
colnam = []
count = 1
for i in range(0,comp):
    name = "NMF"
    name += str(count)
    colnam.append(name)
    count+= 1


#actual NMF
model = NMF(n_components=comp, init='random', random_state=0,
            max_iter=10000)
W = model.fit_transform(prDF)
H = model.components_

#below returns pattern weights. This lets you know what genes are driving your NMF patterns (NRL could be a rod pattern driver, opsins for cones, etc)
"""
weightsDF = pd.DataFrame(H, columns = genenames)
weightsDF.to_csv('PatternWeights.csv')
"""

nmfDF= pd.DataFrame(data = W, columns = colnam )
dr_adata = ad.AnnData(X = nmfDF)
sc.pp.neighbors(dr_adata, n_pcs = 0)

#%%

#clustering, optional, but good in practice. If you are clustering just cones, expect a gross overestimate of clusters. 
#comment out if you dont want the cluster analysis. 
sc.tl.leiden(dr_adata,resolution = 0.3, key_added= 'Leiden_Cluster')
sc.tl.umap(dr_adata,n_components=2)
sc.pl.umap(dr_adata)

for i in list(nmfDF.columns):
    KHadata.obs[i]= list(nmfDF[i].values)

KHadata.obsm['X_umap'] = dr_adata.obsm['X_umap']
KHadata.obs['Leiden'] = list(dr_adata.obs['Leiden_Cluster'].values)

#KHadata.write_h5ad("kasiadata/Photoreceptor_Subset.h5ad")

#%%
