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
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
import umap
import statistics as stat
import scipy.stats 
import scProject as scP
from sklearn.mixture import GaussianMixture
import math

#now we are going to pull the top n_genes genes between S and LM cones from the Roska Adult datset. Use the subsetter script to get just the cones from their data. 

path ='RoskaData/2D_Roska_PeripheralUMAP.h5ad'
data = sc.read_h5ad(path)

sc.tl.rank_genes_groups(data,
                        groupby='cell_type',
                        method = 'wilcoxon',
                        n_genes= 100,
                        )
# 100 genes from S and LM cones, we have done PCA on varying numbers of genes and it looks like 100 is a good balancing point between background noise and real subtype specific genes. 
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

print("Done Sorting")

listofgenes = lm_marks + s_marks

print(listofgenes)


#%%
sourceDF = data.to_df()
coneDF = pd.DataFrame()
for i in listofgenes:
    if i in sourceDF:
        coneDF[i] = list(sourceDF[i].values)

#%%
#YEAH alright we are gonna use NMF as our Dr just to do it. 
genenames = list(coneDF.columns.values)
#number of components we will use for NMF. We are using just 2 so we dont have to worry about losing info on UMAP projections, plus 2 components seem sufficient to separate S and LM cones in peripheral adult. 
comp = 2
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
W = model.fit_transform(coneDF)
H = model.components_

weightsDF = pd.DataFrame(H, columns = genenames)
nmfDF= pd.DataFrame(data = W, columns = colnam )
weightsAD = ad.AnnData(weightsDF)

#Now we have 2 components describing the adult S and LM cones. 
#%%
#lets get Kasias data
KHadata = sc.read_h5ad('kasiadata/ConeSubset_RodFiltered_StandardFilters.h5ad')
KH_orig = sc.read_h5ad('kasiadata/ConeSubset_RodFiltered_StandardFilters.h5ad')
print(KHadata)

sc.pp.log1p(KHadata) 
sc.pp.scale(KHadata)

kasiaDF = KHadata.var
inds = kasiaDF.index.to_numpy()
inds = list(inds)
KHadata.var['gene_id'] = inds

#%%
# Both datasets are prepared, now its time to project our dataset onto the Cowan 2020 dataset. We start with filtering out genes that arent shared between the datasets. 

dataset_filtered, patterns_filtered = scP.matcher.filterAnnDatas(KHadata, 
                                                                 weightsAD, 
                                                                 'gene_id')

#using matrix factorization for projection. 
KHdat = dataset_filtered.to_df()
KHarray = KHdat.to_numpy()
tPat = patterns_filtered.T
Patdat = tPat.to_df()
Patarr = Patdat.to_numpy()
transformed_array = np.matmul(KHarray,Patarr)

tfmDF = pd.DataFrame(transformed_array)

print(tfmDF)

#KHadata.obs['ConeScores'] = transformed_array
KHadata.obs['NMF_1'] = list(tfmDF[0].values)
KHadata.obs['NMF_2'] = list(tfmDF[1].values)
KH_orig.obs['NMF_1'] = list(tfmDF[0].values)
KH_orig.obs['NMF_2'] = list(tfmDF[1].values)

#At this point, we have projected the Hussey Cone dataset onto the two NMF components of the the Cowan dataset

#%%

#Alright, now we are looking to do another projection to see where the real cones are. 
#we will project the Roska data back onto its own NMF patterns as a sanity check. This is essentially a repeat of the previous block.

RoskaDF = data.var
inds = RoskaDF.index.to_numpy()
inds = list(inds)

data.var['gene_id'] = inds
dataset_filtered, patterns_filtered = scP.matcher.filterAnnDatas(data, 
                                                                 weightsAD, 
                                                                 'gene_id')

BRdat = dataset_filtered.to_df()
BRarray = BRdat.to_numpy()
tPat = patterns_filtered.T
Patdat = tPat.to_df()
Patarr = Patdat.to_numpy()

R_transformed_array = np.matmul(BRarray,Patarr)

tfmDF = pd.DataFrame(R_transformed_array)
data.obs['NMF_1'] = list(tfmDF[0].values)
data.obs['NMF_2'] = list(tfmDF[1].values)

#%%

# max normalizing after projection; It seems to better apporoximate how we view histological data, and models the biology well as seen in our pseudobulk. 
KH_Vals = list(KHadata.obs['NMF_1'].values)
KH_Vals2=list(KHadata.obs['NMF_2'].values)

BR_vals = list(data.obs['NMF_1'].values)
BR_Vals2 = list(data.obs['NMF_2'].values)

nKH_val1 = []
nKH_val2 = []
nBR_val1 = []
nBR_val2 = []

for i in KH_Vals:
    maxi = max(KH_Vals)
    new = float(i / maxi)
    nKH_val1.append(new)
for i in KH_Vals2:
    maxi = max(KH_Vals2)
    new = float(i / maxi)
    nKH_val2.append(new)
for i in BR_vals:
    maxi = max(BR_vals)
    new = float(i / maxi)
    nBR_val1.append(new)
for i in BR_Vals2:
    maxi = max(BR_Vals2)
    new = float(i / maxi)
    nBR_val2.append(new)

data.obs['Norm_NMF_1'] = nBR_val1 
data.obs['Norm_NMF_2'] = nBR_val2
KHadata.obs['Norm_NMF_1'] = nKH_val1 
KHadata.obs['Norm_NMF_2'] = nKH_val2
KH_orig.obs['Norm_NMF_1'] = nKH_val1 
KH_orig.obs['Norm_NMF_2'] = nKH_val2


#%%
#alright we have everything, lets get our data in order to do the GMM. We will use the GMM to classify the Hussey Cones as either S or LM.


RoskaDF = pd.DataFrame()
RoskaDF['NMF_1'] = data.obs['Norm_NMF_1']
RoskaDF['NMF_2'] = data.obs['Norm_NMF_2']
RoskaDF['Cell Type or Treatment'] = data.obs['cell_type']

kasiaDF = pd.DataFrame()
kasiaDF['NMF_1'] = KHadata.obs['Norm_NMF_1']
kasiaDF['NMF_2'] = KHadata.obs['Norm_NMF_2']
kasiaDF['Cell Type or Treatment'] = KHadata.obs['treatment']

#Here is the GMM
#first we pull out our NMF values
trainarr = RoskaDF[['NMF_1','NMF_2']].to_numpy()
testarr = kasiaDF[['NMF_1','NMF_2']].to_numpy()

#training the model on the Cowan/Roska dataset. 
model = GaussianMixture(n_components=2,random_state= 13).fit(trainarr)
labels = GaussianMixture(n_components=2, random_state= 13).fit_predict(trainarr)
train_scores = model.score_samples(trainarr)
RoskaDF['Raw_GMM_Labels'] = labels
RoskaDF['Log_Likelihood'] = train_scores

#predicting labels for our data
predicted_labels = model.predict(testarr)
kasiaDF['Raw_GMM_Labels'] = predicted_labels
scores =model.score_samples(testarr)
kasiaDF['Log_Likelihood']= scores

#boom, now we have the Hussey cones annotated. but we arent dont yet. 

#%%

# We get a log likelihood score that reflects our confidence of cluster assignment; we want to filter out cones that we are unsure of. Here is a threshold function. 

def thresh(self):
    log_scores = self['Log_Likelihood']
    thresh_bool = []
    
    for i in log_scores:
        if i > -160:
            thresh_bool.append(True)
        else:
            thresh_bool.append(False)
    
    orig_labels = self['Raw_GMM_Labels']
    thresh_labels = []
    count = 0
    for i in thresh_bool:
        if i == True:
            rename = orig_labels[count]
            if rename == 0:
                thresh_labels.append("S Cone")
            else:
                thresh_labels.append('LM Cone')
        else:
            thresh_labels.append("Inconclusive")
        count += 1

    self['GMM Labels'] = thresh_labels
    return self

kasiaDF = thresh(kasiaDF)
RoskaDF = thresh(RoskaDF)

#Renaming the Roska cones because we decided we are going with M/L name scheme within the paper. 
new_labels = []
for i in RoskaDF['GMM Labels']:
    if i == 'S Cone':
        new_labels.append("S Cone")
    elif i == 'LM Cone':
        new_labels.append('M/L Cone')
    else:
        new_labels.append('Inconclusive')

kh_labels = []
for i in kasiaDF['GMM Labels']:
    if i == 'S Cone':
        kh_labels.append("S Cone")
    elif i == 'LM Cone':
        kh_labels.append('M/L Cone')
    else:
        kh_labels.append('Inconclusive')
        

RoskaDF['GMM Labels'] = new_labels
RoskaDF['Cell Type'] = new_labels
kasiaDF['GMM Labels'] = kh_labels
check = pd.concat([RoskaDF,kasiaDF])

new_labels = []
for i in check['Cell Type or Treatment']:
    if i == 'S cone':
        new_labels.append("S Cone")
    elif i == 'L/M cone':
        new_labels.append('M/L Cone')
    elif i == '200day':
        new_labels.append(i)
    elif i =='Control':
        new_labels.append(i)
    else:
        new_labels.append(i)

check['Cell Type or Treatment'] = new_labels



print(check)
#_________________________________________________________________________________
#everything should be set, its all just graphing below. 

#Threshold filter for graphs is just below, if you want to see or hide the inconclusive cones in your plots 
"""kasiaDF = kasiaDF[kasiaDF['GMM Labels']!= 'Inconclusive']
RoskaDF = RoskaDF[RoskaDF['GMM Labels']!= 'Inconclusive']
check = check[check['GMM Labels']!= 'Inconclusive']"""

kasia_con = kasiaDF[kasiaDF['Cell Type or Treatment']=='Control']
kasia_10 = kasiaDF[kasiaDF['Cell Type or Treatment']=='10day']
kasia_200 = kasiaDF[kasiaDF['Cell Type or Treatment']=='200day']
roska_S =RoskaDF[RoskaDF['Cell Type or Treatment']=='S cone']
roska_LM =RoskaDF[RoskaDF['Cell Type or Treatment']=='M/L cone']
roska_pluscon = pd.concat([RoskaDF,kasia_con])
roska_plus10 = pd.concat([RoskaDF,kasia_10])
roska_plus200 = pd.concat([RoskaDF,kasia_200])



#%%

#here is a generic plotter script. Change the variable 'title' to any column header to change the plot hues, or edit as you see fit. 
from matplotlib import rcParams

title = "GMM Labels"

rcParams['figure.figsize'] = 8,8
fig, ax = plt.subplots()
with sns.axes_style("white"):
    sns.scatterplot(data = check,
                    x = 'NMF_1',
                    y = 'NMF_2',
                    hue = title,
                    #palette={"S Cone": "tab:cyan",  "M/L Cone":"tab:red", 'Control':'tab:blue','10day':'tab:pink','200day':'tab:green'},
                    palette = {"S Cone": "tab:cyan", "M/L Cone": "tab:red", "Inconclusive" : "tab:grey"},
                    #palette = {"S Cone": "tab:cyan", "M/L Cone": "tab:red"},
                    #hue = 'Conds',
                    #hue = 'Log_Likelihood',
                    #color = 'green',
                    s =25,
                    alpha = 0.5
                    )
ax.set_xlim(0,1.05)
ax.set_ylim(0,1.05)
ax.grid(False)
ax.legend(loc = 'upper left', title = title)
plt.show()



########

# The main script ends above. 
# Commit out the blocks below if you do not want to see relative proportions of cone subtypes. 

########
    #%%
"""
#here is a simple function to calculate relative abundance of each population determined by the GMM model. 
def perc_abundance(self):
    calls = self['GMM Labels']
    total = len(calls)
    LM_Count = 0
    S_Count = 0
    Inc_Count = 0
    for i in calls:
        if i == 'M/L Cone':
            LM_Count += 1
        elif i == 'S Cone':
            S_Count += 1
        else:
            Inc_Count += 1
    
    SLM_total = LM_Count + S_Count
    total_LM_Perc = (LM_Count / total) * 100
    LM_Perc_Called = (LM_Count / SLM_total) * 100
    total_S_Perc = (S_Count / total) * 100
    S_Perc_Called = ((S_Count / SLM_total) * 100)
    
    Inc_Perc = (Inc_Count / total) * 100
    
    print("Total Cones in Sample: " + str(total))
    print("Total Cones Failing Threshold: " + str(Inc_Count))
    print("Percent Cones Failing Threshold: " + str(Inc_Perc))
    print("Total Cones Passing Threshold: " + str(SLM_total))
    #print("Percent Cones Passing Threshold: "+ str(100 - Inc_Perc))
    print("")
    print("Of Passing Cones, %" + str(LM_Perc_Called) + " are LM")
    print("Of Passing Cones, %" + str(S_Perc_Called) + " are S")
    print("")
    print("Of total cones, %" +str(total_LM_Perc) + " are LM")
    print("Of total cones, %" +str(total_S_Perc) + " are S")
    
    retDic = {"Inconclusive":Inc_Perc,
              "M/L Cone": total_LM_Perc,
              "S Cone": total_S_Perc,
              }
    
    return(retDic)


print("Control")
conDic = perc_abundance(kasia_con)
#conDic["Sample"] = 'Control'
print(conDic)
print("------------")
print('10day')
day10dic = perc_abundance(kasia_10)
print(day10dic)
#day10dic["Sample"] = '10day'
print("------------")
print('200day')
day200dic = perc_abundance(kasia_200)
#day200dic["Sample"] = '200day'
print(day200dic)
allDics = [conDic,day10dic,day200dic]


abunDF = pd.DataFrame(allDics,index = ['Control', '10day','200day'])
print(abunDF)

abunDF.plot(kind='bar', stacked=True, color=['grey', 'red','cyan'])
plt.title('Cone Subtype Abundance %')
plt.xlabel('Treatment')
plt.ylabel('Percentage')
plt.grid(False)
plt.legend(loc = (1.04,0))"""

#%%


###############
# below we have an export step, so we can work from the pseudobulk. 
###############

KHadata.obs['ConeID'] = list(kasiaDF['GMM Labels'].values)
KH_orig.obs['ConeID'] = list(kasiaDF['GMM Labels'].values)

#throw out inconclusive
#KHadata = KHadata[KHadata.obs['ConeID'] != 'Inconclusive']

treats = list(KHadata.obs['treatment'].values)
coneID =list(KHadata.obs['ConeID'].values)

count = 0
treats_withID = []
for i in treats:
    treatname = str(i)
    conename = str(coneID[count])
    treat_cone = conename + "_" + treatname
    treats_withID.append(treat_cone)
    count += 1

KHadata.obs['ConeID_Treatment'] = treats_withID
KH_orig.obs['ConeID_Treatment'] = treats_withID

lm20_marks = []
s20_marks = []

for one, two in top20:
    lm20_marks.append(one)
    s20_marks.append(two)

treatlist = lm20_marks + s20_marks
treatlist.remove('MTND2P28')
treatlist.remove('OPN1LW')

#KH_orig.write_h5ad("kasiadata/KHCones_GMM_100Genes160Thresh.h5ad")




