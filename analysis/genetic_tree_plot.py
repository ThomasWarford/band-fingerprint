#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install hdbscan')
#get_ipython().system('pip install pymatgen')


# In[2]:


import hdbscan
import pandas as pd
import numpy as np
#%matplotlib ipympl
#%matplotlib notebook
import matplotlib
import matplotlib.pyplot as plt
from sklearn import manifold
from ipywidgets import interact, Output
from IPython.display import clear_output


from sklearn import manifold
from sklearn import decomposition
from sklearn import metrics
from functools import partial
import hdbscan
#from s_dbw import S_Dbw
#from internal_validation import internalIndex
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas
import networkx as nx
#import seaborn as sns
#from scipy.spatial.distance import euclidean
from matplotlib.colors import LinearSegmentedColormap
from ete3 import Tree, TreeStyle, NodeStyle

print ('import_complete')


FINGERPRINT_LENGTH = 60

#FINGERPRINT_NAME = "functional_10dpi_bernoulli_VAE_L={0}".format(FINGERPRINT_LENGTH)
#FINGERPRINT_NAME = "224_2channel_resnet_L={0}".format(FINGERPRINT_LENGTH)
FINGERPRINT_NAME = "all_k_branches_histogram_-8_to_8".format(FINGERPRINT_LENGTH)
#FINGERPRINT_NAME = "128x128_random_erase_resnet18_VAE_L={0}".format(FINGERPRINT_LENGTH)

PERPLEXITY = 30
FLAT_ONLY = True
BORING_COLUMNS = ["flat_segments", "flatness_score", "binary_flatness", "horz_flat_seg", "exfoliation_eg", "A", "B", "C", "D", "E", "F"]
INPUT_NAME = f"{FINGERPRINT_NAME}_perplexity_{PERPLEXITY}_length_{FINGERPRINT_LENGTH}.csv"


# ## Load Data


df = pd.read_csv(f"{INPUT_NAME}", index_col="ID")
if FLAT_ONLY:
    df = df[df.horz_flat_seg>0]
df.head()





# ## Cluster
###################################
####  HDBSCAN
MS=4
SS=3

fingerprint_cols = [str(i) for i in range(FINGERPRINT_LENGTH)]
BORING_COLUMNS += fingerprint_cols


# ML print
# clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,\
#                         gen_min_span_tree=False, leaf_size=40, metric='minkowski', cluster_selection_method='leaf', min_cluster_size=6, min_samples=2, p=0.2, cluster_selection_epsilon=0.0)
# #clusterer.fit(30*np.tanh(df[fingerprint_cols])/30)
# clusterer.fit(df[fingerprint_cols])

# DOS old print
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,\
                        gen_min_span_tree=True, leaf_size=40, metric='minkowski', cluster_selection_method='leaf', min_cluster_size=4, min_samples=3, p=0.2)




db = clusterer.fit(df[fingerprint_cols])
labels = db.labels_

df["labels"] = db.labels_
df["member_strength"] = db.probabilities_
print(len(df[df.labels==-1]))
#################
#### plot objects
cond_tree=db.condensed_tree_
plot_obj=cond_tree.get_plot_data()
#single_link_tree=db.single_linkage_tree_


#########################################
############## Colormap
##############################
import matplotlib
cmap = plt.cm.get_cmap('turbo')
norm = matplotlib.colors.Normalize(vmin=min(labels), vmax=max(labels))

##################################
###### Pandas data
##################################
panda_data=cond_tree.to_pandas()
#print(G.number_of_nodes())
#print(panda_data)
selected_clusters=cond_tree._select_clusters()
G1 = panda_data[panda_data['child_size'] > 1]
#New_Nx=nx.from_pandas_edgelist(G1,'parent','child',['lambda_val', 'child_size'])
#nx.write_edgelist(New_Nx,'New_edgelist', encoding = 'latin-1')

len_G1=[]
cluster_id=[]
for ind1 in G1.index:
    len_G1.append(0.1)
    if G1.at[ind1,'child'] in selected_clusters:
        cluster_id.append(str(selected_clusters.index(G1.at[ind1,'child'])))
    else:
        cluster_id.append('-1')
print(cluster_id)
G1.insert(4, 'dist_G1', len_G1)
G1.insert(5, 'cluster_id', cluster_id)
G2=G1.copy()
print(G2)
del G1['cluster_id']
del G1['lambda_val']
del G1['child_size']
g1_list=G1.values.tolist()

# In[ ]:



##############################
################ ETE treee from parent child relations
tree = Tree.from_parent_child_table(g1_list)
#tree.write(format=9,outfile='new_tree.nw')
print(G2)
for node in tree.traverse():
    nstyle = NodeStyle()
    if node.is_leaf():
        index1=G2.index[G2['child'] == int(node.name)]
        node.name=G2.at[index1[0],'cluster_id']
        #nstyle = NodeStyle()
        #print(int(node.name))
        #print(matplotlib.colors.rgb2hex(cmap(norm(int(node.name)))))
        nstyle["fgcolor"] = str(matplotlib.colors.rgb2hex(cmap(norm(int(node.name)))))
        #nstyle['fgcolor']='#FF0000'
        nstyle["size"] = G2.at[index1[0],'child_size']/2
    else:
        nstyle["fgcolor"] ='black'
    node.set_style(nstyle)
tree.write(format=1,outfile='new_tree.nw')
#################################
################### Plot
ts = TreeStyle()
ts.mode='c'
ts.arc_start = -180 # 0 degrees = 3 o'clock
ts.arc_span = 360
ts.scale = 40
ts.show_leaf_name=True
tree.show(tree_style=ts)


