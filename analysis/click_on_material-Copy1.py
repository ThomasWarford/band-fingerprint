#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install hdbscan')
# get_ipython().system('pip install pymatgen')


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

import sys
sys.path.append('..')
sys.path.append('../autoencoder')
# sys.path.append('/notebooks/Beta-VAE/')
from models import*

from src.band_plotters import*
from src.Tiff32Image import*
from src.TensorImageNoised import *

sys.path.append('/notebooks/band-fingerprint/autoencoder/resnet_autoencoder')
sys.path.append('/notebooks/band-fingerprint/src')

from model import *

from ae_misc import *


# In[3]:


FINGERPRINT_LENGTH = 98

#FINGERPRINT_NAME = "functional_10dpi_bernoulli_VAE_L={0}".format(FINGERPRINT_LENGTH)
FINGERPRINT_NAME = "224_2channel_resnet_L={0}".format(FINGERPRINT_LENGTH)
#FINGERPRINT_NAME = "all_k_branches_histogram_-8_to_8".format(FINGERPRINT_LENGTH)
#FINGERPRINT_NAME = "128x128_random_erase_resnet18_VAE_L={0}".format(FINGERPRINT_LENGTH)

PERPLEXITY = 30
FLAT_ONLY = False
BORING_COLUMNS = ["flat_segments", "flatness_score", "binary_flatness", "horz_flat_seg", "exfoliation_eg", "A", "B", "C", "D", "E", "F"]
INPUT_NAME = f"{FINGERPRINT_NAME}_perplexity_{PERPLEXITY}_length_{FINGERPRINT_LENGTH}.csv"


# ## Load Data

# In[4]:


df = pd.read_csv(f"../fingerprints/{INPUT_NAME}", index_col="ID")
if FLAT_ONLY:
    df = df[df.horz_flat_seg>0]
df.head()


# In[13]:


df[df.formula=="InCl3"]


# In[14]:



# ## Cluster

# In[5]:


fingerprint_cols = [str(i) for i in range(FINGERPRINT_LENGTH)]
BORING_COLUMNS += fingerprint_cols


# In[8]:


clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,\
                        gen_min_span_tree=False, leaf_size=40, metric='minkowski', cluster_selection_method='leaf', min_cluster_size=6, min_samples=2, p=0.2, cluster_selection_epsilon=0.0)
#clusterer.fit(30*np.tanh(df[fingerprint_cols])/30)
clusterer.fit(df[fingerprint_cols])


df["labels"] = clusterer.labels_


# In[9]:


#import sys
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
from ete3 import Tree, TreeStyle
print ('import_complete')


# In[10]:


###################################
####  HDBSCAN
MS=6
SS=2

        
# fname='HDBSCAN_244_DOS_minsize_'+str(min_size)+'_minsamp_'+str(min_samp)
# f = open(fname+'_summary_india.txt', 'w')
# f.write('Minkowski_metric_p=0.2\n')
# f.write('Total#sample   Min_size   Min_samples   N_clusters   N_noise  Max_persistence  Avg_persistence \
#         Max_Lambda_in_bar  Max_cluster_size  Silhouette  CH_measure DB_measure DBCV s-dbw \n')  

clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,\
                        gen_min_span_tree=False, leaf_size=3, metric='manhattan', cluster_selection_method='leaf', min_cluster_size=4, min_samples=2, p=0.2, cluster_selection_epsilon=0.0)
#clusterer.fit(30*np.tanh(df[fingerprint_cols])/30)
db = clusterer.fit(df[fingerprint_cols])
labels = db.labels_


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



# In[ ]:


from ete3 import Tree,TreeStyle,NodeStyle


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


