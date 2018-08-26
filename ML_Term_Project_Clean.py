# coding: utf-8

# # Library Imports

from time import time
import datetime
import random
from collections import Counter

from IPython.display import Image

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.options.display.max_rows = 999

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#connects JS to notebook so plots work inline
init_notebook_mode(connected=True)
#allow offline use of cufflinks
cf.go_offline()

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.cluster import silhouette_score as sklearn_silhouette_score
from sklearn.metrics.cluster import calinski_harabaz_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE, ADASYN

from pyts.transformation import PAA

from tslearn.utils import ts_size
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.metrics import dtw, soft_dtw
from tslearn.barycenters import DTWBarycenterAveraging, SoftDTWBarycenter
from tslearn.clustering import TimeSeriesScalerMeanVariance
from tslearn.clustering import silhouette_score as tslearn_silhouette_score
from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict

import keras.backend as K
from keras import optimizers
from keras.optimizers import rmsprop
from keras import losses
from keras import metrics
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# # Data Import & Preprocessing

#read data from csv
df = pd.read_csv('./Data/Sensor_Data_8_17-Onward.csv')

#shape of the dataframe
df.shape

#example row
df.head(1)

#set 'time' column as index
df.set_index(['time'], inplace=True)

#convert to datetime index
df.index = pd.to_datetime(df.index)

#subset into dataframes with photo sensor, ir, and all sensor data
df_ps = df[['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13']]

df_ir = df[['sky1','amb1','sky2','amb2']]

df_psir = df[['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','sky1','amb1','sky2','amb2']]

#create dataframe lists whose items have the same year, month, and day
DF_PS_List = [ group[1] for group in df_ps.groupby([df_ps.index.year,df_ps.index.month,df_ps.index.day]) ]
DF_IR_List = [ group[1] for group in df_ir.groupby([df_ir.index.year,df_ir.index.month,df_ir.index.day]) ]
DF_PSIR_List = [ group[1] for group in df_psir.groupby([df_psir.index.year,df_psir.index.month,df_psir.index.day]) ]

#first row of example dataframe in the list of photo sensor dataframes
DF_PS_List[0].head(1)

#for all columns of all data frames, convert all Lux values to W/m^2
for i in range(len(DF_PS_List)):
    for j in range(len(DF_PS_List[0].columns)):
        DF_PS_List[i].iloc[:, j] = DF_PS_List[i].iloc[:, j]/126.5736

#for all data frames, create a column holding the maximum value across all columns for each row
for i in range(len(DF_PS_List)):
    DF_PS_List[i]['MAX'] = DF_PS_List[i][['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10',
                                         'd11','d12','d13']].max(axis=1)

#for all data frames, subset only rows whose MAX value is > 1
DF_PS_List_Day = [ data_frame[data_frame['MAX'] > 1] for data_frame in DF_PS_List]

#across all dataframes, find minimum length of the first dimension
min_list = []
for i in range(len(DF_PS_List_Day)-1):
    min_list.append(DF_PS_List_Day[i+1].shape[0])
print(min(min_list))

#across all dataframes, find maximum length of the first dimension
max_list = []
for i in range(len(DF_PS_List_Day)-1):
    max_list.append(DF_PS_List_Day[i+1].shape[0])
print(max(max_list))

#across all dataframes, find maximum length of the first dimension
mean_list = []
for i in range(len(DF_PS_List_Day)-1):
    mean_list.append(DF_PS_List_Day[i+1].shape[0])
print(np.mean(mean_list))

#create list of 1D arrays of maximum photo sensor values (excluding the first, which is empty after subsetting for values > 1)
all_max_list = []
for i in range(len(DF_PS_List_Day[:])-1):
    all_max_list.append(DF_PS_List_Day[i+1].iloc[:,-1].values.reshape(-1,1))

# ### On Piecewise Aggregate Approximation, Cf. http://www.cs.ucr.edu/~eamonn/kais_2000.pdf

#for each day's 1D array, scale all values, transpose, Piecewise-Aggregate Approximation (PAA) transform, 
#      and flatten to output a list of series
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

from pyts.transformation import PAA
paa = PAA(window_size=None, output_size=514, overlapping=True)

TS_List = [pd.Series(paa.transform(scaler.fit_transform(ts).T).flatten()) for ts in all_max_list]

#concatenate series in the list into a single dataframe with dimensions (# of data points) x (# of days)
points_x_days = pd.concat(TS_List, axis=1)
print(points_x_days.shape)

#transpose dataframe, whose resulting dimensions are (# of days) x (# of data points)
days_x_points = points_x_days.T
print(days_x_points.shape)

# # Visualizing and Hand-Labeling Classes using Domain Knowledge
# 
# #### 0 - sunny
# #### 1 - cloudy
# #### 2 - partially cloudy
# #### 3 - mixed sunny and partially cloudy
# #### 4 - sunny with occlusion
# #### 5 - partially cloudy with occlusion
# #### 6 - cloudy with sunspike

#visualize values for all days in dataframe
for i in range(len(points_x_days.columns)):
    plt.plot(points_x_days.iloc[:, i])
    plt.show()
    plt.close()

#array of hand-labeled classes
y_act = np.array([3,3,2,1,0,0,1,2,2,3,3,0,2,3,2,3,5,2,1,1,2,5,0,0,0,0,2,3,0,5,2,0,0,3,0,3,3,2,2,2,3,1,3,2,1,1
          ,1,4,4,4,4,3,3,2,1,2,2,4,3,1,5,4,2,5,2,2,2,2,6,2,2,5,4,3,2,2,5,6,4,1,2,5,4,3,4,4,4,5,5,4,4
          ,6,4,5,1,5,1,6,4,2,1,4,5,2,6,5,2,2,2,2,1,2,4,2,2,2,2,6,4,4,3,2,6,4,4,3,2,4,4,4,1,4,4,2
          ,2,5,4,4,2,2,4,2,4,5,2,6,5,4,4,3,2,6,6,4,2,1,5,2,4,2,2,5,3,2,1,6,5,2,6,2,3])

#counts of all classes in the data set
y = np.bincount(y_act)
ii = np.nonzero(y)[0]
np.vstack((ii,y[ii])).T

# # Exploratory Data Analysis
# ## DTW K-Means Clustering with LB-Keogh Bounding 
# ### On Dynamic Time-Warping (DTW) and LB-Keogh Bounding, Cf. http://www.cs.ucr.edu/~eamonn/KAIS_2004_warping.pdf
# ### On Soft Dynamic Time-Warping (Soft-DTW), Cf. https://arxiv.org/pdf/1703.01541.pdf
# #### Cf. also https://github.com/alexminnaar/time-series-classification-and-clustering/blob/master/clustering/ts_cluster.py

#create ts_cluster class object
class ts_cluster(object):
    def __init__(self,num_clust):
        '''
        num_clust is the number of clusters for the k-means algorithm
        assignments holds the assignments of data points (indices) to clusters
        centroids holds the centroids of the clusters
        '''
        self.num_clust=num_clust
        self.assignments={}
        self.centroids=[]

    def k_means_clust(self,data,num_iter,w,progress=False):
        '''
        k-means clustering algorithm for time series data.  dynamic time warping Euclidean distance
         used as default similarity measure. 
        '''
        self.centroids=random.sample(data,self.num_clust)

        for n in range(num_iter):
            if progress:
                print('iteration '+str(n+1))
            #assign data points to clusters
            self.assignments={}
            for ind,i in enumerate(data):
                min_dist=float('inf')
                closest_clust=None
                for c_ind,j in enumerate(self.centroids):
                    if self.LB_Keogh(i,j,5)<min_dist:
                        cur_dist=self.DTWDistance(i,j,w)
                        if cur_dist<min_dist:
                            min_dist=cur_dist
                            closest_clust=c_ind
                if closest_clust in self.assignments:
                    self.assignments[closest_clust].append(ind)
                else:
                    self.assignments[closest_clust]=[]

            #recalculate centroids of clusters
            for key in self.assignments:
                clust_sum=0
                for k in self.assignments[key]:
                    clust_sum=clust_sum+data[k]
                self.centroids[key]=[m/len(self.assignments[key]) for m in clust_sum]


    def get_centroids(self):
        return self.centroids

    def get_assignments(self):
        return self.assignments

    def plot_centroids(self):
        for i in self.centroids:
            plt.plot(i)
        plt.show()

    def DTWDistance(self,s1,s2,w=None):
        '''
        Calculates dynamic time warping Euclidean distance between two
        sequences. Option to enforce locality constraint for window w.
        '''
        DTW={}

        if w:
            w = max(w, abs(len(s1)-len(s2)))

            for i in range(-1,len(s1)):
                for j in range(-1,len(s2)):
                    DTW[(i, j)] = float('inf')

        else:
            for i in range(len(s1)):
                DTW[(i, -1)] = float('inf')
            for i in range(len(s2)):
                DTW[(-1, i)] = float('inf')

        DTW[(-1, -1)] = 0

        for i in range(len(s1)):
            if w:
                for j in range(max(0, i-w), min(len(s2), i+w)):
                    dist= (s1[i]-s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            else:
                for j in range(len(s2)):
                    dist= (s1[i]-s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

        return np.sqrt(DTW[len(s1)-1, len(s2)-1])

    def LB_Keogh(self,s1,s2,r):
        '''
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        '''
        LB_sum=0
        for ind,i in enumerate(s1):

            lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

            if i>upper_bound:
                LB_sum=LB_sum+(i-upper_bound)**2
            elif i<lower_bound:
                LB_sum=LB_sum+(i-lower_bound)**2

        return np.sqrt(LB_sum)

#instantiate class object
ts_c = ts_cluster(num_clust=7)

#perform clustering using 7 clusters, three iterations, a window of size 2, and enabling the model progression argument
ts_c.k_means_clust(list(days_x_points.values),3,2,1)

# ## Visualize the centroids identified by DTW K-Means clustering with LB-Keogh bounding

for i in ts_c.centroids:
    plt.plot(i)
    plt.show()

# ## Interactive plots of centroid radiation profiles

for i in range(len(ts_c.centroids)):
    pd.DataFrame(ts_c.centroids).iloc[i].iplot()

# ## Visualizing Barycenter Averages
# ### On DTW Barycenter Averaging, Cf. http://lig-membres.imag.fr/bisson/cours/M2INFO-AIW-ML/papers/PetitJean11.pdf

#convert dataframe to array of values
days_x_points_vals = days_x_points.values

#standardize the data
days_x_points_vals_trans = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(days_x_points_vals)

#instantiate DTW and Soft-DTW Barycenter Averaging classes
dtw_bar = DTWBarycenterAveraging()
soft_dtw_bar = SoftDTWBarycenter()

# #### Static Plots

#Visualize Barycenters
dtw_barycenters = []
soft_dtw_barycenters = []
for i in range(24):
    dtw_barycenters.append(dtw_bar.fit(days_x_points_vals_trans[7*i:7*i+1,:,:]))
    soft_dtw_barycenters.append(soft_dtw_bar.fit(days_x_points_vals_trans[7*i:7*i+1,:,:]))
    pd.DataFrame(dtw_barycenters[i].T[0]).plot()
    pd.DataFrame(soft_dtw_barycenters[i].T[0]).plot()

# #### Interactive Plots

#Visualize Barycenters
dtw_barycenters = []
soft_dtw_barycenters = []
for i in range(24):
    dtw_barycenters.append(dtw_bar.fit(days_x_points_vals_trans[7*i:7*i+1,:,:]))
    soft_dtw_barycenters.append(soft_dtw_bar.fit(days_x_points_vals_trans[7*i:7*i+1,:,:]))
    pd.DataFrame(dtw_barycenters[i].T[0]).iplot()
    pd.DataFrame(soft_dtw_barycenters[i].T[0]).iplot()

# # Unsupervised Model 1: K-Means++ Clustering

#convert dataframe to array of values
days_x_points_vals = days_x_points.values
print(days_x_points_vals.shape)
print(y_act.shape)

#instantiate Kmeans++ clustering object
k_means = KMeans(n_clusters=7, init='k-means++', n_init=1000, max_iter=1000, random_state=222, n_jobs=4)

#fit and predict using the array of values
y_pred_kmeans = k_means.fit_predict(days_x_points_vals)

#cluster assignments for each row (i.e., each day) in the array
y_pred_kmeans

# ### Interactive visualization of centroids identified by K-Means++ clustering

for i in range(len(k_means.cluster_centers_)):    
    pd.DataFrame(k_means.cluster_centers_).iloc[i].iplot()

# #### Static plots

for i in range(len(k_means.cluster_centers_)):    
    pd.DataFrame(k_means.cluster_centers_).iloc[i].plot()
    plt.show()

# ### Model evaluation of K-means++ Clustering

#inertia score (i.e., sum of the distances of samples to their closest cluster center)
print('The inertia score of the above modeling procedure is: \n')
print(k_means.inertia_)

# ### On the Silhouette Score, Cf. http://svn.donarmstrong.com/don/trunk/projects/research/papers_to_read/statistics/silhouettes_a_graphical_aid_to_the_interpretation_and_validation_of_cluster_analysis_rousseeuw_j_comp_app_math_20_53_1987.pdf

print('Silhouette Score using Euclidean metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, y_pred_kmeans, metric='euclidean'), '\n')
print('Silhouette Score using Cosine metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, y_pred_kmeans, metric='cosine'), '\n')
print('Silhouette Score using Minkowski metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, y_pred_kmeans, metric='minkowski'), '\n')

print('Calinski-Harabaz Score is: \n')
print(calinski_harabaz_score(days_x_points_vals, y_pred_kmeans))

print('Silhouette Score of DTW K-Means++ clustering, using DTW metric is: \n')
print(tslearn_silhouette_score(days_x_points_vals, y_pred_kmeans, metric=dtw))

print('Silhouette Score of Soft-DTW K-Means++ clustering, using Soft-DTW metric is: \n')
print(tslearn_silhouette_score(days_x_points_vals, y_pred_kmeans, metric=soft_dtw))

# # Supervised Model 1: KNN Classification using Hand-Labeled Classes

#crreate X and y arrays for splitting into training and testing sets
X = days_x_points_vals
y = y_act

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# ### Elbow method for choosing the best K Value

error_rate = []

for i in range(1,7):
    #iterate through k-values
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance',algorithm='ball_tree')
    #model fitting for every k-value
    knn.fit(X_train,y_train)
    #test set predictions for given k-value
    pred_i = knn.predict(X_test)
    #append error rates for each k to list
    error_rate.append(np.mean(pred_i != y_test))

#plot K-value vs. error rates
plt.figure(figsize=(10,6))
plt.plot(range(1,7),error_rate,color='blue',linestyle='--',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K-Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# ### Model Fitting and Evaluation of KNN Classification

#instantiate KNN classifier object, using the optimal number of neighbors shown above
knn = KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='ball_tree', metric='minkowski')

#fit KNN model to training data
knn.fit(X_train,y_train)

#predict method to generate predictions from KNN model and test data
pred = knn.predict(X_test)

#create confusion matrix and classification report
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

#classification accuracy score
print("Correct classification rate:", accuracy_score(y_test,pred))

#Visualize confusion matrix as a heatmap
sns.set(font_scale=3)
conf_matrix = confusion_matrix(y_test,pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
plt.title('Confusion matrix', fontsize=20)
plt.ylabel('True label', fontsize=15)
plt.xlabel('Clustering label', fontsize=15)
plt.show()

print('Adjusted Mutual Information Score is: \n')
print(adjusted_mutual_info_score(y_test,pred), '\n')
print('Adjusted Rand Score is: \n')
print(adjusted_rand_score(y_test,pred), '\n')
print('Normalized Mutual Information Score is: \n')
print(normalized_mutual_info_score(y_test,pred), '\n')
print('Homogeneity, Completeness, and V_Measure Scores are: \n')
sklearn.metrics.homogeneity_completeness_v_measure(y_test,pred)

# # Unsupervised Model 2: K-Shape Time Series Clustering
# ### On K-Shape clustering, Cf. http://www1.cs.columbia.edu/~jopa/Papers/PaparrizosSIGMOD2015.pdf

# ### K-Shape Clustering using the DTW metric

#standardize the data for use with KShape clustering
days_x_points_vals_trans = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(days_x_points_vals)

#shape of the array holding the resulting standardized values
print(days_x_points_vals_trans.shape)
print(days_x_points_vals_trans[:,:,0].shape)

#fit the standardize values using KShape clustering
ks = KShape(n_clusters=7, max_iter=50, n_init=50, random_state=222).fit(days_x_points_vals_trans)

#KShape clustering assignments
print(ks.labels_)

# ### Interactive visualization of centroids identified by K-Shape clustering

for i in range(len(ks.cluster_centers_[:,:,0])):    
    pd.DataFrame(ks.cluster_centers_[:,:,0]).iloc[i].iplot()

# #### Static plots

for i in range(len(ks.cluster_centers_[:,:,0])):    
    pd.DataFrame(ks.cluster_centers_[:,:,0]).iloc[i].plot()
    plt.show()

# ### Model evaluation of K-Shape clustering

#inertia score (i.e., sum of the distances of samples to their closest cluster center)
print('The inertia score of the above modeling procedure is: \n')
print(ks.inertia_)

print('Silhouette Score of K-Shape clustering, using Euclidean metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, ks.labels_, metric='euclidean'), '\n')
print('Silhouette Score of K-Shape clustering, using Cosine metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, ks.labels_, metric='cosine'), '\n')
print('Silhouette Score of K-Shape clustering, using Minkowski metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, ks.labels_, metric='minkowski'), '\n')

print('Calinski-Harabaz Score of K-Shape clustering is: \n')
print(calinski_harabaz_score(days_x_points_vals, ks.labels_))

print('Silhouette Score of K-Shape clustering, using the DTW metric is: \n')
print(tslearn_silhouette_score(days_x_points_vals_trans, ks.labels_, metric=dtw))

print('Silhouette Score of K-Shape clustering, using the Soft-DTW metric is: \n')
print(tslearn_silhouette_score(days_x_points_vals_trans, ks.labels_, metric=soft_dtw))

# # Unsupervised Model 3: Time Series K-Means

# ### Time Series K-Means using the DTW Metric

#fitting to the data using the specially designed DTW Time Series KMeans class 
ts_km = TimeSeriesKMeans(n_clusters=7, metric='dtw',max_iter=100,verbose=True,random_state=222).fit(days_x_points_vals)

# ### Interactive visualization of centroids identified by DTW Time Series K-Means

for i in range(len(ts_km.cluster_centers_)):    
    pd.DataFrame(ts_km.cluster_centers_[i]).iplot()

# #### Static plots

for i in range(len(ts_km.cluster_centers_)):    
    pd.DataFrame(ts_km.cluster_centers_[i]).plot()
    plt.show()

# ### Model evaluation of DTW Time Series K-Means clustering

#inertia score (i.e., sum of the distances of samples to their closest cluster center)
print('The inertia score of the above modeling procedure is: \n')
print(ts_km.inertia_)

print('Silhouette Score of DTW KMeans clustering, using Euclidean metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, ts_km.labels_, metric='euclidean'), '\n')
print('Silhouette Score of DTW KMeans clustering, using Cosine metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, ts_km.labels_, metric='cosine'), '\n')
print('Silhouette Score of DTW KMeans clustering, using Minkowski metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, ts_km.labels_, metric='minkowski'), '\n')

print('Calinski-Harabaz Score of DTW KMeans clustering is: \n')
print(calinski_harabaz_score(days_x_points_vals, ts_km.labels_))

print('Silhouette Score of DTW KMeans clustering, using DTW metric is: \n')
print(tslearn_silhouette_score(days_x_points_vals, ts_km.labels_, metric=dtw))

# ### Time Series K-Means using the Soft-DTW Metric

#fitting using Soft-DTW Time Series KMeans clustering
ts_km_dtw = TimeSeriesKMeans(n_clusters=7, metric='softdtw',max_iter=100,verbose=True,random_state=222).fit(days_x_points_vals)

# ### Interactive visualization of centroids identified by Soft-DTW Time Series K-Means

for i in range(len(ts_km_dtw.cluster_centers_)):    
    pd.DataFrame(ts_km_dtw.cluster_centers_[i]).iplot()

# #### Static plots

for i in range(len(ts_km_dtw.cluster_centers_)):    
    pd.DataFrame(ts_km_dtw.cluster_centers_[i]).plot()
    plt.show()

# ### Model evaluation of Soft-DTW Time Series K-Means clustering

#inertia score (i.e., sum of the distances of samples to their closest cluster center)
print('The inertia score of the above modeling procedure is: \n')
print(ts_km_dtw.inertia_)

print('Silhouette Score of Soft-DTW Time Series KMeans clustering, using Euclidean metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, ts_km_dtw.labels_, metric='euclidean'), '\n')
print('Silhouette Score Soft-DTW Time Series KMeans clustering, using Cosine metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, ts_km_dtw.labels_, metric='cosine'), '\n')
print('Silhouette Score Soft-DTW Time Series KMeans clustering, using Minkowski metric is: \n')
print(sklearn_silhouette_score(days_x_points_vals, ts_km_dtw.labels_, metric='minkowski'), '\n')

print('Calinski-Harabaz Score of Soft-DTW Time Series KMeans clustering is: \n')
print(calinski_harabaz_score(days_x_points_vals, ts_km_dtw.labels_))

print('Silhouette Score of Soft-DTW Time Series KMeans clustering, using Soft-DTW metric is: \n')
print(tslearn_silhouette_score(days_x_points_vals, ts_km_dtw.labels_, metric=tslearn.metrics.soft_dtw))

# # Supervised Model 2: K-Neighbors Time Series Classifier

# ## DTW K-Neighbors Time Series Classification

#split the data into train and test sets
X_train_knts, X_test_knts, y_train_knts, y_test_knts = train_test_split(days_x_points_vals_trans, y_act, 
                                                                        test_size=0.25, random_state=222)
																		
error_rate = []

for i in range(1,7):
    #iterate through k-values
    knn_ts = KNeighborsTimeSeriesClassifier(n_neighbors=i, metric="dtw")
    #model fitting for every k-value
    knn_ts.fit(X_train_knts, y_train_knts)
    #test set predictions for given k-value
    pred_i = knn_ts.predict(X_test_knts)
    #append error rates for each k to list
    error_rate.append(np.mean(pred_i != y_test_knts))

#plot K-value vs. error rates
plt.figure(figsize=(10,6))
plt.plot(range(1,7),error_rate,color='blue',linestyle='--',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K-Value')
plt.xlabel('K')
plt.ylabel('Error Rate')																		

# ### Model fitting of DTW K-Neighbors Time Series Classifier

#instantiate the classifier, and fit using the DTW metric
kn_ts_clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, metric="dtw")
kn_ts_clf.fit(X_train_knts, y_train_knts)

#generate model predictions using the test data
kn_ts_clf_pred = kn_ts_clf.predict(X_test_knts)

#create confusion matrix and classification report
print(confusion_matrix(y_test_knts, kn_ts_clf_pred))
print('\n')
print(classification_report(y_test_knts, kn_ts_clf_pred))

#classification accuracy score
print("Correct classification rate:", accuracy_score(y_test_knts, kn_ts_clf_pred))

# Visualize confusion matrix as a heatmap
sns.set(font_scale=3)
conf_matrix = confusion_matrix(y_test_knts, kn_ts_clf_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
plt.title('Confusion matrix', fontsize=20)
plt.ylabel('True label', fontsize=15)
plt.xlabel('Clustering label', fontsize=15)
plt.show()

# ### Model Evaluation of DTW K-Neighbors Time Series Classification

print('Adjusted Mutual Information Score is: \n')
print(adjusted_mutual_info_score(y_test_knts,kn_ts_clf_pred), '\n')
print('Adjusted Rand Score is: \n')
print(adjusted_rand_score(y_test_knts,kn_ts_clf_pred), '\n')
print('Normalized Mutual Information Score is: \n')
print(normalized_mutual_info_score(y_test_knts,kn_ts_clf_pred), '\n')
print('Homogeneity, Completeness, and V_Measure Scores are: \n')
sklearn.metrics.homogeneity_completeness_v_measure(y_test_knts,kn_ts_clf_pred)

# # Supervised Model 3: Shapelet Modeling Classifier
# ### On Shapelet Modeling, Cf. https://www.ismll.uni-hildesheim.de/pub/pdfs/grabocka2014e-kdd.pdf

#create X and y arrays for splitting into training and testing sets
X_shapelet = days_x_points_vals
y_shapelet = y_act

#randomized split into training and testing sets
X_train_shapelet, X_test_shapelet, y_train_shapelet, y_test_shapelet = train_test_split(X_shapelet, y_shapelet, test_size=0.3, random_state=2)

#time series min-max scaling of feature data
X_train_shapelet = TimeSeriesScalerMinMax().fit_transform(X_train_shapelet)
X_test_shapelet = TimeSeriesScalerMinMax().fit_transform(X_test_shapelet)

#create r=7 different shapelet lengths, using 14% of the length of time series for use as the base shapelet length
#NB: l * r must be < 1.0
shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=X_train_shapelet.shape[0],
                                                       ts_sz=X_train_shapelet.shape[1],
                                                       n_classes=len(set(y_train_shapelet)),
                                                       l=0.14,
                                                       r=7)

#instantiate shapelet modeling classifier, with learning_rate=0.05
np.random.seed(6)
shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        optimizer=rmsprop(lr=.05),
#                        weight_regularizer=.01,
                        max_iter=300,
                        verbose_level=1)

#fit the shapelet model
shp_clf.fit(X_train_shapelet, y_train_shapelet)

#predict labels using the fitted shapelet modeling classifier
predicted_labels = shp_clf.predict(X_test_shapelet)

# ### Model Evaluation for Shapelet Modeling Classifier

#classification accuracy score
print("Correct classification rate:", accuracy_score(y_test_shapelet, predicted_labels))

#create confusion matrix and classification report
print(confusion_matrix(y_test_shapelet, predicted_labels))
print('\n')
print(classification_report(y_test_shapelet, predicted_labels))

#visualize confusion matrix with a heatmap
sns.set(font_scale=3)
conf_matrix = confusion_matrix(y_test_shapelet, predicted_labels)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
plt.title('Confusion matrix', fontsize=20)
plt.ylabel('True label', fontsize=15)
plt.xlabel('Clustering label', fontsize=15)
plt.show()

print('Adjusted Mutual Information Score is: \n')
print(adjusted_mutual_info_score(y_test_shapelet, predicted_labels), '\n')
print('Adjusted Rand Score is: \n')
print(adjusted_rand_score(y_test_shapelet, predicted_labels), '\n')
print('Normalized Mutual Information Score is: \n')
print(normalized_mutual_info_score(y_test_shapelet, predicted_labels), '\n')
print('Homogeneity, Completeness, and V_Measure Scores are: \n')
print(sklearn.metrics.homogeneity_completeness_v_measure(y_test_shapelet, predicted_labels))

## visualize shapelets identified by the classifier
fig = plt.figure(figsize=(20,16))
for i, sz in enumerate(shapelet_sizes.keys()):
    ax = plt.subplot(len(shapelet_sizes), 1, i + 1)
    ax.set_title("%d shapelets of size %d" % (shapelet_sizes[sz], sz), fontsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for shp in shp_clf.shapelets_:
        if ts_size(shp) == sz:
            plt.plot(shp.ravel());
plt.xlim([0, max(shapelet_sizes.keys()) - 1])
plt.tight_layout()
plt.show()

# ## Applying Over-Sampling to Under-represented Classes

# ### Using the Synthetic Minority Over-Sampling Technique (SMOTE)
# ### On SMOTE over-sampling, Cf. https://arxiv.org/pdf/1106.1813.pdf
# #### Cf. also http://sci2s.ugr.es/keel/keel-dataset/pdfs/2005-Han-LNCS.pdf and
# #### https://pdfs.semanticscholar.org/f11b/4f012d704f757599bd7881c278dcc6816c7c.pdf

#apply SMOTE balancing to the training set and print number of records associated with each label
X_shapelet_resamp_train_SMOTE, y_shapelet_resamp_train_SMOTE = SMOTE().fit_sample(X_train_shapelet[:,:,0], y_train_shapelet)
print(sorted(Counter(y_shapelet_resamp_train_SMOTE).items()))

#reshape the features of the training set for use with the shapelet classifier, extending the third axis
X_shapelet_resamp_train_SMOTE_reshaped = np.expand_dims(X_shapelet_resamp_train_SMOTE, axis=2)

#create r=7 different shapelet lengths, using 14% of the length of time series for use as the base shapelet length
shapelet_sizes_SMOTE = grabocka_params_to_shapelet_size_dict(n_ts=X_shapelet_resamp_train_SMOTE_reshaped.shape[0],
                                                       ts_sz=X_shapelet_resamp_train_SMOTE_reshaped.shape[1],
                                                       n_classes=len(set(y_shapelet_resamp_train_SMOTE)),
                                                       l=0.14,
                                                       r=7)

#instantiate the SMOTE balanced shapelet classifier, with learning_rate=0.05
np.random.seed(6)
shp_clf_SMOTE = ShapeletModel(n_shapelets_per_size=shapelet_sizes_SMOTE,
                        optimizer=rmsprop(lr=.05),
#                       weight_regularizer=.01,
                        max_iter=300,
                        verbose_level=1)

#fit the SMOTE balanced shapelet model
shp_clf_SMOTE.fit(X_shapelet_resamp_train_SMOTE_reshaped, y_shapelet_resamp_train_SMOTE)

#predict labels using the fitted SMOTE balanced shapelet modeling classifier
predicted_labels_imb_SMOTE = shp_clf_SMOTE.predict(X_test_shapelet)

# ### Model Evaluation for SMOTE balanced Shapelet Modeling Classifier

#classification accuracy score
print("Correct classification rate:", accuracy_score(y_test_shapelet, predicted_labels_imb_SMOTE))

#create confusion matrix and classification report
print(confusion_matrix(y_test_shapelet, predicted_labels_imb_SMOTE))
print('\n')
print(classification_report(y_test_shapelet, predicted_labels_imb_SMOTE))

#visualize confusion matrix as a heatmap
sns.set(font_scale=3)
conf_matrix = confusion_matrix(y_test_shapelet, predicted_labels_imb_SMOTE)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
plt.title('Confusion matrix', fontsize=20)
plt.ylabel('True label', fontsize=15)
plt.xlabel('Clustering label', fontsize=15)
plt.show()

print('Adjusted Mutual Information Score is: \n')
print(adjusted_mutual_info_score(y_test_shapelet, predicted_labels_imb_SMOTE), '\n')
print('Adjusted Rand Score is: \n')
print(adjusted_rand_score(y_test_shapelet, predicted_labels_imb_SMOTE), '\n')
print('Normalized Mutual Information Score is: \n')
print(normalized_mutual_info_score(y_test_shapelet, predicted_labels_imb_SMOTE), '\n')
print('Homogeneity, Completeness, and V_Measure Scores are: \n')
print(homogeneity_completeness_v_measure(y_test_shapelet, predicted_labels_imb_SMOTE))

## visualize shapelets identified by the SMOTE balanced shapelet modeling classifier
fig = plt.figure(figsize=(20,16))
for i, sz in enumerate(shapelet_sizes_SMOTE.keys()):
    ax = plt.subplot(len(shapelet_sizes_SMOTE), 1, i + 1)
    ax.set_title("%d shapelets of size %d" % (shapelet_sizes_SMOTE[sz], sz), fontsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for shp in shp_clf_SMOTE.shapelets_:
        if ts_size(shp) == sz:
            plt.plot(shp.ravel());
plt.xlim([0, max(shapelet_sizes_SMOTE.keys()) - 1])
plt.tight_layout()
plt.show()

# ### Using Adaptive Synthetic Sampling (ADASYN)
# ### On ADASYN, Cf. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.309.942&rep=rep1&type=pdf

#apply ADASYN balancing to the training set and print number of records associated with each label
X_shapelet_resamp_train_ADASYN, y_shapelet_resamp_train_ADASYN = ADASYN().fit_sample(X_train_shapelet[:,:,0], y_train_shapelet)
print(sorted(Counter(y_shapelet_resamp_train_ADASYN).items()))

#reshape the features of the training set for use with the shapelet classifier, extending the third axis
X_shapelet_resamp_train_ADASYN_reshaped = np.expand_dims(X_shapelet_resamp_train_ADASYN, axis=2)

#create r=7 different shapelet lengths, using 14% of the length of time series for use as the base shapelet length
shapelet_sizes_ADASYN = grabocka_params_to_shapelet_size_dict(n_ts=X_shapelet_resamp_train_ADASYN_reshaped.shape[0],
                                                       ts_sz=X_shapelet_resamp_train_ADASYN_reshaped.shape[1],
                                                       n_classes=len(set(y_shapelet_resamp_train_ADASYN)),
                                                       l=0.14,
                                                       r=7)

#instantiate the ADASYN balanced shapelet classifier, with learning_rate=0.05
np.random.seed(15)
shp_clf_ADASYN = ShapeletModel(n_shapelets_per_size=shapelet_sizes_ADASYN,
                        optimizer=rmsprop(lr=.05),
#                        weight_regularizer=.01,
                        max_iter=300,
                        verbose_level=1)

#fit the ADASYN balanced shapelet model
shp_clf_ADASYN.fit(X_shapelet_resamp_train_ADASYN_reshaped, y_shapelet_resamp_train_ADASYN)

#predict labels using the fitted ADASYN balanced shapelet modeling classifier
predicted_labels_imb_ADASYN = shp_clf_ADASYN.predict(X_test_shapelet)

#classification accuracy score
print("Correct classification rate:", accuracy_score(y_test_shapelet, predicted_labels_imb_ADASYN))

#create confusion matrix and classification report
print(confusion_matrix(y_test_shapelet, predicted_labels_imb_ADASYN))
print('\n')
print(classification_report(y_test_shapelet, predicted_labels_imb_ADASYN))

#visualize confusion matrix as a heatmap
sns.set(font_scale=3)
conf_matrix = confusion_matrix(y_test_shapelet, predicted_labels_imb_ADASYN)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
plt.title('Confusion matrix', fontsize=20)
plt.ylabel('True label', fontsize=15)
plt.xlabel('Clustering label', fontsize=15)
plt.show()

print('Adjusted Mutual Information Score is: \n')
print(adjusted_mutual_info_score(y_test_shapelet, predicted_labels_imb_ADASYN), '\n')
print('Adjusted Rand Score is: \n')
print(adjusted_rand_score(y_test_shapelet, predicted_labels_imb_ADASYN), '\n')
print('Normalized Mutual Information Score is: \n')
print(normalized_mutual_info_score(y_test_shapelet, predicted_labels_imb_ADASYN), '\n')
print('Homogeneity, Completeness, and V_Measure Scores are: \n')
print(homogeneity_completeness_v_measure(y_test_shapelet, predicted_labels_imb_ADASYN))

## visualize shapelets identified by the ADASYN balanced shapelet modeling classifier
fig = plt.figure(figsize=(20,16))
for i, sz in enumerate(shapelet_sizes_ADASYN.keys()):
    ax = plt.subplot(len(shapelet_sizes_ADASYN), 1, i + 1)
    ax.set_title("%d shapelets of size %d" % (shapelet_sizes_ADASYN[sz], sz), fontsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for shp in shp_clf_ADASYN.shapelets_:
        if ts_size(shp) == sz:
            plt.plot(shp.ravel());
plt.xlim([0, max(shapelet_sizes_ADASYN.keys()) - 1])
plt.tight_layout()
plt.show()

# # Unsupervised model 4: LSTM Sequence-to-Sequence Autoencoder
# ### On Seq2Seq Autoencoders, Cf. https://www.cs.tut.fi/sgn/arg/dcase2017/documents/workshop_papers/DCASE2017Workshop_Amiriparian_172.pdf
# #### Cf. also https://pdfs.semanticscholar.org/6506/d13a84f90f8620fd028cfe5b8b9d0444a6d2.pdf

#piecewise aggregate approximation to reshape time series from length 514 to 512, facilitating input to LSTM nodes
paa = PAA(window_size=None, output_size=512, overlapping=True)

TS_List_512 = [pd.Series(paa.transform(scaler.fit_transform(ts).T).flatten()) for ts in all_max_list]

#concatenating list of dataframes to create single dataframe with shape (days) x (points)
points_x_days_512 = pd.concat(TS_List_512, axis=1)
print(points_x_days_512.shape)

days_x_points_512 = points_x_days_512.T
print(days_x_points_512.shape)

#converting to array of values
days_x_points_vals_512 = days_x_points_512.values

#normalizing using time series scaling
days_x_points_vals_512_trans = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(days_x_points_vals_512)

#clear the graph of the session to clean up memory
K.clear_session()

#specify input shape (n_batch, timesteps, input_dim), where None allows for an arbitrary batch size (specified during fitting)
inputs = Input(batch_shape=(None, 4, 128))

#LSTM encoder converts input sequences to a single vector containing info about the entire sequence
encoded = LSTM(4)(inputs)

#vector is repeated n number of times (where n is the number of timesteps in the output sequence)
decoded = RepeatVector(4)(encoded)

#the constant repeat vector sequence is converted to the target sequence
decoded = LSTM(128, return_sequences=True)(decoded)

#compile the models for the sequence autoencoder, and for the encoder itself to see the encoded values, if so desired
sequence_autoencoder = Model(inputs=inputs, outputs=decoded)
encoder = Model(inputs=inputs, outputs=encoded)

#compile the sequence autoencoder, including RMSProp optimizer and MSE loss function
sequence_autoencoder.compile(optimizer='rmsprop', loss='mse')

# ### Model Summary of LSTM Sequence-to-Sequence Autoencoder

#display model parameters
sequence_autoencoder.summary()

#split into training and validation sets, reshaping arrays to (n_records, timesteps, (length of series/timesteps))
reshaped_train_x = np.reshape(days_x_points_vals_512_trans[:120,:,0], (120, 4, 128))
reshaped_val_x = np.reshape(days_x_points_vals_512_trans[120:,:,0], (51, 4, 128))

#early stopping and checkpoint saving, for evaluation using the best performing model
AE_early_stop = EarlyStopping(monitor='val_loss', patience=40)
AE_checkpoint = ModelCheckpoint(filepath='./AE_LearningCheckpoint.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

#fitting the sequence autoencoder, using early stopping and TensorBoard to visualize model convergence
#     Note: for time series data, shuffle=False
#           batch_size must be compatible with the shape of your input data
sequence_autoencoder.fit(reshaped_train_x, reshaped_train_x,
                epochs=2000,
                batch_size=32,
                shuffle=False,
                validation_data=(reshaped_val_x, reshaped_val_x),
                callbacks=[TensorBoard(log_dir='./tmp/tb/trial_1', histogram_freq=0, write_graph=True),
                          AE_early_stop, AE_checkpoint])

# ### Model Evaluation of LSTM Sequence-to-Sequence Autoencoder

#To use TensorBoard, pass the specified log directory to the command line, as in:
#      tensorboard --logidr=/tmp/tb/trial_1
#Then use a browser to navigate to the returned URL, which will look something like:
#      http://DESKTOP-08LCPFI:6006
Image(filename="./TensorBoard_Seq2Seq_AE.png")

#load the best performing model
loaded_AE_model = load_model('./AE_LearningCheckpoint.hdf5')

#pass training and validation data to trained model to view decoded sequences, reshaping to original dimensions
decoded_train_seqs = loaded_AE_model.predict(reshaped_train_x)
print(decoded_train_seqs.shape)
reconstructed_train_x = np.reshape(decoded_train_seqs, (120, 512))
print(reconstructed_train_x.shape)

decoded_val_seqs = loaded_AE_model.predict(reshaped_val_x)
print(decoded_val_seqs.shape)
reconstructed_val_x = np.reshape(decoded_val_seqs, (51, 512))
print(reconstructed_val_x.shape)

# ### Interactive visualizations of clusters decoded using the LSTM Sequence-to-Sequence Autoencoder

#visualize decoded sequences from the training set
for i in range(len(reconstructed_train_x)):    
    pd.DataFrame(reconstructed_train_x[i]).iplot()

#visualize decoded sequences from the validation set
for i in range(len(reconstructed_val_x)):    
    pd.DataFrame(reconstructed_val_x[i]).iplot()

# #### Static plots

for i in range(len(reconstructed_train_x)):    
    pd.DataFrame(reconstructed_train_x[i]).plot()
    plt.show()

for i in range(len(reconstructed_val_x)):    
    pd.DataFrame(reconstructed_val_x[i]).plot()
    plt.show()

# ### Interactive visualizations of compressed sequences at the intermediate layer

#compile model to output values at the intermediate LSTM compression layer
intermediate_layer_model = Model(inputs=loaded_AE_model.input,
                                 outputs=loaded_AE_model.get_layer('lstm_1').output)

#generate predictions for training and validation sets at the intermediate layer
intermediate_output_train = intermediate_layer_model.predict(reshaped_train_x)
intermediate_output_val = intermediate_layer_model.predict(reshaped_val_x)

#visualize encoded sequences from the validation set
for i in range(len(intermediate_output_train)):    
    pd.DataFrame(intermediate_output_train[i]).iplot()

#visualize encoded sequences from the validation set
for i in range(len(intermediate_output_val)):    
    pd.DataFrame(intermediate_output_val[i]).iplot()

# #### Static plots

for i in range(len(intermediate_output_train)):    
    pd.DataFrame(intermediate_output_train[i]).plot()
    plt.show()

for i in range(len(intermediate_output_val)):    
    pd.DataFrame(intermediate_output_val[i]).plot()
    plt.show()

# # Unsupervised Model 5: Clustering of LSTM Autoencoded Sequences

# ## K-Means++ Clustering of LSTM Autoencoded Sequences

#concatenate arrays of reconstructed values output by the decoder
X_recon = np.concatenate((reconstructed_train_x,reconstructed_val_x))

#instantiate Kmeans++ clustering object
k_means_enc = KMeans(n_clusters=7, init='k-means++', n_init=1000, max_iter=1000, random_state=222, n_jobs=4)

#fit and predict using the reconstructed array of decoded values
y_pred_kmeans_enc = k_means_enc.fit_predict(X_recon)

# ### Interactive visualization of decoded sequence cluster centroids identified by K-Means++ clustering

#visualize centroids
for i in range(len(k_means_enc.cluster_centers_)):    
    pd.DataFrame(k_means_enc.cluster_centers_).iloc[i].iplot()

# #### Static plots

for i in range(len(k_means_enc.cluster_centers_)):    
    pd.DataFrame(k_means_enc.cluster_centers_).iloc[i].plot()
    plt.show()

# ### Model Evaluation of K-Means++ Clustering of LSTM Autoencoded Sequences

#inertia score (i.e., sum of the distances of samples to their closest cluster center)
print('The inertia score of the above modeling procedure is: \n')
print(k_means_enc.inertia_)

print('Silhouette Score of Autoencoded K-Means++ clustering, using Euclidean metric is: \n')
print(sklearn_silhouette_score(X_recon, y_pred_kmeans_enc, metric='euclidean'), '\n')
print('Silhouette Score of Autoencoded K-Means++ clustering, using Cosine metric is: \n')
print(sklearn_silhouette_score(X_recon, y_pred_kmeans_enc, metric='cosine'), '\n')
print('Silhouette Score of Autoencoded K-Means++ clustering, using Minkowski metric is: \n')
print(sklearn_silhouette_score(X_recon, y_pred_kmeans_enc, metric='minkowski'), '\n')

print('Calinski-Harabaz Score of Autoencoded K-Means++ clustering is: \n')
print(calinski_harabaz_score(X_recon, y_pred_kmeans_enc))

print('Silhouette Score of Autoencoded K-Means++ clustering, using the DTW metric is: \n')
print(tslearn_silhouette_score(X_recon, y_pred_kmeans_enc, metric=dtw))

print('Silhouette Score of Autoencoded K-Means++ clustering, using the Soft-DTW metric is: \n')
print(tslearn_silhouette_score(X_recon, y_pred_kmeans_enc, metric=soft_dtw))

# ## DTW Time Series K-Means Clustering of LSTM Autoencoded Sequences

#fitting using DTW Time Series K-Means clustering
ts_km_dtw_ae_standard = TimeSeriesKMeans(n_clusters=7, metric='dtw',max_iter=100,verbose=True,random_state=222).fit(X_recon)

# ### Interactive visualization of decoded sequence centroids identified by DTW Time Series K-Means clustering

#visualize centroids
for i in range(len(ts_km_dtw_ae_standard.cluster_centers_[:,:,0])):    
    pd.DataFrame(ts_km_dtw_ae_standard.cluster_centers_[:,:,0]).iloc[i].iplot()

# #### Static plots

for i in range(len(ts_km_dtw_ae_standard.cluster_centers_[:,:,0])):    
    pd.DataFrame(ts_km_dtw_ae_standard.cluster_centers_[:,:,0]).iloc[i].plot()
    plt.show()

# ### Model Evaluation of DTW Time Series K-Means Clustering of LSTM Autoencoded Sequences

#inertia score (i.e., sum of the distances of samples to their closest cluster center)
print('The inertia score of the above modeling procedure is: \n')
print(ts_km_dtw_ae_standard.inertia_)

print('Silhouette Score of Autoencoded DTW K-Means clustering, using Euclidean metric is: \n')
print(sklearn_silhouette_score(X_recon, ts_km_dtw_ae_standard.labels_, metric='euclidean'), '\n')
print('Silhouette Score of Autoencoded DTW K-Means clustering, using Cosine metric is: \n')
print(sklearn_silhouette_score(X_recon, ts_km_dtw_ae_standard.labels_, metric='cosine'), '\n')
print('Silhouette Score of Autoencoded DTW K-Means clustering, using Minkowski metric is: \n')
print(sklearn_silhouette_score(X_recon, ts_km_dtw_ae_standard.labels_, metric='minkowski'), '\n')

print('Calinski-Harabaz Score of Autoencoded DTW K-Means clustering is: \n')
print(calinski_harabaz_score(X_recon, ts_km_dtw_ae_standard.labels_))

print('Silhouette Score of Autoencoded DTW K-Means, using the DTW metric is: \n')
print(tslearn_silhouette_score(X_recon, ts_km_dtw_ae_standard.labels_, metric=dtw))

# ## Soft-DTW Time Series K-Means Clustering of LSTM Autoencoded Sequences

#fitting using Soft-DTW Time Series K-Means clustering
ts_km_dtw_ae = TimeSeriesKMeans(n_clusters=7, metric='softdtw', max_iter=100, verbose=True, random_state=222).fit(X_recon)

# ### Interactive visualization of decoded sequence centroids identified by Soft-DTW Time Series K-Means clustering

#visualize centroids
for i in range(len(ts_km_dtw_ae.cluster_centers_[:,:,0])):    
    pd.DataFrame(ts_km_dtw_ae.cluster_centers_[:,:,0]).iloc[i].iplot()

# #### Static plots

for i in range(len(ts_km_dtw_ae.cluster_centers_[:,:,0])):    
    pd.DataFrame(ts_km_dtw_ae.cluster_centers_[:,:,0]).iloc[i].plot()
    plt.show()

# ### Model Evaluation of Soft-DTW Time Series K-Means Clustering of LSTM Autoencoded Sequences

#inertia score (i.e., sum of the distances of samples to their closest cluster center)
print('The inertia score of the above modeling procedure is: \n')
print(ts_km_dtw_ae.inertia_)

print('Silhouette Score of Autoencoded Soft-DTW K-Means clustering, using Euclidean metric is: \n')
print(sklearn_silhouette_score(X_recon, ts_km_dtw_ae.labels_, metric='euclidean'), '\n')
print('Silhouette Score of Autoencoded Soft-DTW K-Means clustering, using Cosine metric is: \n')
print(sklearn_silhouette_score(X_recon, ts_km_dtw_ae.labels_, metric='cosine'), '\n')
print('Silhouette Score of Autoencoded Soft-DTW K-Means clustering, using Minkowski metric is: \n')
print(sklearn_silhouette_score(X_recon, ts_km_dtw_ae.labels_, metric='minkowski'), '\n')

print('Calinski-Harabaz Score of Autoencoded Soft-DTW K-Means clustering is: \n')
print(calinski_harabaz_score(X_recon, ts_km_dtw_ae.labels_))

print('Silhouette Score of Autoencoded Soft-DTW K-Means clustering, using the Soft-DTW metric is: \n')
print(tslearn_silhouette_score(X_recon, ts_km_dtw_ae.labels_, metric=soft_dtw))

# ## K-Shape Clustering of LSTM Autoencoded Sequences

#standardize the data for use with K-Shape clustering
X_recon_trans = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_recon)

#fit the standardize values using K-Shape clustering
ks_ae = KShape(n_clusters=7, max_iter=50, n_init=50, random_state=222).fit(X_recon_trans)

# ### Interactive visualization of decoded sequence centroids identified by K-Shape clustering

#visualize centroids
for i in range(len(ks_ae.cluster_centers_[:,:,0])):    
    pd.DataFrame(ks_ae.cluster_centers_[:,:,0]).iloc[i].iplot()

# #### Static plots

for i in range(len(ks_ae.cluster_centers_[:,:,0])):    
    pd.DataFrame(ks_ae.cluster_centers_[:,:,0]).iloc[i].plot()
    plt.show()

# ### Model Evaluation of K-Shape Clustering of LSTM Autoencoded Sequences

#inertia score (i.e., sum of the distances of samples to their closest cluster center)
print('The inertia score of the above modeling procedure is: \n')
print(ks_ae.inertia_)

print('Silhouette Score of Autoencoded K-Shape clustering, using Euclidean metric is: \n')
print(sklearn_silhouette_score(X_recon_trans[:,:,0], ks_ae.labels_, metric='euclidean'), '\n')
print('Silhouette Score of Autoencoded K-Shape clustering, using Cosine metric is: \n')
print(sklearn_silhouette_score(X_recon_trans[:,:,0], ks_ae.labels_, metric='cosine'), '\n')
print('Silhouette Score of Autoencoded K-Shape clustering, using Minkowski metric is: \n')
print(sklearn_silhouette_score(X_recon_trans[:,:,0], ks_ae.labels_, metric='minkowski'), '\n')

print('Calinski-Harabaz Score of Autoencoded K-Shape clustering is: \n')
print(calinski_harabaz_score(X_recon_trans[:,:,0], ks_ae.labels_))

print('Silhouette Score of Autoencoded K-Shape clustering, using the DTW metric is: \n')
print(tslearn_silhouette_score(X_recon_trans[:,:,0], ks_ae.labels_, metric=dtw))

print('Silhouette Score of Autoencoded K-Shape clustering, using the Soft-DTW metric is: \n')
print(tslearn_silhouette_score(X_recon_trans[:,:,0], ks_ae.labels_, metric=soft_dtw))
