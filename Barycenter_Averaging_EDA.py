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
%matplotlib inline

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

#read data from csv
df = pd.read_csv('./Data/Sensor_Data_8_17-Onward.csv')

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

#create list of 1D arrays of maximum photo sensor values (excluding the first, which is empty after subsetting for values > 1)
all_max_list = []
for i in range(len(DF_PS_List_Day[:])-1):
    all_max_list.append(DF_PS_List_Day[i+1].iloc[:,-1].values.reshape(-1,1))
	
#unscaled PAA
from pyts.transformation import PAA
paa = PAA(window_size=None, output_size=514, overlapping=True)

Unscaled_TS_List = [pd.Series(paa.transform(ts.T).flatten()) for ts in all_max_list]

#concatenate series in the list into a single dataframe with dimensions (# of data points) x (# of days)
points_x_days_unscaled = pd.concat(Unscaled_TS_List, axis=1)
print(points_x_days_unscaled.shape)

#transpose dataframe, whose resulting dimensions are (# of days) x (# of data points)
days_x_points_unscaled = points_x_days_unscaled.T
print(days_x_points_unscaled.shape)

#visualize values for all days in dataframe
for i in range(len(points_x_days_unscaled.columns)):
    plt.plot(points_x_days_unscaled.iloc[:, i])
    plt.show()
    plt.close()
	
from tslearn.barycenters import EuclideanBarycenter

euc_bar = EuclideanBarycenter()
for i in range(57):
    for ts in days_x_points_unscaled.values[(3*i):3+(3*i)]:
        plt.plot(ts.ravel(), "k-", alpha=.3, linewidth=1)
        plt.plot(euc_bar.fit(days_x_points_unscaled.values[(3*i):3+(3*i)]), "r-", linewidth=1)
        plt.title("Euclidean barycenter")
    plt.show()
	
w = np.array([0.3, 0.6, 0.9])

euc_bar = EuclideanBarycenter(weights=w)
for i in range(57):
    for ts in days_x_points_unscaled.values[(3*i):3+(3*i)]:
        plt.plot(ts.ravel(), "k-", alpha=.3, linewidth=1)
        plt.plot(euc_bar.fit(days_x_points_unscaled.values[(3*i):3+(3*i)]), "r-", linewidth=1)
        plt.title("Euclidean barycenter")
    plt.show()
	
sdtw_bar = SoftDTWBarycenter(max_iter=50, weights=w, gamma=100.)
for i in range(57):
    for ts in days_x_points_unscaled.values[(3*i):3+(3*i)]:
        plt.plot(ts.ravel(), "k-", alpha=.3, linewidth=1)
        plt.plot(sdtw_bar.fit(days_x_points_unscaled.values[(3*i):3+(3*i)]), "r-", linewidth=1)
        plt.title("Soft-DTW Barycenter")
    plt.show()
	
##############################

#for all columns of all data frames, convert all Lux values to W/m^2
for i in range(len(DF_PSIR_List)):
    for j in range(len(DF_PSIR_List[0].columns)-4):
        DF_PSIR_List[i].iloc[:, j] = DF_PSIR_List[i].iloc[:, j]/126.5736
		
#for all data frames, create a column holding the maximum value across all columns for each row
for i in range(len(DF_PSIR_List)):
    DF_PSIR_List[i]['MAX'] = DF_PSIR_List[i][['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10',
                                         'd11','d12','d13']].max(axis=1)
										 
#for all data frames, subset only rows whose MAX value is > 1
DF_PSIR_List_Day = [ data_frame[data_frame['MAX'] > 1] for data_frame in DF_PSIR_List]

DF_PSIR_List_Day_Sorted = [ df[['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','MAX',
                                   'sky1','amb1','sky2','amb2']] for df in DF_PSIR_List_Day]
								   
DF_PSIR_MIN = [ df.assign(sky_min = df.loc[:,['sky1','sky2']].min(axis=1)) for df in DF_PSIR_List_Day_Sorted]

DF_PSIR_MINS = [ df.assign(amb_min = df.loc[:,['amb1','amb2']].min(axis=1)) for df in DF_PSIR_MIN]

DF_PSIR_SkyDiff = [ df.assign(sky_diff = ((df['sky_min'] - df['amb_min'])/1000)-(-17)) for df in DF_PSIR_MINS]

#create list of 1D arrays of maximum photo sensor values (excluding the first, which is empty after subsetting for values > 1)
all_min_list = []
for i in range(len(DF_PSIR_SkyDiff[:])-1):
    all_min_list.append(DF_PSIR_SkyDiff[i+1].iloc[:,-1].values.reshape(-1,1))
	
#across all dataframes, find minimum length of the first dimension
min_length_list = []
for i in range(len(DF_PSIR_SkyDiff)-1):
    min_length_list.append(DF_PSIR_SkyDiff[i+1].shape[0])
print(min(min_length_list))

#unscaled PAA
from pyts.transformation import PAA
paa = PAA(window_size=None, output_size=514, overlapping=True)

Unscaled_TS_List_IR = [pd.Series(paa.transform(ts.T).flatten()) for ts in all_min_list]

#concatenate series in the list into a single dataframe with dimensions (# of data points) x (# of days)
points_x_days_unscaled_IR = pd.concat(Unscaled_TS_List_IR, axis=1)
print(points_x_days_unscaled_IR.shape)

#transpose dataframe, whose resulting dimensions are (# of days) x (# of data points)
days_x_points_unscaled_IR = points_x_days_unscaled_IR.T
print(days_x_points_unscaled_IR.shape)

w = np.array([0.3, 0.6, 0.9])

euc_bar = EuclideanBarycenter(weights=w)
for i in range(57):
    for ts in days_x_points_unscaled_IR.values[(3*i):3+(3*i)]:
        plt.plot(ts.ravel(), "k-", alpha=.3, linewidth=1)
        plt.plot(euc_bar.fit(days_x_points_unscaled_IR.values[(3*i):3+(3*i)]), "r-", linewidth=1)
        plt.title("Euclidean barycenter")
    plt.show()
	

