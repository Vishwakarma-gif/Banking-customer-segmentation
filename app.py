import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st
#from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = "Compunnel digital")
st.image("compunnel.png", width = 120)
st.markdown(""" # Bank Customer Segmentation App

	""")

input_data = st.file_uploader("**Upload Your Data**")

#data = pd.DataFrame()

if input_data is not None:

    data = pd.read_csv(input_data)
    data.dropna(axis=1,inplace=True)
    data.drop('CUST_ID', axis = 1, inplace = True)
    #Normalization
    scaler = StandardScaler()
    creditcard_df_scaled = scaler.fit_transform(data)

    #Kmeans Clustering

    kmeans = KMeans(7)
    kmeans.fit(creditcard_df_scaled)
    labels = kmeans.labels_ 
    cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [data.columns])
    cluster_centers = scaler.inverse_transform(cluster_centers)
    cluster_centers = pd.DataFrame(data = cluster_centers, columns = [data.columns])
    y_kmeans = kmeans.fit_predict(creditcard_df_scaled)
    creditcard_df_cluster = pd.concat([data, pd.DataFrame({'cluster': labels})], axis = 1)

    st.subheader('Data with Cluster Labels')
    st.write(creditcard_df_cluster.head(5))
    st.subheader('Customer per Cluster')
    plt.figure(figsize = (3,2))
    sns.countplot(x ='cluster', data = creditcard_df_cluster)
    plt.legend()
    st.pyplot()
    # Pairplot
    st.subheader('Cluster Pair Plots')
    for i in data.columns:
        plt.figure(figsize = (35,5))
        for j in range(7):
            plt.subplot(1,7,j+1)
            cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
            cluster[i].hist(bins = 20)
            plt.title('{}   \nCluster  {}'.format(i,j))
        st.pyplot()
    
    # PCA
    pca = PCA(n_components = 2)
    principal_comp = pca.fit_transform(creditcard_df_scaled)
    pca_df = pd.DataFrame(data = principal_comp, columns = ['pca1', 'pca2'])
    pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis = 1)

    st.subheader('Principle Components with Cluster Labels')
    st.write(pca_df.head(5))

    plt.figure(figsize = (10,10))
   
    ax = sns.scatterplot(x='pca1', y='pca2', hue="cluster", data = pca_df, palette = ['red','green','blue','pink',
                                                                                    'yellow','gray','purple'])

    st.pyplot()

else:
	st.text("Please Upload the CSV File")



