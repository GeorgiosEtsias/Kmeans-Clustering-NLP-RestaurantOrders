# Import libraries
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

#import custom functions
from functions.plotting import matplotlib_pie_chart, plotly_pie_chart

class Clustering_functions():
    '''
    Contains all the class methods necessary for conducting 
    K-menas clustering in the main script 
    '''
    def complete_clustering(self, original_dataset, target_column, nclusters,\
        max_features, plot_title, predicted_attribute_name, plotting_package, export_graph):
        '''
        Function includes al the necessary steps for succesfully coducting
        K-means clustering
        '''
        dataset = original_dataset.iloc[:,target_column].values

        # 1. Create the Bag of Words for nonpizza products
        cv = CountVectorizer(max_features = max_features)
        BoW = cv.fit_transform(dataset).toarray() 
  
        # 2. Calculate the minimum % of products that will be classified as various
        self.products_as_various(dataset, BoW)

        # 3. Conduct clustering according to category
        kmeans, y_kmeans = self.conduct_clustering(BoW, nclusters)

        # 4. Automatic cluster name generation 
        cluster_radius = 0.6 #Pick a cluster radius value
        clusternames = self.automatic_cluster_name(cv, kmeans, nclusters, max_features, cluster_radius)

        # 5. Calculate the prder distribution per cluster5     
        product_distribution = self.orders_per_cluster(dataset, nclusters, y_kmeans, clusternames) 

        # 6. Visualize the data
        if plotting_package == 'matplotlib':
            fig = matplotlib_pie_chart(clusternames, product_distribution[:,2], plot_title, export_graph)
        elif plotting_package == 'plotly':
            fig = plotly_pie_chart(clusternames, product_distribution[:,2], plot_title, export_graph)
              
        # 7. Adding the results of this round of clustering to the original dataframe
        predicted_attribute = []
        for i in range (0,len(BoW)):   predicted_attribute.append(clusternames[y_kmeans[i]])
        new_dataset= original_dataset.copy()
        new_dataset[predicted_attribute_name] = np.array(predicted_attribute)

        return y_kmeans, clusternames, new_dataset

    def products_as_various(self, dataset, BoW):
        '''
        Calculates the minimum % of products that will be classified as various.
        Represents the information loss due to the chosen number of max features
        '''
        sumvar = 0
        for i in range (len(dataset)):
            if np.sum(BoW[i,:]) < 1: sumvar=sumvar+1
        lost_info = round((sumvar/(len(dataset)))*100,2)
        print ('Complete loss of information will occur for ' + str(lost_info) +'% of products') 

    def conduct_clustering(self, BoW, nclusters):
        '''
        Conducts the clustering and plots initial clustering inertia
        '''
        kmeans = KMeans(n_clusters = nclusters, init = 'k-means++', random_state = 2)
        y_kmeans = kmeans.fit_predict(BoW)
        print ('initial clustering inertia: ',round((kmeans.inertia_),2))
        return kmeans, y_kmeans
         
    def automatic_cluster_name(self, cv, kmeans, nclusters, max_features, cluster_radius):
        '''
        Automatically generates cluster names
        '''
        centers = np.array(kmeans.cluster_centers_) # cluster centroids
        for i in range (nclusters):
            for j in range(centers.shape[1]): #in case actual features are less than max_features
                if centers[i,j]>cluster_radius: centers[i,j]=1 # assign word to cluster name
                else: centers[i,j]=0 # do not assign word
        clusternames = cv.inverse_transform(centers)
        glue = ' '
        for i in range(len(clusternames)): 
            clusternames [i] = glue.join(clusternames [i])
            if (len(clusternames [i]) == 0): clusternames [i] = 'various'
    
        return clusternames

    def orders_per_cluster(self, dataset, nclusters, y_kmeans, clusternames):
        '''
        Calculates the order distributuin per cluster
        '''
        norders = [] # orders per cluster
        for i in range (0,nclusters): norders.append(np.count_nonzero(y_kmeans == i))
        # Product category percentage of total orders
        percorders = [] # % of orders per cluster
        for i in range (0, nclusters): percorders.append(round(norders[i]/(len(dataset)*0.01),2))   
        
        # Final product category distribution
        product_distribution = np.concatenate((np.array(clusternames).reshape(len(clusternames),1),
                        np.array(norders).reshape(len(norders),1), 
                        np.array(percorders).reshape(len(percorders),1)),1)
    
        return (product_distribution)
