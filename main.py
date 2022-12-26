# Georgios Etsias
"""
Conduct K-means clustering on a dataset containing 
menus of NY city pizza places
"""
# imort libraries
import pandas as pd

# Import custom functions
from functions.data_preprocess import stopwords_n_stemming  
from functions.plotting import plot_cluster_pie_chart
from functions.clustering import Clustering_functions 

# Instantiate class
clf = Clustering_functions()

## Step1: data preprocess
# Read original dataset
original_dataset = pd.read_csv('orderItems.csv')
# Get product_category and product name
testset = original_dataset.iloc[0:len(original_dataset), [7,11]].values

corpusname = stopwords_n_stemming(testset[:,0])
corpuscat = stopwords_n_stemming(testset[:,1])

# Creating a pd DataFrame with original index, product category, name and price
dataset = original_dataset.iloc[0:len(testset), [9]]
indexx=list(range(0, len(testset)))
dataset.insert(0, "original index", indexx, True)
dataset.insert(1, "product_category_name", corpuscat, True)
dataset.insert(2, "product_name", corpusname, True)

## Step 2: Split pizzas from the rest of the products(using category)
pizza_df = dataset[dataset['product_category_name'].str.contains(r'pizza')].copy()
nonpizza_df = dataset[~dataset['product_category_name'].str.contains(r'pizza')].copy()

# create a pizza - non pizza pie chart
labels = 'pizzas', 'non-pizza products'
plot_title = 'pizza - non-pizza product distribution'
sizes = [(len(pizza_df)/len(dataset))*100, (len(nonpizza_df)/len(dataset))*100]
fig = plot_cluster_pie_chart(labels, sizes, plot_title)

## Step 3: Define the clustering variables
nclusters_pizza = 30 # the number of clusters
nclusters_cat_nopizza, nclusters_name_nopizza = 30, 15 # the number of clusters
max_features = 50 # the maximum amount of features for the Bag of Words

# define the plotting package
plotting_package = 'matplotlib'

## Step 4: Pizza clustering
# Conduct K-means clustering based on product category
print('K-means clustering: pizza products, by product-category')
cat_y_kmeans, cat_clusternames, pizza_categories_df = clf.complete_clustering(pizza_df, 1,\
     nclusters_pizza, max_features, 'Initial pizza categories', 'predicted_category', plotting_package)

# Conduct K-means clustering based on product name
print('K-means clustering: pizza products, by product-name')
name_y_kmeans, name_clusternames, pizza_names_df = clf.complete_clustering(pizza_df, 2,\
     nclusters_pizza, max_features, 'Pizza products', 'predicted_name', plotting_package)

# Update the pizza dataframe
pizza_df.insert(4, 'predicted_category', pizza_categories_df['predicted_category'])
pizza_df.insert(5, 'predicted_name', pizza_names_df['predicted_name'])

## Step 5: Non-pizza clustering

#Conduct K-means clustering on product category
print('K-means clustering: non-pizza products, by product-category')
cat_y_kmeans, cat_clusternames, nonpizza_categories_df = clf.complete_clustering(nonpizza_df, 1,\
     nclusters_cat_nopizza, max_features, 'Initial nonpizza categories', 'predicted_category', plotting_package)
    
# Update the nonpizza dataframe
nonpizza_df.insert(4, 'predicted_category', nonpizza_categories_df['predicted_category'])

# Conduct K-means clustering based on product name
nonpizza_df.insert(5, 'predicted_name', '') # create an empty column to be updated
for jj in range(nclusters_cat_nopizza):
    print('K-means clustering: ' + cat_clusternames[jj] + ' products, by product-name')
    
    # Get the data sub-set
    target_product_cat = nonpizza_df[nonpizza_df['predicted_category'] == cat_clusternames[jj]].copy()
       
    # Conduct K-means clustering
    name_y_kmeans, name_clusternames, target_product_names = clf.complete_clustering(target_product_cat, 2,\
                 nclusters_name_nopizza, max_features, cat_clusternames[jj], 'predicted_name', plotting_package)
    
    # Update the nonpizza dataframe
    nonpizza_df.update(target_product_names) 
    
    del target_product_cat, name_y_kmeans, name_clusternames, target_product_names

## Step 6: Merge pizza and nonpizza results in the orginaldataset
final_df = original_dataset.copy()
final_df.insert(12, 'predicted_category', '')
final_df.insert(13, 'predicted_name', '')
final_df.update(nonpizza_df)
final_df.update(pizza_df)