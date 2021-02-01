## Slice project, Georios Etsias January 2021, Part 3: pizza products 

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
ordataset = pd.read_csv('orderItems.csv') # Modified in NonPizzaClustering.py
dataset0 = pd.read_pickle('pizza.pkl')
dataset= dataset0.iloc[0:len(dataset0),1].values

pizzacategory = ["" for x in range(len(ordataset))]
pizzaname = ["" for x in range(len(ordataset))]

## Clustering (1st round) based on product_category_name
# Creating the Bag of Words model for nonpizza products
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 50)
X = cv.fit_transform(dataset).toarray()
  
# K - means clustering
nclusters = 30
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = nclusters, init = 'k-means++', random_state = 2)
y_kmeans = kmeans.fit_predict(X)
          
# Automatic cluster name generation 
centers = np.array(kmeans.cluster_centers_) # cluster centroids
for i in range (0, nclusters):
    for j in range(0, 50):
        if centers[i,j]>0.8:
            centers[i,j]=1
        else:
            centers[i,j]=0
clusternames = cv.inverse_transform(centers) # reversing bag of words for centers
glue = ' '
for i in range(0, len(clusternames)):
    clusternames [i] = glue.join(clusternames [i])
    if (len(clusternames [i]) == 0):
        clusternames [i] = 'various'
        
# Final clustering results and evaluation
# Calculating orders per product category
norders = []
for i in range (0,nclusters):
    norders.append(np.count_nonzero(y_kmeans == i))
# Product category percentage of total orders
percorders = []
for i in range (0, nclusters):
    percorders.append(norders[i]/(len(dataset)*0.01))
    
# Final product category distribution
result = np.concatenate((np.array(clusternames).reshape(len(clusternames),1), 
                         np.array(percorders).reshape(len(percorders),1)),1)
# print ('Initial clustering results:',result)
  
# Pie chart - more helpfull - no numbers in there
labels = clusternames
sizes = percorders
explode = [0]*nclusters  
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, 
            shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Initial pizza categories')
plt.show()

# Adding the results of first round of clustering to pd dataframe
pred_cat = []
for i in range (0,len(X)):  
    pred_cat.append(clusternames[y_kmeans[i]])
    pizzacategory[dataset0.index[i]]=clusternames[y_kmeans[i]]
dataset2= dataset0.iloc[0:len(dataset),:]
dataset2.insert(4, "pred_cat", pred_cat, True) 

# -------------------------------------------------------------------------- #
## Clustering (2nd round) based on product_name
dataset1= dataset0.iloc[0:len(dataset0),2].values

cv = CountVectorizer(max_features = 30)
X1 = cv.fit_transform(dataset1).toarray()
# clustering 
nclusters= 40
kmeans = KMeans(n_clusters = nclusters, init = 'k-means++', random_state = 2)
y_kmeans1 = kmeans.fit_predict(X1)

# Automatic cluster name generation 
centers = np.array(kmeans.cluster_centers_) # cluster centroids
for i in range (0, nclusters):
    for j in range(0, 30):
        if centers[i,j]>0.8:
                centers[i,j]=1
        else: 
                centers[i,j]=0
clusternames = cv.inverse_transform(centers) # reversing bag of words for centers
glue = ' '
for i in range(0, len(clusternames)):
            clusternames [i] = glue.join(clusternames [i])
            if len(clusternames [i]) < 2:
                clusternames [i] = 'various pizzas' 
            
print ('Pizza product name clustering inertia: ',(kmeans.inertia_))
# Clustering results and evaluation
# Calculating orders per product category
norders = []
for i in range (0,nclusters):
    norders.append(((np.count_nonzero(y_kmeans1 == i))/len(dataset2))*100)
                       
# Pie chart - more helpfull - no numbers in there
labels = clusternames
sizes = norders
explode = [0]*nclusters  # only "explode" the 2nd slice (i.e. 'Hogs')
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, 
    shadow=True, startangle=90)
ax1.axis('equal')  
ax1.set_title('Pizza products')
plt.show()           

# Final product category distribution
result1 = np.concatenate((np.array(clusternames).reshape(len(clusternames),1), 
                        np.array(norders).reshape(len(norders),1)),1)
#print ('Final clustering results:',result1)

# Adding the results of first round of clustering to pd dataframe
pred_name = []
for i in range (0,len(X1)):  
    pizzaname[dataset0.index[i]]=clusternames[y_kmeans1[i]]
    pred_name.append(clusternames[y_kmeans1[i]])
dataset2.insert(5, "pred_name", pred_name, True)                   
            
# Calculating average price and price range for each specific product name
avprice = []
pricerange = []
for i in range(0, nclusters):
    indivproduct = dataset2.loc[dataset2['pred_name'] == result1[i,0]]
    avprice.append(indivproduct['product_type_price'].mean())
    maxprice = indivproduct['product_type_price'].max()
    minprice = indivproduct['product_type_price'].min()
    pricerange.append(maxprice - minprice) 
                   
# pd - saving product name, distribution, average price and price range
# Defining column names of the pd array
column = np.array(['pizza products', 'pizza prod. % distr', 
                  'pizza prod. mean price', 'pizza prod. price range'])
d = {column[0]: result1[:,0], column[1]: result1[:,1],
             column[2]: avprice,column[3]: pricerange}
AnalysisPizza = pd.DataFrame(data=d)

# -------------------------------------------------------------------------- #
## Saving the final results of the analysisnng
# exporting pizzaname and pizzacategory
d = {'pizacategory': pizzacategory, 'pizzaname': pizzaname}
pizzaresluts = pd.DataFrame(data=d)
pizzaresluts.to_pickle('./pizzaresults.pkl') 

# Exporting the clusterig reults in .csv
AnalysisPizza.to_csv('./AnalysisPizza.csv', index=False)


        
        

