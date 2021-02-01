## Slice project, Georios Etsias January 2021, Part 2: non-pizza products 

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the original and the nonpizza dataset
ordataset = pd.read_csv('orderItems.csv')
dataset0 = pd.read_pickle('nonpizza.pkl')
dataset= dataset0.iloc[0:len(dataset0),1].values

nonpizzacategory = ["" for x in range(len(ordataset))]
nonpizzaname = ["" for x in range(len(ordataset))]

## Clustering (1st round) based on product_category_name
# Creating the Bag of Words model for nonpizza products
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 50)
X = cv.fit_transform(dataset).toarray()

# Chosing the maximum amount of features for the Bag of Words
# Calculate the minimum % of products that will be classified as various 
sumvar = 0
for i in range (0,len(dataset)):
    if np.sum(X[i,:]) < 1:
        sumvar=sumvar+1
print ((sumvar/(len(dataset)))*100,'% of products complete loss of information -class.various')

        
# K - means clustering
nclusters = 30
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = nclusters, init = 'k-means++', random_state = 2)
y_kmeans = kmeans.fit_predict(X)

print ('initial clustering inertia: ',(kmeans.inertia_))
            
# Automatic cluster name generation 
centers = np.array(kmeans.cluster_centers_) # cluster centroids
for i in range (0, nclusters):
    for j in range(0, 50):
        if centers[i,j]>0.6:
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

# Pie chart - more helpfull - no numbers in there
labels = clusternames
sizes = percorders
explode = [0]*nclusters  # only "explode" the 2nd slice (i.e. 'Hogs')
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, 
            shadow=True, startangle=90)
ax1.axis('equal')  
ax1.set_title('Initial nonpizza categories')
plt.show()


# Final product category distribution
result1 = np.concatenate((np.array(clusternames).reshape(len(clusternames),1), 
                             np.array(norders).reshape(len(norders),1)),1)
#print ('Final clustering results:',result1)

# Adding the results of first round of clustering to pd dataframe
pred_cat = []
for i in range (0,len(X)):  
    pred_cat.append(clusternames[y_kmeans[i]])
    nonpizzacategory[dataset0.index[i]]=clusternames[y_kmeans[i]]
dataset2= dataset0.iloc[0:len(dataset),:]
dataset2.insert(4, "pred_cat", pred_cat, True) 
# -------------------------------------------------------------------------- #
## Clustering (2nd round) based on product_name
for jj in range(0,30): # clustering for every product category
    prod = dataset2.loc[dataset2['pred_cat'] == result[jj,0]]
    prod2 = prod.iloc[0:len(prod),2].values
    # print(result[jj,0])
    # New bag of words using product_name instead of product_category_name
    cv = CountVectorizer(max_features = 30)
    X1 = cv.fit_transform(prod2).toarray()
    # clustering 
    nclusters=15
    kmeans = KMeans(n_clusters = nclusters, init = 'k-means++', random_state = 2)
    y_kmeans1 = kmeans.fit_predict(X1)
    
    # Automatic cluster name generation 
    centers = np.array(kmeans.cluster_centers_) # cluster centroids
    for i in range (0, nclusters):
        for j in range(0, 30):
            if centers[i,j]>0.6:
                centers[i,j]=1
            else:
                centers[i,j]=0
    clusternames = cv.inverse_transform(centers) # reversing bag of words for centers
    glue = ' '
    for i in range(0, len(clusternames)):
        clusternames [i] = glue.join(clusternames [i])
        if len(clusternames [i]) < 2 or clusternames [i] == result[jj,0] :
            clusternames [i] = 'various ' + result[jj,0]

             
    print (result[jj,0],'clustering inertia: ',(kmeans.inertia_))
    # Clustering results and evaluation
    # Calculating orders per product category
    norders = []
    for i in range (0,nclusters):
        norders.append(((np.count_nonzero(y_kmeans1 == i))/len(prod))*100)
  
    # Pie chart - more helpfull - no numbers in there
    labels = clusternames
    sizes = norders
    explode = [0]*nclusters  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, 
            shadow=True, startangle=90)
    ax1.axis('equal')  
    ax1.set_title(result[jj,0])
    plt.show()

    # Final product category distribution
    result1 = np.concatenate((np.array(clusternames).reshape(len(clusternames),1), 
                             np.array(norders).reshape(len(norders),1)),1)
    #print ('Final clustering results:',result1)

    # Adding the results of first round of clustering to pd dataframe
    pred_name = []
    for i in range (0,len(X1)):  
        nonpizzaname[prod.index[i]]=clusternames[y_kmeans1[i]]
        pred_name.append(clusternames[y_kmeans1[i]])
    prod.insert(5, "pred_name", pred_name, True)   
    
    # Calculating average price and price range for each specific product name
    avprice = []
    pricerange = []
    for i in range(0, nclusters):
        indivproduct = prod.loc[prod['pred_name'] == result1[i,0]]
        avprice.append(indivproduct['product_type_price'].mean())
        maxprice = indivproduct['product_type_price'].max()
        minprice = indivproduct['product_type_price'].min()
        pricerange.append(maxprice - minprice)
        
    # pd - saving product name, distribution, average price and price range
    # Defining column names of the pd array
    column = np.array([result[jj,0]+' products', result[jj,0] + ' % distr', 
                      result[jj,0]+' mean price', result[jj,0]+' price range'])
    # create the pd array
    if jj == 0:
        d = {column[0]: result1[:,0], column[1]: result1[:,1],
             column[2]: avprice,column[3]: pricerange}
        AnalysisNonPizza = pd.DataFrame(data=d)
    else:
        AnalysisNonPizza.insert(jj*4, column[0], result1[:,0], True) 
        AnalysisNonPizza.insert(jj*4+1, column[1], result1[:,1], True)
        AnalysisNonPizza.insert(jj*4+2, column[2], avprice, True)
        AnalysisNonPizza.insert(jj*4+3, column[3], pricerange, True)  
# -------------------------------------------------------------------------- #
## Saving the final results of the analysisnng
# exporting non-pizzaname and non-pizzacategory
d = {'nonpizacategory': nonpizzacategory, 'nonpizzaname': nonpizzaname}
nonpizzaresluts = pd.DataFrame(data=d)
nonpizzaresluts.to_pickle('./nonpizzaresults.pkl') 

# Exporting the clusterig reults in .csv
AnalysisNonPizza.to_csv('./AnalysisNonPizza.csv', index=False)

