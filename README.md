# K-means clustering of products from 100 NYC pizzerias

![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/nyc_pizza.png)

Compatibility: Python 3.8.5 

Libraries: numpy, pyplot, pandas, sklearn, nltk, plotly, matplolib

## Table of contents
- [Repository structure](#repository-structure)

- [The dataset](#the-dataset)

- [Machine learning approach](#machine-learning-approach)

- [Source-code breakdown](#source-code-breakdown)

     [main.py and main_notebook.ipynb](#mainpy-and-main_notebookipynb-scripts)
     
     [Python scripts in the 'functions/' directory](#python-scripts-in-functions-directory)

- [Trends in the post-pocessed dataset](#trends-in-the-post-pocessed-dataset)

     [Identifying the most popular products / clustering results](#identifying-the-most-popular-products--clustering-results)
     
     [Identifying price related trends](#identifying-price-related-trends)

## Repository structure
```
┌── ProjectFigures/
│── data/
│ └── orderItems.csv
│── figures/
│ │── html_figures/
│ │── png_figures/
│ └── svg_figures/
│── functions/
│ │── clustering.py
│ │── data_preprocess.py
│ └── plotting.py
│── python_environment_req/
│ │──  myenv.yml
│ └── requirements.txt
│── .gitattributes
│── README.md
│── main.py 
└── main_notebook.ipynb
```

## The [dataset](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/data//orderItems.csv)

- A dataset including **16.500 food orders**,  from **100 different pizza shops**,  during June – November 2019, was analysed. 
- There are two columns of interest **product category** and **product name**. K-means clustering of the products was conducted using the data in both columns.

## Machine learning approach

- The given problem was solved by using multiple iterations of the **‘Bag of Words’** model alongside **the scikit-learn K-means clustering**. The FinalResult.csv file, incudes the original dataset, plus two new columns, Predicted Product Category and Predicted Product Name. 
- The orders were categorized aprropiertly, and 2 new columns were created called **Predicted Product Category** and **Predicted Product Name**.

## Source-code breakdown
### [main.py](https://github.com/GeorgiosEtsias/Kmeans-Clustering-NLP-RestaurantOrders/blob/main/main.py) and [main_notebook.ipynb](https://github.com/GeorgiosEtsias/Kmeans-Clustering-NLP-RestaurantOrders/blob/main/main_notebook.ipynb) scripts
1) The data .csv is read as a pandas dataframe.
2) The **product_name** and **product_category_name** columns of the dataframe are pre-processed using nltk. PorterStemmer.
3) A finalised dataframe with 4 coluns are created.

```python
# Read original dataset
original_dataset = pd.read_csv('data/orderItems.csv')
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
```
5) Then the orders are manually split into pizza and non-pizza products,  and a pie chart showing this distribution is created

```python
pizza_df = dataset[dataset['product_category_name'].str.contains(r'pizza')].copy()
nonpizza_df = dataset[~dataset['product_category_name'].str.contains(r'pizza')].copy()

# create a pizza - non pizza pie chart
labels = 'pizzas', 'non-pizza products'
plot_title = 'pizza - non-pizza product distribution'
sizes = [(len(pizza_df)/len(dataset))*100, (len(nonpizza_df)/len(dataset))*100]
fig = plotly_pie_chart(labels, sizes, plot_title, export_graph)
```

#### pizza clustering
Two rounds of clustering were conducted, one using the product-name and the other using the product-category column.

```python
# Conduct K-means clustering based on product category
print('K-means clustering: pizza products, by product-category')
cat_y_kmeans, cat_clusternames, pizza_categories_df = clf.complete_clustering(pizza_df, 1,\
     nclusters_pizza, max_features, 'Initial pizza categories', 'predicted_category', plotting_package, export_graph)
# Conduct K-means clustering based on product name
print('K-means clustering: pizza products, by product-name')
name_y_kmeans, name_clusternames, pizza_names_df = clf.complete_clustering(pizza_df, 2,
                 nclusters_pizza, max_features, 'Pizza products', 'predicted_name', plotting_package, export_graph)```

```

#### non-pizza clustering

For the non-pizza products K-means clustering was conducted using the product_type_name. Then for the products in each initial cluster, clustering is applied again using the data in the product_name column. The results of each clustering are presented using pie charts. In total 30 product types where identified, with 15 different products for each type, resulting in the identification of 450 different non-pizza product names. 

```python
#Conduct K-means clustering on product category
print('K-means clustering: non-pizza products, by product-category')
cat_y_kmeans, cat_clusternames, nonpizza_categories_df = clf.complete_clustering(nonpizza_df, 1,\
     nclusters_cat_nopizza, max_features, 'Initial nonpizza categories', 'predicted_category', plotting_package, export_graph)
    
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
                 nclusters_name_nopizza, max_features, cat_clusternames[jj], 'predicted_name', plotting_package, export_graph)
    
    # Update the nonpizza dataframe
    nonpizza_df.update(target_product_names) 
    
    del target_product_cat, name_y_kmeans, name_clusternames, target_product_name
```
### Python scripts in functions/ directory

The solution consists of 3 different python (.py) scripts

#### [clustering.py](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/functions/clustering.py): 
includes a method-only class called **Clustering_functions** that conduct k-means clustering for a given dataset, and return the extracted clusters and the corresonding cluster names.

#### Clustering_functions, class layout 
```python
──class Clustering_functions()
 │── def complete_clustering
 │── def products_as_various
 │── def conduct_clustering
 │── def automatic_cluster_name
 └── def orders_per_cluster
```
#### complete_clustering, function
```python
   def complete_clustering(self, original_dataset, target_column, nclusters,\
        max_features, plot_title, predicted_attribute_name, plotting_package, export_graph):
        '''
        Function includes al the necessary steps for succesfully coducting
        K-means clustering
        '''
        dataset = original_dataset.iloc[:,target_column].values

        # 1. Create the Bag of Words for the given dataset
        cv = CountVectorizer(max_features = max_features)
        BoW = cv.fit_transform(dataset).toarray() 
  
        # 2. Calculate the minimum % of products that will be classified as various
        self.products_as_various(dataset, BoW)

        # 3. Conduct clustering 
        kmeans, y_kmeans = self.conduct_clustering(BoW, nclusters)

        # 4. Automatic cluster name generation 
        cluster_radius = 0.6 #Pick a cluster radius value
        clusternames = self.automatic_cluster_name(cv, kmeans, nclusters, max_features, cluster_radius)

        # 5. Calculate the order distribution per cluster    
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

```
 
#### [data_preprocess.py](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/functions/data_preprocess.py): 
Removes **stopwords** and conducts **stemming** in a given dataset using the **nltk** package.

#### [plotting.py](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/functions/plotting.py): 
Scipt includes 2 finctions: **matplotlib_pie_chart** and **plotly_pie_chart** that create pie charts using the **maplotlib** and **plotly** python packages respectively.

```python
def matplotlib_pie_chart(clusternames, product_distribution, plot_title, export_graph):
    '''
    Creates a pie chart to show the distribution of the various products
    using matplotlib
    '''
    fig, ax = plt.subplots()
    ax.pie( product_distribution, labels = clusternames, shadow=True, startangle=90)
    ax.axis('equal')  
    ax.set_title(plot_title)
    plt.show()

    if export_graph:
        # export png
        export_png(fig, plot_title)  

    return fig
```

```python
def plotly_pie_chart(clusternames, product_distribution, plot_title, export_graph):
    '''
    Creates a pie chart to show the distribution of the various products using Plotly
    '''
    fig = px.pie(values=product_distribution, names=clusternames, title=plot_title)
    fig.show()
    
    if export_graph:
        # export html
        export_html(fig, plot_title)  
    
    return fig
```
Three additional functions **export_html**, **export_png** and **export_svg** export the plots localy as **.html**, **.png** and **.svg** respectively.

![alt text](https://github.com/GeorgiosEtsias/Kmeans-Clustering-NLP-RestaurantOrders/blob/main/figures/png_figures/dessert.png)
![alt text](https://github.com/GeorgiosEtsias/Kmeans-Clustering-NLP-RestaurantOrders/blob/main/figures/svg_figures/dessert.svg)

Figure 1: A matplotlib-generated pie-chart (top) displaying the identified clusters of deserts vs the equivalent plotly-generated graph (bottom).

## Trends in the post-pocessed dataset

### Identifying the most popular products / clustering results

30.8 % of the products that were ordered were pizzas. These pizza orders were distributed according to their corresponding ‘style’ as seen in Figure 1. Cheese pizza was by far the most popular product, constituting 32.5 % of the total orders.  If the different ways in which cheese pizza was represented in the dataset, such as ‘thin crust cheese pizza’ or ‘traditional NY cheese pizza’, are taken into account, then over 46 %  (Figure 2)of the total pizzas consumed were cheese pizzas. 

![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/Figure1.png)

Figure 2: Distribution of various styles of pizzas.

![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/Figure2.png)

Figure 3: Identifying the actual percentage of cheese pizzas.

The remaining 69.2 % of the orders were classified into 30 categories, as seen in Figure 3. Of these categories, appetizers were the most common products (18.22 %). An analysis of the individual products in each one of these 30 categories, was conducted in order to identify the most popular non-pizza product for each one. The results can be found on Table 1. 

![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/Figure3.png)

Figure 4: Distribution of non-pizza products in categories.

In general, the distributions of orders, regarding both the pizza and non-pizza products was not surprising. As expected, NY style cheese pizza was the most popular product in general, while standard fast-food choices like mozzarella sticks, French fries, Caesar’s salad and sodas were also very common in the dataset.

Table 1: Most popular non-pizza products per identified product category, arranged in alphabetical order.

![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/Table1.png)

### Identifying price related trends 

Except for individual pizza slices, average pizza prices did not fluctuate that much, raging between 10 $ and 20$ (Figure 4). When examining price range, i.e. the difference between the highest and lowest recorded pizza prices (apart from the most popular product, cheese pizza), the biggest variations were observed in specialty pizzas with a lot of ingredients such as the meat lover and veggie pizzas (Figure5). In Figure 6 the correlation between the price range and the average price for each pizza style was identified. With the exception of cheese pizzas, for all other products the price range was never bigger than double the average product value. This indicates that cheese pizza has a relatively greater profit margin.

![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/Figure4.png)

Figure 5: Average prices of the 10 most common pizza products.

![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/Figure5.png)

Figure 6: Price range distribution of the 10 most common pizza products.

![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/Figure6.png)

Figure 7: Price range divided by average product category price for the 10 most common pizza products.

Regarding the non-pizza products, average price and price range distribution are presented in Figures 7 and 8 respectively. Taking into account the correlation between these two variables, it was established that vary little price variation exists on beverages, sandwiches and subs, while the profit margins are much greater for appetizers and salads. 
![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/Figure7.png)

Figure 8: Average prices of the 10 most common pizza products.

![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/Figure8.png)

Figure 9: Price range distribution for the 10 most common non-pizza product categories.

![alt text](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/ProjectFigures/Figure9.png)

Figure 10: Price range divided by average product category price for the 10 most common non-pizza product categories.
