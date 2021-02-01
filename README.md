# NLP-Clustering-RestaurantOrders

Compatibility: Python 3.8.5 

Libraries: numpy, pyplot, pandas, sklearn, nltk

## The dataset

A dataset including 16.500 food orders,  from 100 different pizza shops,  during June – November 2019, was analysed. The orders were categorized aprropiertly and conclusions were derived on the most popular products and their price distribution.

## Machine learning approach

The given problem was solved by using multiple iterations of the ‘Bag of Words’ model alongside K-means clustering. The FinalResult.csv file, incudes the original dataset, plus two new columns, Predicted Product Category and Predicted Product Name. 

## The 4 Python scripts

The solution consists of 4 different python (.py) scripts

1)	[DataPreProcessing.py](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/DataPreProcessing.py): product_name and product_category_name columns of the original .csv file are pre-processed using nltk. PorterStemmer. Then the orders are manually split into pizza and non-pizza products, and 2 .plk files are created.

2)	[NonPizzaClusterring.py](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/NonPizzaClustering.py): Uses the nonpizza.plk file. K-means clustering was conducted using the product_type_name. Then for the products in each initial cluster, clustering is applied again using the data in the product_name column. The results of each clustering are presented using pie charts. In total 30 product types where identified, with 15 different products for each type, resulting in the identification of 450 different non-pizza product names. The script exports the AnalysisNonPizza.csv file, which includes the product distribution per product category, alongside their average price and price range.

3)	[PizzaClustering.py](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/PizzaClustering.py): Conducts a similar analysis using the pizza.plk file. Clustering is conducted only 2 times, once for the product_type_name (30 clusters) and once for the product_name (45 clusters). This script generates similar outputs with the previous NonPizzaClusterring.py. 

4)	[FinalResults.py](https://github.com/GeorgiosEtsias/NLP-Clustering-RestaurantOrders/blob/main/FinalResults.py): Creates a final .csv file that is equal to the original dataset plus two columns, representing the Predicted Product Categories and Predicted Product Names.

## Trends in the post-pocessed dataset, respones to 'customers'

### Identifying the most popular products

30.8 % of the products that were ordered were pizzas. These pizza orders were distributed according to their corresponding ‘style’ as seen in Figure 1. Cheese pizza was by far the most popular product, constituting 32.5 % of the total orders.  If the different ways in which cheese pizza was represented in the dataset, such as ‘thin crust cheese pizza’ or ‘traditional NY cheese pizza’, are taken into account, then over 46 %  (Figure 2)of the total pizzas consumed were cheese pizzas. 

Figure 1: Distribution of various styles of pizzas.

Figure 2: Identifying the actual percentage of cheese pizzas.

The remaining 69.2 % of the orders were classified into 30 categories, as seen in Figure 3. Of these categories, appetizers were the most common products (18.22 %). An analysis of the individual products in each one of these 30 categories, was conducted in order to identify the most popular non-pizza product for each one. The results can be found on Table 1. 

Figure 3: Distribution of non-pizza products in categories.

In general, the distributions of orders, regarding both the pizza and non-pizza products was not surprising. As expected, NY style cheese pizza was the most popular product in general, while standard fast-food choices like mozzarella sticks, French fries, Caesar’s salad and sodas were also very common in the dataset.

Table 1: Most popular non-pizza products per identified product category, arranged in alphabetical order.

### Identifying price related trends 

Except for individual pizza slices, average pizza prices did not fluctuate that much, raging between 10 $ and 20$ (Figure 4). When examining price range, i.e. the difference between the highest and lowest recorded pizza prices (apart from the most popular product, cheese pizza), the biggest variations were observed in specialty pizzas with a lot of ingredients such as the meat lover and veggie pizzas (Figure5). In Figure 6 the correlation between the price range and the average price for each pizza style was identified. With the exception of cheese pizzas, for all other products the price range was never bigger than double the average product value. This indicates that cheese pizza has a relatively greater profit margin.

Figure 4: Average prices of the 10 most common pizza products.

Figure 5: Price range distribution of the 10 most common pizza products.

Figure 6: Price range divided by average product category price for the 10 most common pizza products.

Regarding the non-pizza products, average price and price range distribution are presented in Figures 7 and 8 respectively. Taking into account the correlation between these two variables, it was established that vary little price variation exists on beverages, sandwiches and subs, while the profit margins are much greater for appetizers and salads. 

Figure 7: Average prices of the 10 most common pizza products.

Figure 8: Price range distribution for the 10 most common non-pizza product categories.

Figure 9: Price range divided by average product category price for the 10 most common non-pizza product categories.


