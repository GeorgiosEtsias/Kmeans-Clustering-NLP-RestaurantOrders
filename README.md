# NLP-Clustering-RestaurantOrders
Project for Slice
Compatibility: Python 3.8.5, libraries: numpy, matplotlib.pyplot, pandas, sklearn, nltk
Comments on the Python scripts
The given problem was solved by using multiple iterations of the ‘Bag of Words’ model alongside K-means clustering. The FinalResult.csv file, incudes the original dataset, plus two new columns, Predicted Product Category and Predicted Product Name. 
The solution consists of 4 different python (.py) scripts
1)	DataPreProcessing.py: product_name and product_category_name columns of the original .csv file are pre-processed using nltk. PorterStemmer. Then the orders are manually split into pizza and non-pizza products, and 2 .plk files are created.
2)	NonPizzaClusterring.py: Uses the nonpizza.plk file. K-means clustering was conducted using the product_type_name. Then for the products in each initial cluster, clustering is applied again using the data in the product_name column. The results of each clustering are presented using pie charts. In total 30 product types where identified, with 15 different products for each type, resulting in the identification of 450 different non-pizza product names. The script exports the AnalysisNonPizza.csv file, which includes the product distribution per product category, alongside their average price and price range.
3)	PizzaClustering.py: Conducts a similar analysis using the pizza.plk file. Clustering is conducted only 2 times, once for the product_type_name (30 clusters) and once for the product_name (45 clusters). This script generates similar outputs with the previous NonPizzaClusterring.py. 
4)	FinalResults.py: Creates a final .csv file that is equal to the original dataset plus two columns, representing the Predicted Product Categories and Predicted Product Names.
