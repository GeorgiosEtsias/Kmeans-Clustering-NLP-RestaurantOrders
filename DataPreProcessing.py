# Georios Etsias, Part 1: Data pre-processing 
# Clustering Pizzeria menus 

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Read original dataset
ordataset = pd.read_csv('orderItems.csv')
# Get product_category and product name
testset = ordataset.iloc[0:len(ordataset), [7,11]].values 

def stopwords_n_stemming(dataset_column):
    '''
    Remove stopwords, turn lower case and apply stemming in a column of the order dataset
    '''
    all_stopwords = stopwords.words('english')
    corpus = []
    for i in range(len(dataset_column)):
        review = re.sub('[^a-zA-Z]', ' ', dataset_column[i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return (corpus)

corpusname = stopwords_n_stemming(testset[:,0])
corpuscat = stopwords_n_stemming(testset[:,1])
            
# Creating a pd DataFrame with original index, product category, name and price
dataset = ordataset.iloc[0:len(testset), [9]]
indexx=list(range(0, len(testset)))
dataset.insert(0, "original index", indexx, True)
dataset.insert(1, "product_category_name", corpuscat, True)
dataset.insert(2, "product_name", corpusname, True)

# Split pizzas from the rest of the products(using category)
pizza = dataset[dataset['product_category_name'].str.contains(r'pizza')]
nonpizza = dataset[~dataset['product_category_name'].str.contains(r'pizza')]

# Plot the pizza - non-pizza distribution
def pie_chart(dataset, subset1, subset2, labels):
    sizes = [(len(subset1)/len(dataset))*100, (len(subset2)/len(dataset))*100]
    explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    return fig1

labels = 'pizzas', 'non-pizza products'
fig1 = pie_chart(dataset, pizza, nonpizza, labels)

## Saving the 2 new pd dataframes: pizza, nonpizza - final preprocessed data
pizza.to_pickle('./pizza.pkl')
nonpizza.to_pickle('./nonpizza.pkl')





