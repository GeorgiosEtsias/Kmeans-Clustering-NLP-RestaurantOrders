## Slice project, Georios Etsias January 2021, Part 1: Data pre-processing 

## Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

## Importing the original dataset
ordataset = pd.read_csv('orderItems.csv')
# Getting product category, product name
testset = ordataset.iloc[0:len(ordataset), [7,11]].values 

## Cleaning names and categories using PorterStemmer (NLP) -not in pandas-
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
corpuscat = []
corpusname = []
for j in range(0,2):
    for i in range(0, len(testset)):
        review = re.sub('[^a-zA-Z]', ' ', testset[i,j])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        if j == 0:
            corpusname.append(review)
        else:
            corpuscat.append(review)
            
## Creating pd DataFrame with original index, product category, name and price
dataset = ordataset.iloc[0:len(testset), [9]]
indexx=list(range(0, len(testset)))
dataset.insert(0, "original index", indexx, True)
dataset.insert(1, "product_category_name", corpuscat, True)
dataset.insert(2, "product_name", corpusname, True)

## Hard-coded splitting of pizzas from the rest of the products(using category)
pizza = dataset[dataset['product_category_name'].str.contains(r'pizza')]
nonpizza = dataset[~dataset['product_category_name'].str.contains(r'pizza')]
    
## Plotting pizza / non-pizza pie chart
labels = 'pizzas', 'non-pizza products'
sizes = [(len(pizza)/len(dataset))*100, (len(nonpizza)/len(dataset))*100]
explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

## Saving the 2 new pd dataframes: pizza, nonpizza - final preprocessed data
pizza.to_pickle('./pizza.pkl')
nonpizza.to_pickle('./nonpizza.pkl')

nonpizza.to_csv('./nonpizza.csv')



