# Import libraries
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

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