## Georios Etsias January 2021, Part 4: Final Results

# Importing pandas
import pandas as pd

ordataset = pd.read_csv('orderItems.csv')
nonpizzaresults = pd.read_pickle('nonpizzaresults.pkl')
pizzaresults = pd.read_pickle('pizzaresults.pkl')

# Creating the final category and product name column
nonpizzacat = nonpizzaresults.iloc[0:len(ordataset),0].values
pizzacat= pizzaresults.iloc[0:len(ordataset),0].values

nonpizzaname = nonpizzaresults.iloc[0:len(ordataset),1].values
pizzaname = pizzaresults.iloc[0:len(ordataset),1].values

name=pizzaname
category = pizzacat

for i in range(0,len(ordataset)):
    if pizzacat[i]=='':
        category[i]=nonpizzacat[i]
        name[i]=nonpizzaname[i]
        
# Creating the final results dataframe
FinalResults = ordataset
FinalResults.insert(12, "Predicted Product Category", category) 
FinalResults.insert(13, "Predicted Product Name", name) 

# Exporting the final reults in .csv
FinalResults.to_csv('./FinalResults.csv', index=False)
