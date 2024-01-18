import pandas as pd
import numpy as np

diet = pd.read_csv('diet.csv')
#print(diet.head())
recipe = pd.read_csv('recipes.csv')
#print(recipe.head())
requests = pd.read_csv('requests.csv')
reviews = pd.read_csv('reviews.csv')
#print(reviews.head())
subissions = pd.read_csv('submission_random.csv')
#print(subissions.head())



# all unique values in lowsugar of the requests
print(requests['LowSugar'].unique())
print(requests['LowFat'].unique())
print(requests['HighProtein'].unique())
print(requests['HighFiber'].unique())
print(requests['HighCalories'].unique())

print(requests.columns)


