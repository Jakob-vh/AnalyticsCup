import pandas as pd
import numpy as np
from scipy import stats

diet = pd.read_csv('diet.csv')
#print(diet.head())
recipes = pd.read_csv('recipes.csv')
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

recipes["Calories"] = recipes["Calories"].astype(int)
recipes["FatContent"] = recipes["FatContent"].astype(int)
recipes["SaturatedFatContent"] = recipes["SaturatedFatContent"].astype(int)
recipes["FiberContent"] = recipes["FiberContent"].astype(int)
recipes["SugarContent"] = recipes["SugarContent"].astype(int)
recipes["ProteinContent"] = recipes["ProteinContent"].astype(int)
recipes["CookTime"] = recipes["CookTime"].astype(int)
recipes["PrepTime"] = recipes["PrepTime"].astype(int)


requests["Time"] = requests["Time"].astype(int)


#detect and remove outliers

#exludes rows that have values outside a range of 3 standart deviations from the mean (using zscore methode)
recipes= recipes[(np.abs(stats.zscore(recipes[['Calories','FatContent','SaturatedFatContent','FiberContent','SugarContent','ProteinContent']]))<3).all(axis=1)]
#exludes rows that have values outside a range of 1 standart deviations from the mean (using zscore methode)
recipes= recipes[(np.abs(stats.zscore(recipes[['CookTime']]))<1).all(axis=1)]
#exludes rows that have values outside a range of 0.5 standart deviations from the mean (using zscore methode)
recipes= recipes[(np.abs(stats.zscore(recipes[['PrepTime']]))<0.5).all(axis=1)]
requests= requests[(np.abs(stats.zscore(requests[['Time']]))<0.5).all(axis=1)]