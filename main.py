#1. Imports
import pandas as pd
import numpy as np
from scipy import stats


#2.Seed and Data Loading
seed = 2024
np.random.seed(seed)


#  2.1 reading the csv files
diet = pd.read_csv('diet.csv')
recipes = pd.read_csv('recipes.csv')
requests = pd.read_csv('requests.csv')
reviews = pd.read_csv('reviews.csv')
subissions = pd.read_csv('submission_random.csv')



#3. Data Cleaning
# all unique values in lowsugar of the requests
print(requests['LowSugar'].unique())
print(requests['LowFat'].unique())
print(requests['HighProtein'].unique())
print(requests['HighFiber'].unique())
print(requests['HighCalories'].unique())

print(requests.columns)

#Casting the columns to int
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