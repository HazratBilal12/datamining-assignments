# Assignment 1: Understanding Data and Preprocessing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 1. Identify and characterize a dataset

Data = pd.read_csv(r'D:\CNG514_Spring_2021\CNG514-Assignment-1_Spring22\cng514 covid survey data.csv')
size = Data.size
shape = Data.shape
print('Size={}\n shape={}'.format(size,shape))
print(Data)
Data.info()

# ----------------------------------------------------------------------------------------
# 2. Identify and characterize attributes

# Central Tendency Measures
# i. Annual HouseHold Income(Continuous Attribute)
print("Mean of the 'Annual House Hold Income', Which is a Continuous Attribute")
print(Data['AnnualHouseholdIncome'].mean())
# print("Median")
# print(Data['AnnualHouseholdIncome'].median())
# print("Mode")
# print(Data['AnnualHouseholdIncome'].mode())

# ii. CoronavirusIntent_WashHands

print("Median of the 'CoronavirusIntent_WashHands'")
print(Data['CoronavirusIntent_WashHands'].median())
#print(Data['CoronavirusIntent_WashHands'].mean())

# iii. ZipCode

print('Mode of the ZipCode')
print(Data['ZipCode'].mode())

# --------------------------------------------------------------------------------------------

d = {'Annual_Income':Data['AnnualHouseholdIncome'],'Washing_Hands': Data['CoronavirusIntent_WashHands'],
     'ZipCode': Data['ZipCode']}
df_d=pd.DataFrame(d)
df_dmu = pd.DataFrame(d)
# print(df_dmu)

print('The Range of Data "d" is:')
Range = df_d.max()-df_d.min()
print(Range)
print('The Variance of Data "d" is:')
print(df_d.std())
print('The Quantiles of Data "d" is:')
Quantiles=df_d.quantile([0, 0.25, 0.5, 0.75, 1])
print(Quantiles)

# ----------------------------------------------------------------------------------------------

print('The Box and Histogram Plots of the Data "d" are:')
# df_d.plot.box(grid='True')

# Data['AnnualHouseholdIncome'].plot(kind='box', title='Annual_Household_Income')
# Data['CoronavirusIntent_WashHands'].plot(kind='box', title='Corona_virusIntent_WashHands')
# Data['ZipCode'].plot(kind='box', title='Zip_Code')
# plt.show()

Data.hist('AnnualHouseholdIncome', bins=[1000,10000,30000,50000,100000,200000,300000,500000])
Data.hist('CoronavirusIntent_WashHands', bins=[0, 10, 30, 50, 70, 90, 100])
Data.hist('ZipCode', bins=[0,100,300,400,500,600,700,1000])
plt.show()

# -----------------------------------------------------------------------------------------------

# 3. Data preprocessing

# print(df_d.isnull())
# print(df_d.notnull())
print("The missing values in each attributes:")

print(df_d['Annual_Income'].isna().sum())
print(df_d['Washing_Hands'].isna().sum())
print(df_d['ZipCode'].isna().sum())

print(df_d.isna().sum().sum())

print("Median is used to fill missing values in 'Annual_Income', 'Washing_Hands' and 'ZipCode'")
df_d['Annual_Income'].fillna(df_d['Annual_Income'].median(), inplace=True)
df_d['Washing_Hands'].fillna(df_d['Washing_Hands'].median(), inplace=True)
df_d['ZipCode'].fillna(df_d['ZipCode'].median(), inplace=True)

print(df_d.isna().sum().sum())


print(df_d.max())
print(df_d.min())

print('Identification of Noisy Data')
df_d['Annual_Income']= df_d['Annual_Income'].clip(lower=0)
df_d['Washing_Hands']= df_d['Washing_Hands'].clip(lower=0, upper=100)
df_d['ZipCode']= df_d['ZipCode'].clip(lower=0, upper=1000)

print(df_d.max())
print(df_d.min())


print('Discretization of Data:')

# A continuous attribute is discretized by equal frequency

df_d['Annual_Income']= pd.cut(df_d['Annual_Income'], bins=[0,1000,10000,30000,50000,100000,200000,300000,500000])
df_d['Washing_Hands']= pd.cut(df_d['Washing_Hands'], bins=[0,40,70,100])
df_d['ZipCode']= pd.cut(df_d['ZipCode'], bins=[0,200,400,600,800,1000])
print(df_d)


print('Normalization of Data Attributes Using min-max Normalization:')

df_dmm = df_dmu
for column in df_dmm.columns:
    df_dmm[column] = (df_dmm[column]-df_dmm[column].min())/(df_dmm[column].max()-df_dmm[column].min())
print(df_dmm)


print('Normalization of Data Attributes Using Z-Score Normalization:')

df_dz= df_dmu
for column in df_dz.columns:
    df_dz[column] = (df_dz[column] - df_dz[column].mean()) / df_dz[column].std()
print(df_dz)


print('Correlation between "AnnualHouseholdIncome" and two other attributes')

correlation = df_dmu.corr()
print(correlation)
