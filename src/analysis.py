import csv
import matplotlib.pyplot as plt
from numpy.lib.function_base import average
import pandas as pd
from pathlib import Path
import numpy as np

# Import and convert data to df:
datapath = Path('./data/car_insurance_claim.csv')
data_df = pd.read_csv(datapath)

# Clean Income data:
def format_income(df):
    '''removes '$' and ',' from incomes, replaces nans with mean and converts all to int dtype.'''
    for index, entry in enumerate(df['INCOME']):
        if type(entry) == str:
            income = int(entry[1:].replace(',', ''))
            df.at[index, 'INCOME'] = income
    for index, entry in enumerate(df['INCOME']):
        if type(entry) == float:
            income = int(df['INCOME'].mean())
            df.at[index, 'INCOME'] = income
    data_df['INCOME'] = data_df['INCOME'].astype(int)
    
format_income(data_df)

def check_types(df, df_index:str):
    unique_types = set()
    for entry in df[df_index]:
        unique_types.add(type(entry))
    return unique_types

income_data_types = check_types(data_df, 'INCOME')

assert len(income_data_types) == 1 and income_data_types.pop() == int
# print(data_df['INCOME'].describe())

data_df.sort_values(by=['INCOME'], ascending=False, inplace=True)

x, y = [], []

incomes_and_flags = list(zip(data_df['INCOME'].values, data_df['CLAIM_FLAG'].values))

zero, under50k, under100k, under150k, under200k, under250k, under300k, over300k  = ([] for _ in range(8))

for instance in incomes_and_flags:
    income, claim_flag = instance
    if income == 0:
        zero.append(claim_flag)
    if 0 < income < 50000:
        under50k.append(claim_flag)
    if 50000 < income < 100000:
        under100k.append(claim_flag)
    if 100000 < income < 150000:
        under150k.append(claim_flag)
    if 150000 < income < 200000:
        under200k.append(claim_flag)
    if 200000 < income < 250000:
        under250k.append(claim_flag)
    if 250000 < income < 300000:
        under300k.append(claim_flag)    
    if income > 300000:
        over300k.append(claim_flag)

incomes = [zero, under50k, under100k, under150k, under200k, under250k, under300k, over300k]
income_labels = ['0','1-50','51-100','101-150','151-200','201-250','251-300','300+']

accident_rate = [round(sum(x)/len(x)*100, 2) for x in incomes]
income_claim_rate =list(zip(income_labels, accident_rate))

average_accident_rate = round(sum(data_df['CLAIM_FLAG'])/len(data_df['CLAIM_FLAG'])*100, 2)
print(average_accident_rate)


plt.plot(income_labels, accident_rate)

plt.plot(income_labels, 
        [average_accident_rate for x in range(len(income_labels))])

plt.title('Income / Claim %')
plt.xlabel('Income')
plt.ylabel('Claim %')
plt.show()

