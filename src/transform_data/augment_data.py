import pandas as pd
from clean_data import clean_df

'''
Create some more data?:
- Income/bluebook = value of car as % of income
- MVR / age = more points at young age suggests riskier driver
- Identify collectable cars using bluebook + age?
- OLDCLAIM / age = many claim in short time of driving suggest risk?
'''

# Create new fields:

df = clean_df

df['BLUE%INCOME*'] = df['INCOME'] / df['BLUEBOOK']  # -0.026 - POOR
df['PTS_PER_AGE*'] = df['MVR_PTS'] / df['AGE']      # +0.232 - GREAT!
df['CLM_PER_AGE*'] = df['CLM_FREQ'] / df['AGE']      # +0.234 - GREAT!
df['RICH_KIDS*'] = df['INCOME'] / df['AGE']         # -0.112 - FAIR
df['HOME_PER_BLUE*'] = df['HOME_VAL'] / df['BLUEBOOK'] # - 0.04 - POOR
df['CLM_PER_MILE*'] = df['CLM_FREQ'] / df['TRAVTIME'] # 0.120 - FAIR

df.info()

with open('/Users/Ed/Desktop/new_claim_data.csv', 'w') as file:
    file.write(df.to_csv())

