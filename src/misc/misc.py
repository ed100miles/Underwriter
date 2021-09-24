# From models.py:

# def scale_data(df, scale: list):
#     for to_scale in scale:
#         df[to_scale] = scaler.fit_transform([[x] for x in df[to_scale]])
#     return df


# to_scale_list = [column for column in df.columns if len(
#     df[column].unique()) > 2] # gets non-binary features


# df = scale_data(df, to_scale_list)

# train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)
# train_y, test_y = train_set['CLAIM_FLAG'], test_set['CLAIM_FLAG']
# train_X = train_set.drop(['CLAIM_FLAG'], axis=1)
# test_X = test_set.drop(['CLAIM_FLAG'], axis=1)


# def get_cat_indicies(df):
#     cat_indicies = []
#     for index, feature in enumerate(df.columns):
#         if df[feature].dtype == 'uint8':
#             cat_indicies.append(index)
#     return cat_indicies


# cat_indicies = get_cat_indicies(train_X)


# rus = RandomUnderSampler()
# ros = RandomOverSampler()
# smotenc = SMOTENC(categorical_features=cat_indicies)

# smote_train_X, smote_train_y = smotenc.fit_resample(train_X, train_y)
# os_train_X, os_train_y = ros.fit_resample(train_X, train_y)
# us_train_X, us_train_y = ros.fit_resample(train_X, train_y)

# print(len(train_y))
# print(train_y.head())

# claim_count = 0
# non_claim_count = 0
# for entry in os_train_y:
#     if entry == 1:
#         claim_count += 1
#     else:
#         non_claim_count += 1
# print(claim_count)
# print(non_claim_count)
# print(f'balance:{claim_count/non_claim_count*100}%')

# train_data = pd.concat([train_X, train_y], axis=1)
# train_data = shuffle(train_data)

# grouped = train_data.groupby('CLAIM_FLAG')
# print(grouped)


# def count_claim_vs_non_claim(df):
#     claims = 0
#     non_claims = 0
#     for index, row in df.iterrows():
#         if row['CLAIM_FLAG'] == 1:
#             claims += 1
#         else:
#             non_claims += 1
#     print('claims:',claims, ' non-claims:', non_claims)

# count_claim_vs_non_claim(train_data)

# def balance_data(df):
#     out_df = pd.DataFrame()
#     claims = 0
#     non_claims = 0
#     for _, row in df.iterrows():
#         if row['CLAIM_FLAG'] == 1:
#             if claims <= non_claims:
#                 out_df = pd.concat([out_df, row])
#                 claims += 1
#         else:
#             if non_claims <= claims:
#                 out_df = pd.concat([out_df, row])
#                 non_claims += 1
#     return out_df
# balanced_df = balance_data(train_data)
# print(balanced_df.head())

# TODO: Balance the training data, think it's throwing it off !!!!

# print(list(test_y))



############################################################################



def get_int64_only_df(df):
    '''Returns new df object comprising of only columns with int64 datatype'''
    int_columns = []
    for column in df.columns:
        if df[column].dtype == int:
            int_columns.append(column)
    ints_df = df[int_columns]
    return ints_df.copy()




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

