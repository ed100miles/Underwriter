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

