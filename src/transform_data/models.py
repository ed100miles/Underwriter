from feature_engineer import df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm

scaler = MinMaxScaler()

def scale_data(df, scale:list):
    for to_scale in scale:
        df[to_scale] = scaler.fit_transform([[x] for x in df[to_scale]])
    return df

to_scale_list = [x for x in df.columns if df[x].dtype == 'float64' 
                or df[x].dtype == 'int64']

df = scale_data(df, to_scale_list)

# print(df['INCOME'].head())
# print(df['AGE'].head())

train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)

# consider stratified samples using StraifiedShuffleSplit?


train_y, test_y = train_set['CLAIM_FLAG'], test_set['CLAIM_FLAG']
train_X = train_set.drop(['CLAIM_FLAG'], axis=1)
test_X = test_set.drop(['CLAIM_FLAG'], axis=1)

# print(list(test_y))

clf = svm.SVC()

clf.fit(train_X, train_y)

correct = 0
for x in range(1000):
    prediction = clf.predict([test_X.to_numpy()[x]])
    # print(f'predicted:{prediction}')
    label = list(train_y)[x]
    # print(f'label:{label}')
    if int(prediction) == int(label):
        correct += 1

    
print(f'accuracy:{correct/1000*100}%')