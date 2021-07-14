import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pandas as pd

original_df = pd.read_csv('data\hotel_bookings_dataset.csv')

dropnull_df = original_df.drop(columns = ['company'])


most_freq = dropnull_df['country'].value_counts(dropna=True).idxmax()
dropnull_df['country'].fillna(most_freq, inplace = True)

most_freq = dropnull_df['agent'].value_counts(dropna=True).idxmax()
dropnull_df['agent'].fillna(most_freq, inplace = True)

most_freq = dropnull_df['children'].value_counts(dropna=True).idxmax()
dropnull_df['children'].fillna(most_freq, inplace = True)

noms = []  ## nominal 범주형, 텍스트형
nums = []  ## numerical 수치형
label = []

## loop를 사용해서 각 칼럼을 확인함
for column in dropnull_df.columns:
    if dropnull_df[column].dtype == 'object':
        noms.append(column)
    else:
        nums.append(column)
nums.remove('agent')
noms.append('agent')

nums.remove('is_canceled')
label.append('is_canceled')

nums.remove('is_repeated_guest')
noms.append('is_repeated_guest')


print('noms, 범주형',len(noms))
print(noms)
print('nums, 수치형',len(nums))
print(nums)
train_features, test_features, train_labels, test_labels = train_test_split(dropnull_df[noms+nums],
                                                                            dropnull_df['is_canceled'])
# Isolation Forest 방법을 사용하기 위해, 변수로 선언을 해 준다.
clf = IsolationForest(max_samples=1000, random_state=1)

# fit 함수를 이용하여, 데이터셋을 학습시킨다. race_for_out은 dataframe의 이름이다.
clf.fit(dropnull_df[nums])

# predict 함수를 이용하여, outlier를 판별해 준다. 0과 1로 이루어진 Series형태의 데이터가 나온다.
y_pred_outliers = clf.predict(dropnull_df[nums])



# 원래의 dataframe에 붙이기. 데이터가 0인 것이 outlier이기 때문에, 0인 것을 제거하면 outlier가 제거된  dataframe을 얻을 수 있다.
out = pd.DataFrame(y_pred_outliers)
out = out.rename(columns={0: "out"})
beforeOutlier_df = pd.concat([dropnull_df, out], 1)
dropoutlier_df = dropnull_df[beforeOutlier_df.out != -1]

dropoutlier_df
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
print(train_features[noms])
le.fit(train_features[noms])
enc_classes = {}

def encoding_label(x):
    le = LabelEncoder()
    le.fit(x)
    label = le.transform(x)
    enc_classes[x.name] = le.classes_

    return label

d1 = dropoutlier_df[noms].apply(encoding_label)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(d1)
print(scaler.transform((d1)))

d1_scaled = scaler.transform(d1)

d1_scaled = pd.DataFrame(d1_scaled)

scaler_nums = MinMaxScaler()
# d2 = dropoutlier[nums]
d2 = dropoutlier_df
d2 = d2[nums]

# scaler_num.fit(d2)
scaler_nums.fit(d2)
d2_scaled = scaler_nums.transform(d2)
d2_scaled = pd.DataFrame(d2_scaled)
d2_scaled.columns = d2.columns
train_features, test_features, train_labels, test_labels = train_test_split(dropoutlier_df[noms+nums],
                                                                            dropoutlier_df['is_canceled'])
scaler.fit(train_features)
scaled_df = d1_scaled.join(d2_scaled)
train_features, test_features, train_labels, test_labels = train_test_split(scaled_df[noms+nums],
                                                                            scaled_df['is_canceled'])
preprocessed_df = scaled_df.join(dropoutlier_df['is_canceled'])
train_features, test_features, train_labels, test_labels = train_test_split(preprocessed_df[noms+nums],
                                                                            preprocessed_df['is_canceled'])

from sklearn.feature_selection import SelectKBest, f_classif

selectK = SelectKBest(score_func=f_classif, k=8)
features = selectK.fit_transform(train_features, train_labels)

print(features[0])
train_features

from sklearn.decomposition import PCA
pca = PCA(n_component = 10)
fit = pca.fit(train_features)
print('ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ'+train_features)

print('ratio',fit.explained_variance_ratio, '\n\n\ncomponents',fit.components_)

pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정
printcipalComponents = pca.fit_transform(train_features)
principalDf = pd.DataFrame(data=printcipalComponents, columns = ['principal component1', 'principal component2'])









#
# # 주성분으로 이루어진 데이터 프레임 구성
# from sklearn.ensemble import ExtraTreeClassifier
# test.fit(train_features, train_labels)
# test = SelectKBest(score_func = chi2 =, k = 8)
# from sklearn.feature_selection import SelectKBest, chi2
# test = SelectKBest(score_func = chi2, k = 8)
# train_features.isnull()
# train_features.round(5)
# fit = test.fit(train_features, train_labels)
# train_lablel.isna().sum()
# train_labels.isna().sun()
# train_features.isNa.sum()
# train_features.isna().sum()
# train_label
# train_labels
# train_labels.astype('int')
# train_label.isna().sum()
# train_labels.isna().sum()
# runfile('C:/Users/soso6/Documents/GitHub/scientificProject2/main.py', wdir='C:/Users/soso6/Documents/GitHub/scientificProject2')
# original_df.isnull().sum()
# dropoutlier_df['is_canceled'].isna().sum()
# scaled_df
# preprocessed_df = scaled_df.join(dropoutlier_df['is_canceled'],how='right')
# emptpy_df = pd.DataFrame(dropoulier_df['is_canceled'])
# emptpy_df = pd.DataFrame(dropoutlier_df['is_canceled'])
# preprocessed_df = scaled_df.join(empty_df)
# preprocessed_df = scaled_df.join(emptpy_df)
# preprocessed_df.isna().sum()