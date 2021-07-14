import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from encoding_noms import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import statsmodels.api as sm



original_df = pd.read_csv('hotel_bookings_dataset.csv')

#null값이 너무 많은 column drop
dropnull_df = original_df.drop(columns=['company'])
dropnull_df = dropnull_df.drop(columns=['agent'])
# dropnull_df = dropnull_df.drop(columns=['assigned_room_type'])
# dropnull_df = dropnull_df.drop(columns=['reserved_room_type'])

#최빈 값으로 각 column의 null 값 대체
most_freq = dropnull_df['country'].value_counts(dropna=True).idxmax()
dropnull_df['country'].fillna(most_freq, inplace = True)

# most_freq = dropnull_df['agent'].value_counts(dropna=True).idxmax()
# dropnull_df['agent'].fillna(most_freq, inplace = True)

most_freq = dropnull_df['children'].value_counts(dropna=True).idxmax()
dropnull_df['children'].fillna(most_freq, inplace = True)

# 범주형과 수치형 column 구분
noms = []  ## nominal 범주형, 텍스트형
nums = []  ## numerical 수치형
label = []

## loop를 사용해서 각 칼럼을 확인함
for column in dropnull_df.columns:
    if dropnull_df[column].dtype == 'object':
        noms.append(column)
    else:
        nums.append(column)
# nums.remove('agent')
# noms.append('agent')

nums.remove('is_canceled')
label.append('is_canceled')

nums.remove('is_repeated_guest')
noms.append('is_repeated_guest')


print('noms, 범주형',len(noms))
print(noms)
print('nums, 수치형',len(nums))
print(nums)
print('label',len(nums))
print(label)




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

# print(dropoutlier_df)

d1 = dropoutlier_df[noms].apply(encoding_label)

d2 = dropoutlier_df[nums+label]

encoded_df = d1.join(d2)

##각 column 데이터 값의 범위가 다르므로 이를 scaling한다.
scaler = MinMaxScaler()

#train set에 fit한 scaler를 train set과 test set에 transform 시킨다
train_features, test_features, train_labels, test_labels = train_test_split(encoded_df[noms+nums],
                                                                            encoded_df['is_canceled'])

print(train_features)
scaler.fit(train_features)
scaled_np = scaler.transform(train_features)

scaled_train_features_df = pd.DataFrame(scaled_np, columns=noms+nums)
train_labels = train_labels.reset_index(drop=True)


print(scaled_train_features_df)
print(train_labels)

#PCA로 변수를 선택한다.
pca = PCA(n_components =0.9)
fit = pca.fit(scaled_train_features_df)
# print('------------------------\n',fit,'\n------------------------------')
print('------------------------\n',fit.explained_variance_ratio_,'\n------------------------------')

train_principalComponents = pca.fit_transform(scaled_train_features_df)
test_principalComponents = pca.transform(test_features)

PCA_column = ['principal component'+str(x+1) for x in range(13)]


principalDf = pd.DataFrame(data=train_principalComponents, columns = PCA_column)
print('여기?')
corr_df = pd.concat([principalDf, scaled_train_features_df], axis=1)
corr = corr_df.corr(method='pearson')

# df_how_corr = corr[abs(corr['principal component1']) >= 0.2]['principal component1']
corr_result = pd.DataFrame()
for i in range(len(corr.columns)):
    if i == 1:
        break
    else:
        # df_how_corr = corr[abs(corr['principal component'+str(i+1)]) >= 0.3]['principal component'+str(i+1)]
        df_how_corr = corr[abs(corr['principal component'+str(i+1)]) >= 0.3]['principal component'+str(i+1)]
        sorted_df = df_how_corr.sort_values(ascending=False)

        print('------------------------\n',sorted_df,'------------------------\n')





#Logistic Model
logistic_model = LogisticRegression()
logistic_model.fit(principalDf, train_labels)


#train data set에 대한 예측률
print('train set 예측률\n',logistic_model.score(principalDf, train_labels))
print()


#test data set에 대한 예측률
print('test set 예측률\n',logistic_model.score(test_principalComponents, test_labels))
print()

print('country,', 'distribution_channel,', 'assigned_room_type,', 'deposit_type,','reservation_status,','\n',
      'lead_time,', 'previous_cancellations,','booking_changes,', 'required_car_parking_spaces,','total_of_special_requests')
#각 변수들의 기울기
print('각 변수들의 coeficient\n',logistic_model.coef_) #순서대로 '연령', '성별', '신입학','편입학','지역'
print()

#로지스틱 모형 정보 요약
model_pred = logistic_model.predict(test_principalComponents)

print('report\n',classification_report(test_labels, model_pred))

#Decision Tree 모형
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(train_principalComponents, train_labels)

print("훈련 세트 정확도: {:.3f}".format(tree.score(train_principalComponents, train_labels)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(test_principalComponents, test_labels)))

print('---------------------------------------------------------')

logit = sm.Logit(train_labels, train_principalComponents)
result = logit.fit()
print(result.summary())


# #selectK 알고리즘을 이용해 변수를 선택한다.
# selectK = SelectKBest(score_func=f_classif, k=10)
# train_selected = selectK.fit_transform(scaled_train_features_df, train_labels)
# test_selected = selectK.transform(test_features)
#
# all_names = scaled_train_features_df.columns
# ## selector.get_support()
# selected_mask = selectK.get_support()
# ## 선택된 특성(변수)들
# selected_names = all_names[selected_mask]
# ## 선택되지 않은 특성(변수)들
# unselected_names = all_names[~selected_mask]
# print('Selected names: ', selected_names)
# print('Unselected names: ', unselected_names)

# #Logistic Model
# logistic_model = LogisticRegression()
# logistic_model.fit(train_selected, train_labels)
#
#
# #train data set에 대한 예측률
# print('train set 예측률\n',logistic_model.score(train_selected, train_labels))
# print()
#
#
# #test data set에 대한 예측률
# print('test set 예측률\n',logistic_model.score(test_selected, test_labels))
# print()
#
# print('country,', 'distribution_channel,', 'assigned_room_type,', 'deposit_type,','reservation_status,','\n',
#       'lead_time,', 'previous_cancellations,','booking_changes,', 'required_car_parking_spaces,','total_of_special_requests')
# #각 변수들의 기울기
# print('각 변수들의 coeficient\n',logistic_model.coef_) #순서대로 '연령', '성별', '신입학','편입학','지역'
# print()
#
# #로지스틱 모형 정보 요약
# model_pred = logistic_model.predict(test_selected)
#
# print('report\n',classification_report(test_labels, model_pred))
#
# #Decision Tree 모형
# tree = DecisionTreeClassifier(max_depth=4, random_state=0)
# tree.fit(train_selected, train_labels)
#
# print("훈련 세트 정확도: {:.3f}".format(tree.score(train_selected, train_labels)))
# print("테스트 세트 정확도: {:.3f}".format(tree.score(test_selected, test_labels)))