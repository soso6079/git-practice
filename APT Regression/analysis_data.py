from math import sqrt
import category_encoders as ce
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import explained_variance_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import missingno as msno
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# 1. Data load
# data = pd.read_pickle('전체 데이터.pickle')
# visualizing null values
# msno.matrix(df=data.iloc[:,10:16], color=(0.1, 0.6, 0.8),fontsize=18)
# plt.xticks(rotation=0, ha='center')
#
# plt.show()
# data.info()
# null_drop = data[data['가까운 지하철 역 이름'] != 'error occur']
#
# del data
#
# 2. Null row drop
# dropped_df = null_drop.drop(columns=['transaction_id', 'transaction_date',
#                                      'apartment_id', '가까운 지하철 역 이름',
#                                      '위도', '경도', 'apt'])
# del null_drop
#
# 3. Nominal data encoding
# 3.1 dong 칼럼 인코딩
# encoder = ce.BinaryEncoder(cols=['dong'])
# df_binary = encoder.fit_transform(dropped_df.loc[:, 'dong'])

# coded_df = pd.concat([dropped_df.iloc[:, 0], df_binary, dropped_df.iloc[:, 2:]], axis=1)
#
# del dropped_df, df_binary
#
# # 3.2 city 칼럼 인코딩
# coded_df = pd.get_dummies(coded_df, columns=['city'], prefix=['city'])
#
# y = coded_df.iloc[:, -5]
# X = pd.concat([coded_df.iloc[:, :-5], coded_df.iloc[:, -4:]], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# columns = X.columns
# del coded_df
# del X, y
# # print('start!')
# #4. Outlier 제거
# X_train.to_pickle('mid/X_train.pickle')
# X_test.to_pickle('mid/X_test.pickle')
# y_train.to_pickle('mid/y_train.pickle')
# y_test.to_pickle('mid/y_test.pickle')
# # print('done!')
#
#
# clf=IsolationForest(n_estimators=50, max_samples=50, contamination=float(0.004),
#                     max_features=1.0, bootstrap=False, n_jobs=-1, random_state=None, verbose=0)
# # 50개의 노드 수, 최대 50개의 샘플
# # 0.04%의 outlier 색출.
# clf.fit(X_train)
# pred = clf.predict(X_train)
# X_train['anomaly']=pred
# outliers=X_train.loc[X_train['anomaly']==-1]
# outlier_index=list(outliers.index)
# #print(outlier_index)
# #Find the number of anomalies and normal points here points classified -1 are anomalous
# print(X_train['anomaly'].value_counts())

X_train = pd.read_pickle('이상치 제거_train.pickle')
X_test = pd.read_pickle('이상치 제거_test.pickle')
y_train = pd.read_pickle('이상치 제거_y_train.pickle')
y_test = pd.read_pickle('이상치 제거_y_test.pickle')
columns = X_train.columns
#
# 4. Scaling with RobustScaler - Outlier에 영향을 적게 받는 알고리즘
robustScaler = RobustScaler()
robustScaler.fit(X_train)
X_train = robustScaler.transform(X_train)
X_test = robustScaler.transform(X_test)

y_train = y_train.reset_index()
y_train = y_train.drop(columns=['index'])
y_test = y_test.reset_index()
y_test = y_test.drop(columns=['index'])
X_train = pd.DataFrame(X_train, columns=columns)
X_test = pd.DataFrame(X_test, columns=columns)
#
# # 4. 정규성 갖게 하기
y_train = y_train.apply(np.log)
# y_test = y_test.apply(np.log)
#
# # del encoder
# del robustScaler
# # 5. 모델 생성
# # 5.1 Linear Regression
# X_train = X_train.drop(columns=['city_부산광역시','가까운 스타벅스와의 거리','가까운 초등학교와의 거리'])
# X_test = X_test.drop(columns=['city_부산광역시','가까운 스타벅스와의 거리','가까운 초등학교와의 거리'])
# X_train = X_train.drop(columns=['city_부산광역시','가까운 스타벅스와의 거리'])
# X_test = X_test.drop(columns=['city_부산광역시','가까운 스타벅스와의 거리'])


# lr = linear_model.LinearRegression()
# model = lr.fit(X_train, y_train)
#
# y_predict = lr.predict(X_test)
#
# # 5.1.1 R Square value
# print('R square:', r2_score(y_test, y_predict))
# print('adjusted R square:',
#       1 - (1 - r2_score(y_test, y_predict)) * ((len(X_test) - 1) / (len(X_test) - len(X_test.columns) - 1)))
#
# # 5.1.2 RMSE
# y_predictions = lr.predict(X_train)
# print('train RMSE:',sqrt(mean_squared_error(y_train, y_predictions)))  # train RMSE score를 출력합니다.
# y_predictions = lr.predict(X_test)
# print('test RMSE:',sqrt(mean_squared_error(y_test, y_predictions)))  # test RMSE score를 출력합니다.

# 5.1.3
# 회귀 모델에서 다중공선성을 파악할 수 있는 대표적인 방법은 VIF이다.
#
# VIF (Variance inflation Factors 분산팽창요인)
#
# 안전 : VIF < 5
#
# 주의 : 5 < VIF < 10
#
# 위험 : 10 < VIF
# 1. 부산, 스타벅스 drop
# 2. 초등학교 drop




# del lr, columns, model, sqrt, y_train, X_test, y_test, y_predict, y_predictions
#
# vif = pd.DataFrame()
# vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
# vif["features"] = X_train.columns
# print(vif)

# X_train = sm.add_constant(X_train)
# model = sm.OLS(y_train, X_train).fit()
# print(model.summary())
#
# plt.scatter(y_test, y_predict, alpha=0.4)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("MULTIPLE LINEAR REGRESSION")
# # plt.axis([0, 700000, 0, 700000])
# plt.show()


# 5.2 Decision Tree 생성
# tree_1 = tree.DecisionTreeRegressor( max_depth=3, random_state=0)
# tree_1.fit(X_train, y_train)
# y_pred_tr = tree_1.predict(X_test)
#
# tree_2 = tree.DecisionTreeRegressor( max_depth=5, random_state=0)
# tree_2.fit(X_train, y_train)
# y_pred_tr_2 = tree_2.predict(X_test)
# # print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_tr))

# plt.figure()
# plt.scatter(X_train, y_train, s=20, edgecolor="black",
#             c="darkorange", label="data")
# plt.plot(X_test, y_pred_tr , color="cornflowerblue",
#          label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_pred_tr_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()
#
# for i in range(2,11):
#     tr_regressor_4 = DecisionTreeRegressor(criterion='poisson',random_state=10, max_depth=i)
#     tr_regressor_4.fit(X_train,y_train)
#
#     pred_tr = tr_regressor_4.predict(X_test)
#
#     decision_score_train=tr_regressor_4.score(X_train, y_train)
#     decision_score_test=tr_regressor_4.score(X_test, y_test)
#
#     expl_tr = explained_variance_score(pred_tr,y_test)
#
#     print("Decision tree  Regression Model",i," Train Score is ",round(decision_score_train*100))
#     print("Decision tree  Regression Model",i," Test Score is ",round(decision_score_test*100))
#     print('Explained Variance Score',i,'depth',expl_tr)

# param_grid = {
#     'criterion': ['mse','friedman_mse','mae'],
#     'max_depth': list(range(2,16)),
#     'min_samples_leaf': list(range(1,6)),
#     'min_samples_split': list(range(2,6)),
#     'random_state':[10]
# }
# clf = GridSearchCV(DecisionTreeRegressor(), param_grid, n_jobs=-1, cv=5)
# clf.fit(X_train, y_train)
#
# print(clf.best_params_)