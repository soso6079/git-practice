import pandas as pd

df = pd.read_csv('result(10000).csv')
df_train = pd.read_csv('train.csv')

print(df)


c = apt_df_local[:1].apply(lambda row: target_df.apply
                        (lambda row_2: row_2, axis=1) + haversine(row_2['지하철 위도'],row_2['지하철 경도']),(row['위도'],row['경도']), unit='m')), axis=1))