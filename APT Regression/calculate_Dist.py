import pandas as pd
from haversine import haversine
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
import time
import datetime
import os

def parallelize_df(df, func):
    num_cores = mp.cpu_count()
    df_split = np.array_split(df, num_cores)  # cpu 수만큼 데이터 분할
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split), ignore_index= True)  # 분할한 데이터에 func 적용
    pool.close()
    pool.join()

    return df

def calcul(apt_df_local):
    target_df = pd.read_pickle('지하철 위경도.pickle')    #TODO 아파트와 거리 계산할 타겟 데이터

    result_distance = []
    result_name = []

    lat = np.array(apt_df_local['위도'].tolist())
    lng = np.array(apt_df_local['경도'].tolist())

    for i in range(len(lat)):
        def getDistance(df):
            return haversine((df['지하철 위도'],df['지하철 경도']), #TODO 타겟 데이터의 위/경도 칼럼 이름
                             (lat[i], lng[i]), unit='m')

        target_df['거리'] = target_df.apply(getDistance, axis=1)

        min_distance = target_df['거리'].min()

        result_distance.append(min_distance)
        if len(target_df[target_df['거리'] == min_distance]['역 이름']) == 0:
            result_name.append('error occur')
        else:
            result_name.append(target_df[target_df['거리'] == min_distance]['역 이름'].iloc[0])  #TODO 지하철 외엔 안해도 될듯

    apt_df_local['가까운 지하철 역과의 거리'] = result_distance    #TODO 데이터에 맞게 칼럼 이름 수정
    apt_df_local['가까운 지하철 역 이름'] = result_name

    return apt_df_local


    #     target_df['거리'] = target_df.apply(getDistance, axis=1)
    # for row in apt_df_local.itertuples():
    #     def getDistance(df):
    #         return haversine((df['지하철 위도'],df['지하철 경도']), #TODO 타겟 데이터의 위/경도 칼럼 이름
    #                          (row[0], row[1]), unit='m')
    #
    #     target_df['거리'] = target_df.apply(getDistance, axis=1)
    #
    #     min_distance = target_df['거리'].min()
    #
    #     result_distance.append(min_distance)
    #     if len(target_df[target_df['거리'] == min_distance]['역 이름']) == 0:
    #         result_name.append('error occur')
    #     else:
    #         result_name.append(target_df[target_df['거리'] == min_distance]['역 이름'].iloc[0])  #TODO 지하철 외엔 안해도 될듯
    #
    # apt_df_local['가까운 지하철 역과의 거리'] = result_distance    #TODO 데이터에 맞게 칼럼 이름 수정
    # apt_df_local['가까운 지하철 역 이름'] = result_name
    #
    # return apt_df_local

if __name__ == '__main__':
    mp.freeze_support()

    if not os.path.exists('save2'):
        os.makedirs('save2')

    start_time = time.time()

    apt_df = pd.read_pickle('아파트_전체.pickle')

    limit = 1234828 #TODO 아파트 전체 값 수

    for i in range(124):

        start = i * 10000
        end = (i + 1) * 10000
        if i == 123:
            end = limit
        data_result = parallelize_df(apt_df[start:end], calcul)

        data_result.to_pickle('save2\가까운 지하철('+str(end)+').pickle')  #TODO 데이터에 맞게 파일명 수정

        sec = time.time()-start_time
        times = str(datetime.timedelta(seconds=sec)).split(".")
        times = times[0]
        print(times)







