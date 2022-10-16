import json
import pandas as pd
import numpy as np
from urllib.request import urlopen
from urllib import parse
from urllib.request import Request
from urllib.error import HTTPError
import copy
from multiprocessing import Pool
import multiprocessing as mp
import os


def parallelize_df(df, func):
    """
    size가 큰 데이터를 병렬처리하는 함수
    :param df: 데이터
    :param func: 병렬 처리를 적용할 함수
    :return: 처리가 끝난 데이터
    """
    num_cores = 8
    df_split = np.array_split(df, num_cores)  # cpu 수만큼 데이터 분할
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))  # 분할한 데이터에 func 적용
    pool.close()
    pool.join()
    return df


def geocoding(data_addr):
    api_url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query="
    client_id = "vzi5bctu0u"
    client_pw = "ounHSxumFoRuXDU4tfnCtjvHoNFexPXQYCJuFHnG"

    gc_url = "?request=coordsToaddr&coords=127.116359386937,37.40612091848614&sourcecrs=epsg:4326&orders=admcode,legalcode,addr,roadaddr&output=json"

    naver_headers = {"X-NCP-APIGW-API-KEY-ID": "vzi5bctu0u",
                     "X-NCP-APIGW-API-KEY": "ounHSxumFoRuXDU4tfnCtjvHoNFexPXQYCJuFHnG"}
    geo_coordi = []
    long_list = []
    for addr in data_addr['addr_kr']:
        addr_urlenc = parse.quote(addr)  # 주소를 URL에서 사용할 수 있도록 encoding
        url = api_url + addr_urlenc
        request = Request(url)
        request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
        request.add_header("X-NCP-APIGW-API-KEY", client_pw)
        try:
            response = urlopen(request)
        except HTTPError as e:
            print('HTTP Error!')
            latitude = None
            longitude = None
        else:
            rescode = response.getcode()  # 정상이면 200 리턴
            if rescode == 200:
                response_body = response.read().decode('utf-8')
                response_body = json.loads(response_body)  # json
                if 'addresses' in response_body:  # 한글 주소 양식 문제로 결과가 나오지 않을 때
                    if response_body['addresses'] == []:

                        latitude = None
                        longitude = None
                    else:
                        latitude = response_body['addresses'][0]['y']
                        longitude = response_body['addresses'][0]['x']
                        print('Success!')
                else:
                    print("'result' not exist!")
                    latitude = None
                    longitude = None

            else:
                print('Response error code: %d' % rescode)
                latitude = None
                longitude = None
        geo_coordi.append([latitude, longitude])

    np_geo_coordi = np.array(geo_coordi)
    pd_geo_coordi = pd.DataFrame({'위도': np_geo_coordi[:, 0],
                                  '경도': np_geo_coordi[:, 1]})

    return pd_geo_coordi


if __name__ == '__main__':
    mp.freeze_support()
    data = pd.read_csv('train.csv')
    limit = 608276
    for i in range(30, 61):
        start = i * 10000
        end = (i + 1) * 10000
        if i == 60:
            end = limit
        data_result = parallelize_df(data[start:end], geocoding)

        data_result.reset_index(inplace=True, drop=True)
        result = pd.concat([data[start:end], data_result], axis=1)

        result.to_csv('result(' + str(end) + ').csv')
        data_result.to_csv('위도경도(' + str(end) + ').csv')
