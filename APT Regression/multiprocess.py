import json
import pandas as pd
import numpy as np
from urllib.request import urlopen
from urllib import parse
from urllib.request import Request
from urllib.error import HTTPError
import copy
from multiprocessing import Pool
import os

data = pd.read_csv('train.csv')

def parallelize_df(df, func):
    print(1)
    num_cores = 4
    print(2)
    df_split = np.array_split(df, num_cores)
    print(df_split)
    pool = Pool(num_cores)
    print(3)
    df = pd.concat(pool.map(func, df_split))
    print()
    pool.close()
    pool.join()
    return df

def ppp(data):
    print('PID:', os.getpid())
    print(data)
    return data

def geocoding(data_addr):
    print('PID:', os.getpid())
    api_url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query="
    client_id = "vzi5bctu0u"
    client_pw = "ounHSxumFoRuXDU4tfnCtjvHoNFexPXQYCJuFHnG"

    gc_url = "?request=coordsToaddr&coords=127.116359386937,37.40612091848614&sourcecrs=epsg:4326&orders=admcode,legalcode,addr,roadaddr&output=json"

    naver_headers = {"X-NCP-APIGW-API-KEY-ID": "vzi5bctu0u",
                     "X-NCP-APIGW-API-KEY": "ounHSxumFoRuXDU4tfnCtjvHoNFexPXQYCJuFHnG"}
    geo_coordi = []
    error_list = []
    for index, addr in enumerate(data_addr):
        addr_urlenc = parse.quote(addr) # 주소를 URL에서 사용할 수 있도록 encoding
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
            rescode = response.getcode() # 정상이면 200 리턴
            if rescode == 200:
                response_body = response.read().decode('utf-8')
                response_body = json.loads(response_body)   #json
                if 'addresses' in response_body:
                    if response_body['addresses'] == []:
                        error_list.append(index)
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
                print('Response error code: %d' %rescode)
                latitude = None
                longitude = None
        geo_coordi.append([latitude, longitude])

    np_geo_coordi = np.array(geo_coordi)
    pd_geo_coordi = pd.DataFrame({'위도': np_geo_coordi[:,0],
                                  '경도': np_geo_coordi[:,1]})

    return pd_geo_coordi

data_addr = data['addr_kr']

data_result = parallelize_df(data_addr, ppp)