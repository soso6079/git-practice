import pandas as pd
import numpy as np

start = 10000
end = 1250000

initial_df = pd.DataFrame(columns=['위도','경도','가까운 지하철 역과의 거리','가까운 지하철 역 이름'])

for i in range(start,end,10000):

    if i == 1240000:
        last = 1234827
        df = pd.read_pickle('save\가까운 지하철('+str(last)+').pickle')
    else:
        df = pd.read_pickle('save\가까운 지하철('+str(i)+').pickle')

    if i == start:
        result = pd.concat([initial_df, df], ignore_index=True)

    else:
        result = pd.concat([result, df], ignore_index=True)

result.to_pickle('지하철_거리.pickle')

    # np.genfromtxt('위도경도('+str(i)+').csv')