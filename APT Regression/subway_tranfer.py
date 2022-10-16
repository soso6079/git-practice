import pandas as pd

apt = pd.read_pickle('분석 데이터_2.pickle')
subway = pd.read_pickle('지하철 위경도.pickle')
value = subway.loc[:,'역 이름'].value_counts()



def countTrans(df, value):

    if df['가까운 지하철 역 이름'] == 'error occur':
        return None
    else:
        count = value[df['가까운 지하철 역 이름']]

    return (count-1)

subway = apt.apply(lambda x: countTrans(x,value), axis=1)

apt['환승역'] = subway

apt.to_pickle('환승역 포함.pickle')