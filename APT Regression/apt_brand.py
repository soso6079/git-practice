import pandas as pd

data = pd.read_pickle('분석 데이터.pickle')

# print(data[data['apt'].str.contains('래미안')])

def codingAPT(df):

    one = ['래미안', '힐스테이트','푸르지오','자이','이편한세상',
           'e편한세상','아이파크','I-PARK']
    two = ['롯데캐슬','롯데','더샵','꿈에그린']
    total = [one, two]
    brand_rank = {1:['래미안', '힐스테이트','푸르지오','자이','이편한세상',
                     'e편한세상','아이파크','I-PARK'],
                  2:['롯데캐슬','롯데','더샵','꿈에그린']}
    for i in brand_rank:
        tier = 0
        for j in brand_rank[i]:
            if tier == 0 or tier == 3:
                if j in df['apt']:
                    return i
                else:
                    return 3
            else:
                return tier

def codingAPT_2(df):

    one = ['래미안', '힐스테이트','푸르지오','자이','이편한세상',
           'e편한세상','아이파크','I-PARK']
    two = ['롯데캐슬','롯데','더샵','꿈에그린']
    total = [one, two]
    brand_rank = {2:['롯데캐슬','롯데','더샵','꿈에그린']}
    for i in brand_rank:
        tier = 0
        for j in brand_rank[i]:
            if tier == 0 or tier == 3:
                if j in df['apt']:
                    return i
                else:
                    return 3
            else:
                return tier

        # for index, j in enumerate(i):
        #     if j in df['apt']:
        #         return index
        #     else:
        #         return 2

data_2 = data.apply(codingAPT, axis=1)

data['브랜드 등급'] = data_2

first = data[data['브랜드 등급']==1]

data = data[data['브랜드 등급'] != 1]

data_3 = data.apply(codingAPT_2, axis=1)

data['브랜드 등급'] = data_3



look = pd.concat([first, data])

look



# data.to_pickle('아파트 등급.pickle')



brand_rank = {1:['래미안', '힐스테이트','푸르지오','자이','이편한세상',
                 'e편한세상','아이파크','I-PARK'],
              2:['롯데캐슬','롯데','더샵','꿈에그린']}

