import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_recommend_exercise_list(df, exercise_part, top=30):

    #코사인 유사도를 구한 벡터
    count_vector = CountVectorizer(ngram_range=(1, 2))
    c_vector_part = count_vector.fit_transform(df['part'])
    part_c_sim = cosine_similarity(c_vector_part, c_vector_part).argsort()[:, ::-1]


    # 자극 부위가 비슷한 운동을 추천해야 하기 때문에 '자극 부위' 정보를 뽑아낸다.
    target_exercise_index = df[df['part'] == exercise_part].index.values
    #코사인 유사도 중 비슷한 코사인 유사도를 가진 정보를 뽑아낸다.
    sim_index = part_c_sim[target_exercise_index, :top].reshape(-1)
    result = df.iloc[sim_index][:3]
    
    return result
