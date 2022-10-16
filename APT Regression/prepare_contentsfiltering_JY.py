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


def prepare_contentsfiltering():
    user_data = pd.read_pickle('./user_random.pickle')
    feedback_data = pd.read_csv('./feeback_random.csv')
    exercise_data = pd.read_csv('./exercise2.csv')

    #운동 데이터셋 전처리
    exercise_data['part'] = exercise_data['part_1'].str.cat(exercise_data['part_2'], sep=' ', na_rep = '')
    exercise_data['part'] = exercise_data['part'].str.strip()
    exercise_data.drop(['part_1', 'part_2', 'part_3'], axis = 1, inplace = True)
    strength_exercise = exercise_data[exercise_data['keyword'] == 'strength']
    diet_exercise = exercise_data[exercise_data['keyword'] == 'diet']

    #피드백 데이터 전처리
    feedback_data = feedback_data.transpose()
    feedback_data = feedback_data.drop(feedback_data.index[0])
    feedback_data = feedback_data.rename(columns=feedback_data.iloc[0])
    feedback_data = feedback_data.drop(feedback_data.index[0])

    diet_feedback = feedback_data.filter(like='diet', axis=0)    #다이어트 운동 피드백 데이터셋
    strength_feedback = feedback_data[~feedback_data.index.str.contains("diet")]    #근력 운동 피드백 데이터셋



    #유저 id / 운동 목적 DB에서 가져오기(현재 임의로 설정함)
    user_id = 1    
    purpose = 'strength' 

    if purpose == 'strength':
        purpose_feedback = strength_feedback
        purpose_exercise = strength_exercise
    else:
        purpose_feedback = diet_feedback
        purpose_exercise = diet_exercise
        


    #사용자 평점 없는 운동만 추출    
    null_feedback = purpose_feedback[purpose_feedback[user_id].isnull()] 
    null_feedback = null_feedback[user_id].to_frame()
    null_feedback = null_feedback.reset_index()
    null_feedback = null_feedback.rename(columns = {'index':'name'})

    ####################필독!!##########################
    #현재 데이터셋에서는 아래 코드 돌려야함. 이후 운동 데이터셋에서 스트레칭 운동 삭제하면 아래 코드도 삭제
    null_feedback = null_feedback.drop([7,11]) #스트레칭 데이터 임의로 삭제



    #사용자 평점 높은 운동 20개 추출
    feedback_data = feedback_data.apply(pd.to_numeric, errors = 'coerce').fillna(0)
    satisfied_exercise = feedback_data.nlargest(20, user_id)[user_id].index


    #사용자 평점 높은 운동 중 랜덤하게 자극부위 선택 
    random_num = random.randrange(0,20)

    for i in range(0, len(exercise_data)-1):
        if exercise_data.name[i] == satisfied_exercise[random_num]:
            target_part = exercise_data.part[i] #만족도 높은 자극 부위
            break
            


    user_feedback = null_feedback
    #운동 목적에 부합하며, 평점이 없는 운동 데이터셋 생성
    candidate_exercise = pd.merge(purpose_exercise, user_feedback, how='right', 
                                on = 'name', left_index=False, right_index=True)
    candidate_exercise = candidate_exercise.reset_index(drop=True)
    candidate_exercise['difficulty'] = candidate_exercise['difficulty'].astype(int)
    candidate_exercise['is_time'] = candidate_exercise['is_time'].astype(int)


    # 보유 장비 점검
    needed_tools = candidate_exercise['tools'] #운동 후보에 필요한 장비
    needed_tools.dropna(inplace = True)

    user_tools = user_data['tool'][user_id] #유저가 보유한 장비
    candidate_exercise['tools'] = candidate_exercise['tools'].astype(str)

    for item in needed_tools:
        if item in user_tools: 
            continue
        else: #필요 장비 없으면 운동 후보에서 제외
            candidate_exercise = candidate_exercise[candidate_exercise.tools != item]
        
            
    return candidate_exercise, target_part
