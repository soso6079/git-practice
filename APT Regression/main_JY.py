from prepare_contentsfiltering import prepare_contentsfiltering
from get_recommend_exercise_list import get_recommend_exercise_list
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



candidate_exercise, target_part = prepare_contentsfiltering()
print(candidate_exercise)
print(target_part)
print("a")

result = get_recommend_exercise_list(candidate_exercise, exercise_part=target_part)
print(result)