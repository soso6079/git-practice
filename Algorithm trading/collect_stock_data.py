import FinanceDataReader as fdr
import pandas as pd

# df_kospi = fdr.StockListing('KOSPI')
# df_kosdaq = fdr.StockListing('KOSDAQ')
#
# print(df_kospi)
# print(df_kosdaq)
# 캔들차트 그리기
df = fdr.DataReader('005930', '2021-01-01')
df.to_pickle()
print(df)


def Stochastic(df: pd.DataFrame, n: int, m: int, t: int, slow=True) -> pd.DataFrame:
    # 입력받은 값이 dataframe이라는 것을 정의해줌
    df = pd.DataFrame(df)

    # n일중 최고가
    ndays_high = df.High.rolling(window=n, min_periods=1).max()

    # n일중 최저가
    ndays_low = df.Low.rolling(window=n, min_periods=1).min()

    # Fast%K 계산
    fast_k = ((df.Close - ndays_low) / (ndays_high - ndays_low)) * 100

    # Fast%D (=Slow%K) 계산
    slow_k = fast_k.ewm(span=m).mean()

    # Slow%D 계산
    slow_d = slow_k.ewm(span=t).mean()

    # dataframe으로 리턴
    if slow == False:  # Stochastic Fast일 때
        result = pd.DataFrame(data={'fast_k': fast_k, 'fast_d': slow_k})

    else:
        result = pd.DataFrame(data={'slow_k': slow_k, 'slow_d': slow_d})

    return result
