from Coin.download_data import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """ 파라미터 설명
    start = (2021,5,1)   원하는 데이터 기간의 시간순서의 앞 날짜
    end = (2021,5,5)     원하는 데이터 기간의 시간순서의 뒤 날짜
    minute_unit = 30     분단위 [1,3,5,15,10,30,60,240] 중 선택
    path = 'C:/coinDB'   다운로드 받을 위치
    usage = 'test'       용도(다운로드 받을 위치)
    unit = 'days'    시간단위 [minutes, days, weeks, months] 중 선택
    """
    start = (2021,5,1)
    end = (2021,5,5)
    minute_unit = 1
    path = 'coin_data'
    usage = 'test'
    unit = 'days'
    downloadData(start,end, minute_unit, path, usage, unit=unit)
