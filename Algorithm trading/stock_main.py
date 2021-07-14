import win32com.client

'''
대신증권 크레온 API: 시세 조회의 경우 15초에 60건, 
실시간 조회의 경우 최대 400건으로 제한되어 있고, 
주문 관련 조희의 경우 15초에 최대 20건으로 제한됨.
'''

instCpCybos = win32com.client.Dispatch("CpUtil.CpCybos")
print(instCpCybos.IsConnect)
