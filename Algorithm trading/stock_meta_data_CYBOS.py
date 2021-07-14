import win32com.client
import pandas as pd
rows = list()
CPE_MARKET_KIND = {'KOSPI':1, 'KOSDAQ':2}
instCpCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
for key, value in CPE_MARKET_KIND.items():
    codeList = instCpCodeMgr.GetStockListByMarket(value)
    for code in codeList:
        name = instCpCodeMgr.CodeToName(code)
        sectionKind = instCpCodeMgr.GetStockSectionKind(code)
        row = [code, name, key, sectionKind]
        rows.append(row)

print('모든 종목을 불러왔습니다')
stockitems = pd.DataFrame(data= rows, columns=['code','name', 'section','sectionKind'])
stockitems.loc[stockitems['sectionKind'] == 10, 'section'] = 'ETF'
stockitems.to_csv('stockitems.csv', index=False)
print('파일을 저장하였습니다.')

# 출처: https://ellun.tistory.com/324?category=475164 [Ellun's Library]