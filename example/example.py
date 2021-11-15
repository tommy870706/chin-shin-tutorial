# -*- coding: UTF-8 -*-
#import 套件
import requests
#發送request去取資料
res = requests.get('https://jsonplaceholder.typicode.com/todos/1')
#查看request是否成功
if res.status_code==200:
    print('request 發送成功')
else:
    print('有些錯誤產生')
#把資料印出來看長怎樣
print(res.json())