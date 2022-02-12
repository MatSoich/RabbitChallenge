
# coding: utf-8

# In[ ]:


import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import urllib
import time
from datetime import datetime, timedelta
from pathlib import Path

def get_data_url(data_nums, pid, rno):
    base = 'hoge'
    baseAdd1 = "/fuga/"
    baseAdd2 = "/piyo/" + pid + "/hogera/" + rno + "/"
    url = base + baseAdd1 + data_nums + baseAdd2
    return urllib.parse.quote_plus(url, "/:?=&")

def date_span(start_date, end_date):
    """start_date、end_dateの期間に含まれる日毎のdatetimeオブジェクトを返すジェネレータ
    """
    for i in range((end_date - start_date).days + 1):
        yield start_date + timedelta(i)

start_date = datetime.strptime('20170822', '%Y%m%d')
end_date = datetime.strptime('20170823', '%Y%m%d')

today = datetime.today()
date_prefix = today.strftime('%Y-%m-%d %H:%M:%S')
file_origin = 'crawl_' + date_prefix + '_Crawler.csv'

my_file = Path(file_origin)
isFirst = True

if my_file.is_file():
    mainDf = pd.read_csv(file_origin, index_col=0, header=0)
    isFirst = False
else:
    mainDf = None
    
place_ids = []
race_noms = []
for i in range(1, 31):
    conv_num = str(i).zfill(2)
    place_ids.append(conv_num)
    if i <= 12:
        race_noms.append(conv_num) 

sleep_time = 5

for target_date in date_span(start_date, end_date):
    for pid in place_ids:
        for rno in race_noms:
            target_url = get_data_url(target_date.strftime('%Y%m%d'), pid, rno)
            date = target_date.strftime('%s')
            headers = {'User-Agent': 'Mozilla/5.0'}
            time.sleep(sleep_time)
            response = requests.get(target_url, headers=headers)# <Response [200]>
            soup = BeautifulSoup(response.text, 'html.parser')
            content1 = soup.find_all("h1", class_="h_content1")
            content2 = soup.find_all("div", class_="blocks")
            content3 = soup.find_all("table", id="detail_program")
            for content3_part in content3:
                tds1 = [td.find_all("p") for td in content3_part.find_all("td", class_="border_all name_kanji")]
                ID = [row[0].string for row in tds1]
                name = [row[1].string for row in tds1]
                prefecture = [row[2].string.split("/") for row in tds1]
                age = [row[3].string.split("/") for row in tds1]
                ages = [row[0].replace('歳', '') for row in age]
                term = [row[1].replace('期', '') for row in age]
                prefectures = [row[0].replace('北海道', '1').replace('青森', '2').replace('岩手', '3').replace('宮城', '4').replace('秋田', '5').replace('山形', '6').replace('福島', '7').replace('茨城', '8').replace('栃木', '9').replace('群馬', '10').replace('埼玉', '11').replace('千葉', '12').replace('東京', '13').replace('神奈川', '14').replace('新潟', '15').replace('富山', '16').replace('石川', '17').replace('福井', '18').replace('山梨', '19').replace('長野', '20').replace('岐阜', '21').replace('静岡', '22').replace('愛知', '23').replace('三重', '24').replace('滋賀', '25').replace('京都', '26').replace('大阪', '27').replace('兵庫', '28').replace('奈良', '29').replace('和歌山', '30').replace('鳥取', '31').replace('島根', '32').replace('岡山', '33').replace('広島', '34').replace('山口', '35').replace('徳島', '36').replace('香川', '37').replace('愛媛', '38').replace('高知', '39').replace('福岡', '40').replace('佐賀', '41').replace('長崎', '42').replace('熊本', '43').replace('大分', '44').replace('宮崎', '45').replace('鹿児島', '46').replace('沖縄', '47') for row in prefecture]
                rank = [row[1].replace('A1', '1').replace('A2', '2').replace('B1', '3').replace('B2', '4') for row in prefecture]
                tds2 = [td.text for td in content3_part.find_all("td", class_="border_all average")[2:][::5]]
                morter_num = [td[:-5] for td in tds2]
                for content2_part in content2:
                    race_number = [p.text.replace('【', '').replace('】', '').replace('予選', '').replace('R', '') for p in content2_part.find_all("p", class_="left")]
                    for num in race_number:
                        race_number = [num for i in range(6)]
                        race_number = [num[:3] for num in race_number]
                for content1_part in content1:
                    race_place = [n[:3] for n in content1_part]
                    race_place = race_place * 6
                raceDict = pd.DataFrame({"date": date,                                         
                                        "ID": ID,
                                        "name": name,
                                        "age": ages,
                                        "term": term,
                                        "prefecture": prefectures,
                                        "rank": rank,
                                        "morter_num": morter_num,
                                        "race_number": race_number,
                                        "race_place": race_place
                                         })

                # 取得した後、追加していく
                if mainDf is None:
                    raceDict.to_csv(file_origin)
                    mainDf = raceDict
                else:
                    raceDict.to_csv(file_origin, mode='a', header=False)
                    mainDf = mainDf.append(raceDict)    

