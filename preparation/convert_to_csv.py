import xmltodict
import os
import pandas as pd
from pprint import pprint

datafiles = [x.split('.')[0] for x in os.listdir('../data/') if x.endswith('.xml')]

for datafile in datafiles:
    with open('../data/'+datafile+'.xml') as xml_fp:
        data_dict = xmltodict.parse(xml_fp.read())
    # pprint(data_dict)
    key = datafile.lower()
    data_dict = [dict(x) for x in data_dict[key]['row']]
    df = pd.DataFrame.from_records(data_dict)
    # print(df.head())
    df.to_excel('../data/'+datafile+'.xlsx', index=False, encoding='utf-8')
