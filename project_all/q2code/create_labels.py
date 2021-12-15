from pprint import pprint
import pandas as pd
accepted = []
with open('../data/skeptics.accepted.answers.csv', 'r') as fin:
    for id in fin.readlines():
        accepted.append(id.strip())
print(accepted)

all = [['id', 'accepted_class']]
with open('../data/skeptics.all.answers.csv', 'r') as fin:
    for id in fin.readlines():
        id = id.strip()
        if id in accepted:
            all.append([id, '1'])
        else:
            all.append ([id.strip (), '0'])

df = pd.DataFrame.from_records(all)
df.to_excel('../data/label.skeptics.answers.xlsx', encoding='utf-8', index=False)
'''remove the first 0-1 row in the created file'''
