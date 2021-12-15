import pandas as pd
from pprint import pprint

labels_df = pd.read_excel ('../data/label.skeptics.answers.xlsx')
print (labels_df.columns)
print (labels_df.shape)
textual_df = pd.read_excel ('../data/textual.features.skeptics.answers.body.xlsx')
print (textual_df.columns)
print (textual_df.shape)
metric_df = pd.read_excel ('../data/metric.features.skeptics.answers.with.parentid.xlsx')
print (metric_df.columns)
print (metric_df.shape)
liwc_df = pd.read_excel('../data/liwc.features.skeptics.answers.body.xlsx')
print(liwc_df.columns)
print(liwc_df.shape)

merged_df = pd.merge(pd.merge (pd.merge (textual_df, metric_df, on='id'), labels_df, on='id'), liwc_df, on='id')
print (merged_df.shape)

merged_df.to_excel ('../data/merged.skeptics.answers.xlsx', index=False)
# merged_df.to_csv('merged.skeptics.answers.csv', index=False)


def rename_col(df, x):
    need_rename = ['pronoun', 'present', 'past', 'future', 'sentiment',
                   'affermative', 'negative', 'interrogative', 'imperative', 'optative',
                   'exclamation', 'readability', 'sentence_count',
                   'words_count_per_sentence', 'proper_noun', 'adjective',
                   'cardinal_number', 'subjectivity', 'url_img_count']
    for col in df.columns:
        if col in need_rename:
            df.rename (columns={col: x + '_' + col}, inplace=True)

print('***')
qtitles_t_df = pd.read_excel ('../data/textual.features.skeptics.questions.title.xlsx')
print(qtitles_t_df.shape)
'''
many titles do not have enough words to be processed by liwc. so the qtitles_l_df has only 330 rows. 
qtitles_l_df = pd.read_excel('../data/liwc.features.skeptics.questions.title.xlsx')
print(qtitles_l_df.shape)
qtitles_df = pd.merge(qtitles_t_df, qtitles_l_df, on='id')
'''
qtitles_df = qtitles_t_df
rename_col (qtitles_df, 'title')
print (qtitles_df.columns)
print (qtitles_df.shape)
qbody_t_df = pd.read_excel ('../data/textual.features.skeptics.questions.body.xlsx')
qbody_l_df = pd.read_excel('../data/liwc.features.skeptics.questions.body.xlsx')
qbody_df = pd.merge(qbody_t_df, qbody_l_df, on='id')
rename_col (qbody_df, 'body')
print (qbody_df.columns)
print (qbody_df.shape)
q_met_df = pd.read_excel ('../data/metric.features.skeptics.questions.xlsx')
print (q_met_df.columns)
print (q_met_df.shape)

merged_df = pd.merge (pd.merge (qtitles_df, qbody_df, on='id'), q_met_df, on='id')
print (merged_df.shape)
merged_df.to_excel ('../data/merged.skeptics.questions.xlsx', index=False)
# merged_df.to_csv('merged.skeptics.questions.csv', index=False)

