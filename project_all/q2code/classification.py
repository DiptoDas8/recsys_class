import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.linear_model import LinearRegression, LogisticRegression
from pprint import pprint

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter ("ignore")

ansdf = pd.read_excel ('../data/merged.skeptics.answers.xlsx')
ansdf.rename (columns={'parentid': 'quesid', 'score': 'ans_score'}, inplace=True)
print (ansdf.shape)
# print(ansdf.columns)

semantic_df = pd.read_excel ('../data/skeptics.answers.semantic.similarity.xlsx')

quesdf = pd.read_excel ('../data/merged.skeptics.questions.xlsx')
quesdf.rename (columns={'id': 'quesid', 'score': 'ques_score'}, inplace=True)
print (quesdf.shape)


# print(quesdf.columns)

def feature_selection(features, X, y):
    print ('infogain')
    selector = mutual_info_classif (X, y, discrete_features=True)
    top_features_names = list (zip (features, selector))
    top_features_names.sort (key=lambda tup: tup[1], reverse=True)
    print (top_features_names[:10])

    # print ('chi2')
    # selector = chi2 (X, y)
    # top_features_names = list (zip (features, selector[1]))
    # top_features_names.sort (key=lambda tup: tup[1], reverse=False)
    # print (top_features_names[:10])
    #
    # print ('f_classif')
    # selector = f_classif (X, y)
    # top_features_names = list (zip (features, selector[1]))
    # top_features_names.sort (key=lambda tup: tup[1], reverse=False)
    # print (top_features_names[:10])

    return


def classification(dataset, normalize=True, smote=True):
    all_features = [f for f in dataset.columns if f not in ('id', 'quesid', 'accepted_class')]
    features = all_features[:]
    smote = SMOTE(random_state=42)
    # print (features)
    X, y = dataset[features], dataset['accepted_class']
    # print(y.value_counts())
    if smote:
        X, y = smote.fit_resample(X, y)
        # print(y.value_counts())
    y = y.astype ('int')
    x = X.values
    min_max_scaler = preprocessing.MinMaxScaler ()
    if normalize:
        x = min_max_scaler.fit_transform (x)
    X = pd.DataFrame (x)
    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=1.0 / 10, random_state=1)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    classifiers = {
        'adaboost': AdaBoostClassifier (DecisionTreeClassifier (max_depth=1), algorithm='SAMME', n_estimators=5),
        'bagging': BaggingClassifier (KNeighborsClassifier (), max_samples=0.5, max_features=0.5),
        'gaussiannb': GaussianNB (),
        'gradientboosting': GradientBoostingClassifier (n_estimators=300, learning_rate=1.0, max_depth=1,
                                                        random_state=18),
        'histgradientboosting': HistGradientBoostingClassifier (max_iter=500),
        'mlp': MLPClassifier (),
        'multinomialnb': MultinomialNB (),
        'randomforest': RandomForestClassifier (n_estimators=150, max_depth=5),
        'sgd': SGDClassifier (loss="hinge", penalty="l2", max_iter=150),
        'svm': svm.SVC (kernel='poly', degree=5, C=1),
    }

    result = {}

    '''
       performing_classifiers = ['adaboost', 'gradientboosting', 'histgradientboosting', 'mlp', 'randomforest']
    '''

    for key in ['adaboost', 'gradientboosting', 'histgradientboosting', 'mlp', 'randomforest']:
        # break
        classifier = classifiers[key]
        model = classifier.fit (X_train, y_train)
        y_pred = model.predict (X_test)
        result[key] = {
            # 'accuracy': accuracy_score(y_test, y_pred),
            # 'precision': precision_score(y_test, y_pred),
            # 'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
    pprint (result)

    # feature_selection (features, X, y)
    return


print ('only answers features')
# classification (ansdf)
print ('answers and semantic similarities')
ans_semantic_df = pd.merge (ansdf, semantic_df, on='id')
# classification (ans_semantic_df)
print ('answers, semantic similarities, and questions')
all_df = pd.merge (ans_semantic_df, quesdf, on='quesid')
# classification (all_df)
all_df.to_csv('../../topic/result/ans_question_semantic.csv', index=False)
print('answers and questions')
all_df = pd.merge(ansdf, quesdf, on='quesid')
# classification(all_df)

print('used question topics')
topics = pd.read_csv('../../topic/data/tags_one_hot_encodings.csv')
all_df = pd.merge(all_df, topics, on='quesid')
# classification(all_df)
# all_df.to_csv('../../topic/result/all.csv', index=False)

# print('**********without normalization************')
# print('only answers features')
# classification(ansdf, False)
# print('answers features and semantic similarities')
# ans_semantic_df = pd.merge(ansdf, semantic_df, on='id')
# classification(ans_semantic_df, False)
# # all_df = pd.merge(ansdf, quesdf, on='quesid')
# # classification(all_df)
# print('answers, semantic similarity, questions')
# all_df = pd.merge(ans_semantic_df, quesdf, on='quesid')
# classification(all_df, False)
