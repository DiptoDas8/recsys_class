import pandas as pd
import time
import datetime
from bs4 import BeautifulSoup
import textstat
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler

import gender_guesser.detector as gender

d = gender.Detector ()

from geopy.geocoders import Nominatim

geolocator = Nominatim (user_agent="myapp")
import pycountry

scaler = MinMaxScaler ()

user_k_threshold = 5
item_k_threshold = 5


def ensure_min_k(engagement_metrics):
    print(engagement_metrics.shape)
    while (True):
        user_freq = engagement_metrics['UserId'].value_counts ()
        item_freq = engagement_metrics['ItemId'].value_counts ()
        if min (user_freq) < user_k_threshold:
            print(f'Less than {user_k_threshold} ratings for user')
            user_freq_values = user_freq[user_freq >= user_k_threshold].index
            engagement_metrics = engagement_metrics[engagement_metrics['UserId'].isin (user_freq_values)]
            print(engagement_metrics.shape)
        elif min (item_freq) < item_k_threshold:
            print(f'Less than {item_k_threshold} ratings for item')
            item_freq_values = item_freq[item_freq >= item_k_threshold].index
            engagement_metrics = engagement_metrics[engagement_metrics['ItemId'].isin (item_freq_values)]
            print(engagement_metrics.shape)
        else:
            break
    return engagement_metrics


def item_features(posts_df):
    '''
    This function prepares the item-features.csv file.
    :param posts_df: This is df read from posts.xlsx file.
    :return: a df with the following information: item id, feature name, numeric feature value
    '''
    questions_df = posts_df[posts_df['PostTypeId'] == 1]
    questions_df = questions_df[['Id', 'OwnerUserId', 'Score', 'ViewCount', 'Body', 'Title', 'Tags', 'AnswerCount',
                                 'CommentCount', 'FavoriteCount']]
    tags = set (pd.read_excel ('../data/Tags.xlsx')['@TagName'])
    print (tags)
    item_features = [['ItemId', 'FeatureName', 'FeatureValue']]
    for idx, row in questions_df.iterrows ():
        item_id = row['Id']
        for feature_name in ['OwnerUserId', 'Score', 'ViewCount', 'Body', 'Title', 'Tags', 'AnswerCount',
                             'CommentCount']:
            if feature_name == 'Tags':
                mytags = row[feature_name].replace ('><', ',').replace ('<', '').replace ('>', '').split (',')
                for tag in tags:
                    if tag in mytags:
                        item_features.append ([item_id, 'tag_' + tag, 1])
                    else:
                        item_features.append ([item_id, 'tag_' + tag, 0])
                continue
            if feature_name in ['Body', 'Title']:
                text = BeautifulSoup (row[feature_name], "lxml").text
                feature_value = len (text)
                item_features.append ([item_id, feature_name + '_length', feature_value])
                feature_value = textstat.flesch_reading_ease (text)
                item_features.append ([item_id, feature_name + '_readability', feature_value])
                continue
            else:
                try:
                    feature_value = int (row[feature_name])
                except:
                    feature_value = row[feature_name]
                item_features.append ([item_id, feature_name, feature_value])
    item_features_df = pd.DataFrame.from_records (item_features[1:], columns=item_features[0])
    item_features_df.to_csv ('../data/item-features.csv', index=False)
    return


def rating(posts_df):
    '''
    This fuction prepares the ratings.csv file.
    :param answers_df: This is df read from Posts.xlsx file
    :return: returns a df with columns user id, item id, numeric rating
    '''
    questions_df = posts_df[posts_df['PostTypeId'] == 1]
    answers_df = posts_df[posts_df['PostTypeId'] == 2]
    comments_df = pd.read_excel ('../data/Comments.xlsx')
    comments_df.columns = ['Comment_' + x.replace ('@', '') for x in comments_df.columns]
    comments_to_answers = pd.merge (comments_df, answers_df, left_on='Comment_PostId', right_on='Id')
    comments_to_answers = comments_to_answers[['Comment_UserId', 'ParentId', 'Comment_Text', 'Comment_CreationDate']]
    comments_to_answers.columns = ['UserId', 'ItemId', 'text_engagement', 'time']
    comments_to_questions = pd.merge (comments_df, questions_df, left_on='Comment_PostId', right_on='Id')
    comments_to_questions = comments_to_questions[['Comment_UserId', 'Id', 'Comment_Text', 'Comment_CreationDate']]
    comments_to_questions.columns = ['UserId', 'ItemId', 'text_engagement', 'time']
    answers_to_questions = answers_df[['OwnerUserId', 'ParentId', 'Body', 'CreationDate']]
    answers_to_questions.columns = ['UserId', 'ItemId', 'text_engagement', 'time']
    engagement_df = pd.concat ([answers_to_questions, comments_to_questions, comments_to_answers]).sort_values (
        ['UserId',
         'ItemId'])
    engagement_df = engagement_df[~engagement_df[['UserId', 'ItemId']].apply (tuple, 1).isin (
        questions_df[['OwnerUserId', 'Id']].apply (tuple, 1))]
    engagement_df = engagement_df[engagement_df['UserId'] > 0]
    engagement_df['time'] = engagement_df['time'].apply (
        lambda x: time.mktime (datetime.datetime.strptime (x, '%Y-%m-%dT%H:%M:%S.%f').timetuple ()))

    engagement_frequency = engagement_df[['UserId', 'ItemId']].value_counts (ascending=True).reset_index (
        name='frequency')
    # print (engagement_frequency.shape)

    engagement_volume = engagement_df.groupby (['UserId', 'ItemId'])['text_engagement'].apply (' '.join).apply (
        lambda x: len (BeautifulSoup (x, 'lxml').text)).reset_index (name='aggregated_text_length')
    # print (engagement_volume.shape)

    engagement_urgency = engagement_df.groupby (['UserId', 'ItemId'])['time'].apply (min).apply (
        lambda x: 1 / x).reset_index (name='response_urgency')
    # print (engagement_urgency.shape)

    engagement_metrics = pd.merge (pd.merge (engagement_frequency, engagement_volume, on=['UserId', 'ItemId']),
                                   engagement_urgency, on=['UserId', 'ItemId'])
    # print (engagement_metrics.head (10))
    # print (engagement_metrics.shape)

    engagement_metrics[['frequency']] = engagement_metrics[['frequency']].apply (expit)
    engagement_metrics[['aggregated_text_length']] = engagement_metrics[['aggregated_text_length']].apply (np.log)
    engagement_metrics[['response_urgency']] = scaler.fit_transform (engagement_metrics[['response_urgency']])
    engagement_metrics.to_excel ('../data/scaled_engagement_metrics.xlsx', index=False)

    engagement_metrics['Rating'] = 0.3 * engagement_metrics['frequency'] + 0.5 * engagement_metrics[
        'aggregated_text_length'] + 0.2 * engagement_metrics['response_urgency']
    engagement_metrics[['Rating']] = MinMaxScaler (feature_range=(1, 10)).fit_transform (engagement_metrics[['Rating']])

    engagement_metrics = engagement_metrics[['UserId', 'ItemId', 'Rating']]
    engagement_metrics = ensure_min_k (engagement_metrics)

    engagement_metrics.to_csv ('../data/ratings.csv', index=False)
    return engagement_metrics[['UserId', 'ItemId', 'Rating']]


def user_features(users_df, rating_df):
    '''
    This function prepares the user-features.csv file.
    :param users_df: This is df read from users.xlsx
    :param rating_df: This is the processed ratings file
    :return: a df with the following information: user id, feature name, numeric feature value
    '''
    retained_users = rating_df['UserId'].unique ().tolist ()
    print (len (retained_users))
    users_df = users_df[users_df['Id'].isin (retained_users)]
    print (users_df.columns)

    genders = {'unknown': 0, 'male': 1, 'mostly_male': 2, 'female': 3, 'mostly_female': 4, 'andy': 5}
    users_df['Gender'] = users_df['DisplayName'].apply (lambda x: genders[d.get_gender (x)])
    countries = []
    for idx, row in users_df.iterrows ():
        if str (row['Location']) == 'nan':
            countries.append ('')
        else:
            print (row['Location'])
            try:
                location = geolocator.geocode (row['Location']).raw
                lat, lon = location['lat'], location['lon']
                country_alpha_2 = geolocator.reverse ([lat, lon]).raw['address']['country_code']
                country = pycountry.countries.get (alpha_2=country_alpha_2).numeric
                print ('>>>>>>', country_alpha_2, country)
                countries.append (country)
            except Exception as e:
                print (e)
                countries.append ('')
    users_df['Country'] = countries
    users_df = users_df[['Id', 'Reputation', 'Views', 'UpVotes', 'DownVotes', 'Gender', 'Country']]
    print (users_df.head (100))

    user_features = [['UserId', 'FeatureName', 'FeatureValue']]

    for idx, row in users_df.iterrows ():
        userId = row['Id']
        for feature_name in ['Reputation', 'Views', 'UpVotes', 'DownVotes', 'Gender', 'Country']:
            numeric_feature_value = row[feature_name]
            # print(userId, feature_name, numeric_feature_value)
            user_features.append ([userId, feature_name, numeric_feature_value])
    user_features_df = pd.DataFrame.from_records (user_features[1:], columns=user_features[0])
    user_features_df.to_csv ('../data/user-features.csv', index=False)
    return


posts_df = pd.read_excel ('../data/Posts.xlsx')
posts_df.columns = [x.replace ('@', '') for x in posts_df.columns]
item_features (posts_df)

# rating_df = rating (posts_df)
#
# users_df = pd.read_excel ('../data/Users.xlsx')
# users_df.columns = [x.replace ('@', '') for x in users_df.columns]
# user_features (users_df, rating_df)
