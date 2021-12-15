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

# prehoc_features_only = True

def item_features(posts_df):
    '''
    This function prepares the item-features.csv file.
    :param posts_df: This is df read from posts.xlsx file.
    :return: a df with the following information: item id, feature name, numeric feature value
    '''
    questions_df = posts_df[posts_df['PostTypeId'] == 1]
    prehoc_features = ['OwnerUserId', 'Body', 'Title', 'Tags']
    adhoc_features = ['CommentCount']
    posthoc_features = ['Score', 'ViewCount', 'AnswerCount', 'FavoriteCount']
    # features = prehoc_features
    features = prehoc_features+adhoc_features+posthoc_features
    questions_df = questions_df[['Id']+features]
    tags = set (pd.read_csv('../../raw_data/Tags.csv')['TagName'])
    print (tags)
    item_features = [['ItemId', 'FeatureName', 'FeatureValue']]
    for idx, row in questions_df.iterrows ():
        item_id = str(row['Id'])
        for feature_name in prehoc_features:
            if feature_name == 'Tags':
                mytags = row[feature_name].replace ('><', ',').replace ('<', '').replace ('>', '').split (',')
                for tag in tags:
                    if tag in mytags:
                        item_features.append ([item_id, 'tag_' + tag, 1])
                    else:
                        item_features.append ([item_id, 'tag_' + tag, 0])
                continue
            elif feature_name in ['Body', 'Title']:
                text = BeautifulSoup (row[feature_name], "lxml").text
                feature_value = len (text)
                item_features.append ([item_id, feature_name + '_length', feature_value])
                feature_value = textstat.flesch_reading_ease (text)
                item_features.append ([item_id, feature_name + '_readability', feature_value])
                continue
            elif feature_name=='OwnerUserId':
                feature_value = str(row[feature_name])
                item_features.append ([item_id, feature_name, feature_value])
            else:
                try:
                    feature_value = int (row[feature_name])
                except:
                    feature_value = row[feature_name]
                item_features.append ([item_id, feature_name, feature_value])
    item_features_df = pd.DataFrame.from_records (item_features[1:], columns=item_features[0])
    item_features_df.to_csv ('../../raw_data/item-features.csv', index=False)
    return

def rating(posts_df):
    '''
    This fuction prepares the ratings.csv file.
    :param answers_df: This is df read from Posts.xlsx file
    :return: returns a df with columns user id, item id, numeric rating
    '''
    questions_df = posts_df[posts_df['PostTypeId'] == 1]
    answers_df = posts_df[posts_df['PostTypeId'] == 2]

    answers_to_questions = answers_df[['OwnerUserId', 'ParentId']]
    answers_to_questions.columns = ['UserId', 'ItemId']

    answer_engagement_df = answers_to_questions.sort_values (['UserId', 'ItemId'])
    engagement_frequency = answer_engagement_df[['UserId', 'ItemId']].value_counts (ascending=True).reset_index (
        name='Rating')
    # engagement_frequency = engagement_frequency.astype({'UserId': str, 'ItemId': str}, errors = 'raise')
    # print(engagement_frequency.dtypes)
    # print(engagement_frequency['Rating'])
    engagement_frequency.to_csv ('../../raw_data/ratings.csv', index=False)
    return engagement_frequency

def user_features(users_df, rating_df):
    '''
    This function prepares the user-features.csv file.
    :param users_df: This is df read from users.xlsx
    :param rating_df: This is the processed ratings file
    :return: a df with the following information: user id, feature name, numeric feature value
    '''
    retained_users = rating_df['UserId'].unique ().tolist ()
    print (retained_users)
    users_df = users_df[users_df['Id'].isin (retained_users)]
    print (users_df.shape)

    genders = {'unknown': 0, 'male': 1, 'mostly_male': 2, 'female': 3, 'mostly_female': 4, 'andy': 5}
    users_df['Gender'] = users_df['DisplayName'].apply (lambda x: genders[d.get_gender (x)])
    countries = []
    for idx, row in users_df.iterrows ():
        # print(row)
        if str (row['Location']) == 'nan':
            countries.append ('')
        else:
            # print (row['Location'])
            try:
                location = geolocator.geocode (row['Location']).raw
                lat, lon = location['lat'], location['lon']
                country_alpha_2 = geolocator.reverse ([lat, lon]).raw['address']['country_code']
                country = pycountry.countries.get (alpha_2=country_alpha_2).numeric
                # print ('>>>>>>', country_alpha_2, country)
                countries.append (country)
            except Exception as e:
                print (e)
                countries.append ('')
    users_df['Country'] = countries
    users_df = users_df[['Id', 'Reputation', 'Views', 'UpVotes', 'DownVotes', 'Gender', 'Country']]
    # print (users_df.head (100))

    user_features = [['UserId', 'FeatureName', 'FeatureValue']]

    for idx, row in users_df.iterrows ():
        userId = row['Id']
        for feature_name in ['Reputation', 'Views', 'UpVotes', 'DownVotes', 'Gender', 'Country']:
            numeric_feature_value = row[feature_name]
            print(userId, feature_name, numeric_feature_value)
            user_features.append ([userId, feature_name, numeric_feature_value])
    user_features_df = pd.DataFrame.from_records (user_features[1:], columns=user_features[0])
    user_features_df.to_csv ('../../raw_data/user-features.csv', index=False)
    return

posts_df = pd.read_csv ('../../raw_data/Posts.csv')
item_features (posts_df)

rating_df = rating (posts_df)

users_df = pd.read_csv ('../../raw_data/Users.csv')
user_features (users_df, rating_df)
