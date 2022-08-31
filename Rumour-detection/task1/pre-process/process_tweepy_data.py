 # -*- coding: utf-8 -*
import emoji
from nltk.corpus import stopwords
import re
import nltk
import json
import pandas as pd
# from utils import timer
from datetime import datetime
import numpy as np
import os
import tqdm
import json
from textblob import TextBlob
from time import strftime
import argparse
from datetime import datetime
nltk.download('wordnet')
stemmer = nltk.stem.porter.PorterStemmer()
stopword = stopwords.words('english') 

parser= argparse.ArgumentParser(description='ArgUtils')
parser.add_argument('-type', type=str, default='test', help="data type, train or dev or test or covid")
# parser.add_argument('-a', type=str, default=None, help="agent_id_from_platform id")
args = parser.parse_args()
TYPE = args.type


def clean_tweet(content):
    

    # def compute_num_month(content):
    #     month_num = 0
    #     month = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "octorber", "november", "december", 
    #             "jan.", "feb.", "mar.", "apr.", "may.", "jun.", "jul.", "aug.", "sept.", "oct.", "nov.", "dec."]
    #     for i in content:
    #         if i in month:
    #             month_num += 1
    #     return month_num
    # replace_abbreviations
    
    content = content.lower()
    content = re.sub(r"won't", "will not", content)
    content = re.sub(r"can't", "can not", content)
    content = re.sub(r"cannot", "can not", content)
    content = re.sub(r"n't", " not", content)
    content = re.sub(r"'re", " are", content)
    content = re.sub(r"'s", " is", content)
    content = re.sub(r"'d", " would", content)
    content = re.sub(r"'ll", " will", content)
    content = re.sub(r"'t", " not", content)
    content = re.sub(r"'ve", " have", content)
    content = re.sub(r"'m", " am", content)
    content = re.sub(r".”", " ", content)
    
    # get the number of month be mentioned
    # month_num = compute_num_month(content)

    # get the number of url
    question_mark_num = len(re.findall(r'\?', content))
    mentioned_url_num = len(re.findall(r'https?://[^ ]+', content))
    mentioned_url_num += len(re.findall(r'www.[^ ]+', content))
    # get the number of twitter ID be mentioned
    id_num = len(re.findall(r'@[A-Za-z0-9_]+', content))
    content = re.sub(r'@[A-Za-z0-9_]+', '', content) # remove twitter ID
    # remove url 
    ## http, https
    content = re.sub(r'http', '', content)
    content = re.sub(r'https', '', content) 
    # content = re.sub(r'https?://[^ ]+', '', content) 
    ## www.
    content = re.sub(r'www', '', content) 
    # content = re.sub(r'www.[^ ]+', '', content)
    # get the emoji and replace them as words
    emojis = emoji.distinct_emoji_list(content)
    for e in emojis:
        try:
            content = re.sub(e, emoji.demojize(e), content)
        except:
            # print('no corresponding emoji!')
            # print(content)
            content = re.sub(e, '', content)
    content = re.sub('\w+\d+\w+', '', content) # remove the word contains numbers
    
    content = re.sub(r'[:_!\+“\-=——,$%^\?\\~\"\'@#$%&\*<>{}\[\]()/]', ' ', content) # remove punctuation, except .
    
    content = re.sub(r"\s+", " ", content) # conver multiple spaces as a single space
    content = content.strip()
    
    # remove stop words 
    # TODO: and keep only english letters
    
    content = [c for c in content.split(' ') if c not in stopword and c.isalpha()]
    # do stemming
    content = [stemmer.stem(token.strip()) for token in content]
    
    return ' '.join(content), mentioned_url_num, id_num, question_mark_num 
    #, month_num
# @timer('ms')
def json2df(json_file, json_content=None):
    """
    json_file: 'train_reply.json'
    """
    print(json_content is None)
    if json_file is not None:
        with open(json_file,'r+') as file:
            content = file.read()
    else:
        content = json_content
    content = json.loads(content)
    df = pd.DataFrame(content)
    df = df.T
    return df
def clean_test_data(df, is_test=True):
    df['temp'] = df['text'].apply(lambda x: clean_tweet(x))
    df['text'] = df['temp'].apply(lambda x: x[0])
    df['mentioned_url_num'] = df['temp'].apply(lambda x: x[1])
    df['id_num'] = df['temp'].apply(lambda x: x[2])

    df['question_mark'] = df['temp'].apply(lambda x: x[3])
    df['tweet_id'] = [str(i) for i in df.index]
    if is_test:
        df['tweet_id'] = df['id'].apply(lambda x: str(x))
        df = df.drop(columns=['temp', 'id', 'id_str'])
    else:
        df = df.drop(columns=['temp'])
    return df

def check_weekday_test(df):
    """
    df: source_df or reply_df
    """
    df['isoweekday'] = df['created_at'].apply(lambda x: datetime.strptime(str(x).split(' ')[0], '%Y-%m-%d').isoweekday())
    df['isweekday'] = df['isoweekday'].apply(lambda x: 1 <= x <= 5)
    df = df.drop(columns='isoweekday')
    return df

def split_source_reply(txt_file):
  """
  txt_file: 'train.data.txt'
  """
  with open(txt_file) as f:
      ids = f.readlines()
  source_ids = []
  reply_ids = []
  reply_source = []
  source_txt_file = txt_file.split('.')[0] + '_source.txt'
  reply_txt_file = txt_file.split('.')[0] + '_reply.txt'
  reply_source_txt_file = txt_file.split('.')[0] + '_reply_with_source.txt'
  for i in range(len(ids)):
      source_ids.append(ids[i].split(',')[0].strip())
      reply_ids.extend([r.strip() for r in ids[i].split(',')[1:]])
      reply_source.extend([[r.strip(), ids[i].split(',')[0].strip()] for r in ids[i].split(',')[1:]])
# save source_ids
  with open(source_txt_file,'w') as f:
      for i in source_ids:
          f.write(i)
          f.write('\n')
# save reply_ids
  with open(reply_txt_file,'w') as f:
      for i in reply_ids:
        f.write(i)
        f.write('\n')
    # save reply with source_id
  with open(reply_source_txt_file,'w') as f:
      for i in reply_source:
          f.write(','.join(i))
          f.write('\n')
def merge_json(data_type, source_or_reply, ids_list):
    merges_file = os.path.join(f'./tweepy_data/objects/', f'{data_type}_{source_or_reply}.json')
    path_results = f'./tweepy_data/objects/{data_type}_objects'
    
    with open(merges_file, "w", encoding="utf-8") as f0:
        for file in os.listdir(path_results):
            if file.split('.')[0].strip() in ids_list:
                print('===', file.split('.')[0].strip())
                with open(os.path.join(path_results, file), "r", encoding="utf-8") as f1:
                    for line in f1:
                        line_dict = json.loads(line)
                        js = json.dumps(line_dict, ensure_ascii=False)
                        f0.write(js + '\n')
                    f1.close()
        f0.close()

def sort_by_time(raw_file, json_file):
    with open(raw_file) as file:
        ids = file.readlines()
        
    df = pd.read_json(path_or_buf=json_file, lines=True)
    df.index = [str(i) for i in df['id']]
    save_name = raw_file[:-4] + '_sorted.txt'
    with open(save_name, 'w') as file:
        date = pd.Series(pd.DatetimeIndex(df['created_at']), index=df.index)
        df.drop(['created_at'], axis=1, inplace=True)
        df['time'] = date
        for id_ in ids:
            ids_ = id_.strip().split(',')
            source_id = ids_[0]
            file.write(source_id)
            if len(ids_) > 1:
                reply_ids = ids_[1:]
                reply_ids[-1] = reply_ids[-1].strip()
                valid_ids = [index for index in reply_ids if index in df.index]
                sorted_replies = df.loc[valid_ids].sort_values(by='time')
                if len(valid_ids) > 0:
                    file.write(',')

                for i, index in enumerate(sorted_replies.index):
                    file.write(index)
                    if i != len(sorted_replies.index) - 1:
                        file.write(',')

            file.write('\n')
def concat_reply(data_type, source_df):
    """concat replies on source tweets
    data_type: 'dev', 'train', 'test'
    """
    df = pd.DataFrame(columns=['tweet_id', 'reply'])
    with open(f'./tweepy_data/original_data/{data_type}.data_sorted.txt', 'r') as f:
    # with open(f'./data/original_data/{data_type}.data_sorted.txt', 'r') as f:
        content = f.readlines()
    df['tweet_id'] = [c.split(',')[0].strip() for c in content]
    df['reply'] = [','.join([i.strip() for i in c.split(',')[1:]]) for c in content]
    source_df = pd.merge(source_df, df, on='tweet_id', how='left')
    return source_df
def concat_label(data_type, source_feature_df):
    """Concat labels on source tweets
    data_type: 'dev', 'train', 'test'
    """
    df = pd.DataFrame(columns=['tweet_id', 'label'])
    with open(f'./tweepy_data/original_data/{data_type}_source.txt', 'r') as f:
        ids = f.readlines()
    with open(f'./tweepy_data/original_data/{data_type}.label.txt', 'r') as f:
        labels = f.readlines()
    df['tweet_id'] = [id.strip() for id in ids]
    df['label'] = [label.strip() for label in labels]
    df_labels = pd.merge(source_feature_df, df, on='tweet_id', how='left')
    df_labels['label'] = df_labels['label'].apply(lambda x: 0 if x == 'nonrumour' else 1)
    return df_labels
def processing(data_type):

    # './data/tweet-objects/test_source.json'
    used_cols = ['created_at', 'id', 'id_str', 'text', 'truncated', 'entities', 'source',
       'in_reply_to_status_id', 'in_reply_to_status_id_str',
       'in_reply_to_user_id', 'in_reply_to_user_id_str',
       'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place',
       'contributors', 'is_quote_status', 'retweet_count', 'favorite_count',
       'favorited', 'retweeted', 'lang', 'extended_entities',
       'possibly_sensitive', 'possibly_sensitive_appealable']
    #    'quoted_status_id', 'quoted_status_id_str', 'quoted_status']
    source_df = pd.read_json(path_or_buf=f'./tweepy_data/objects/{data_type}_source.json', lines=True)
    source_df = source_df[used_cols]
    reply_df= pd.read_json(path_or_buf=f'./tweepy_data/objects/{data_type}_reply.json', lines=True)
    reply_df = reply_df[used_cols]
    source_df = clean_test_data(source_df)
    reply_df = clean_test_data(reply_df)
    print('cleaned')
    # get 'verified', 'followers_count', 'listed_count'
    for i in ['protected', 'followers_count', 'friends_count', 
                'listed_count', 'favourites_count', 'geo_enabled', 'verified', 
                'statuses_count','contributors_enabled', 
                'is_translator', 'is_translation_enabled','has_extended_profile', 
                'default_profile', 'default_profile_image', 'following', 
                'follow_request_sent', 'notifications']:
        source_df[i] = source_df['user'].apply(lambda x: x.get(i, 0))
        reply_df[i] = reply_df['user'].apply(lambda x: x.get(i, 0))
    
    source_df['count_age'] = source_df['user'].apply(lambda x: datetime.now().year - int(x['created_at'].split(' ')[-1]))
    
    source_df['user_engagement'] = source_df.apply(lambda x: x['statuses_count'] / (x['count_age']+1) if isinstance(x['statuses_count'], int) and isinstance(x['count_age'], int)  else 0, axis=1)
    source_df['following_rate'] = source_df.apply(lambda x: x['following'] / (x['count_age']+1) if isinstance(x['following'], int) and isinstance(x['count_age'], int)  else 0, axis=1)
    source_df['favourite_rate'] = source_df.apply(lambda x: x['favourites_count'] / (x['count_age']+1) if isinstance(x['favourites_count'], int) and isinstance(x['count_age'], int)  else 0, axis=1)
    source_df['has_url'] = source_df['entities'].apply(lambda x: 0 if len(x['urls']) == 0 else 1)
    # get reply statistic info
    reply_df['count_age'] = reply_df['user'].apply(lambda x: datetime.now().year - int(x['created_at'].split(' ')[-1]))   
    reply_df['user_engagement'] = reply_df.apply(lambda x: x['statuses_count'] / (x['count_age']+1)  if isinstance(x['statuses_count'], int) and isinstance(x['count_age'], int)  else 0, axis=1)
    reply_df['following_rate'] = reply_df.apply(lambda x: x['following'] / (x['count_age']+1)  if isinstance(x['following'], int) and isinstance(x['count_age'], int)  else 0, axis=1)
    reply_df['favourite_rate'] = reply_df.apply(lambda x: x['favourites_count'] / (x['count_age']+1)  if isinstance(x['favourites_count'], int) and isinstance(x['count_age'], int)  else 0, axis=1)

    reply_df['has_url'] = reply_df['entities'].apply(lambda x: 0 if len(x['urls']) == 0 else 1)
    
    source_df = concat_reply(data_type, source_df)
    # get reply count
    source_df['reply_count'] = source_df['reply'].apply(lambda x: len(x.split(',')))
    
    source_df.index = [str(i) for i in source_df['tweet_id']]

    # sorted ids
    if data_type == 'test':
        with open('tweepy_data/original_data/test_source.txt', 'r') as f:
            c = f.readlines()
        source_df = source_df.loc[[i.strip() for i in c]]
    source_df = check_weekday_test(source_df)
    reply_df = check_weekday_test(reply_df)
    # add sentiment score
    source_df['senti_score'] = source_df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity > 0 else 0)
    reply_df['senti_score'] = reply_df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity > 0 else 0)
    reply_df.index = [str(i) for i in reply_df['tweet_id']]
    
    data_txt = np.loadtxt(f'./tweepy_data/original_data/{data_type}_reply_with_source.txt',dtype=str, delimiter=',')
    data_txtDF = pd.DataFrame(columns=['tweet_id', 'source_id'], data=data_txt)
    reply_df['tweet_id'] = reply_df.index
    reply_df.index = list(range(len(reply_df)))
    reply_df_source = pd.merge(reply_df, data_txtDF, on='tweet_id', how='left')
    reply_df_source.index = [str(i) for i in reply_df_source['tweet_id']]
    stat_data = []
    statis_feature=[ 'contributors', 'user_engagement','following_rate','favourite_rate',
        'possibly_sensitive', 'possibly_sensitive_appealable',
            'retweet_count', 'favorite_count', 'mentioned_url_num', 'id_num', 'question_mark',
        'followers_count', 'friends_count', 'listed_count', 'favourites_count',
        'statuses_count', 'has_url', 'senti_score','truncated', 'is_quote_status', 'favorited', 'retweeted', 'protected',
        'geo_enabled', 'verified', 'isweekday','contributors_enabled', 'is_translator', 'is_translation_enabled',
        'has_extended_profile', 'default_profile', 'default_profile_image', 'following', 'follow_request_sent', 'notifications']
    print('compute reply stat feat')
    for source_id, df in reply_df_source.groupby('source_id'):
        if str(source_id) not in source_df.index:
            continue
        if isinstance(source_df.loc[source_id]['reply'], pd.Series):
            cur_data = [source_id] + [np.nan] * (len(statis_feature)+1)
        else:
            ids = [str(i).strip() for i in source_df.loc[source_id]['reply'].split(',')]
            cur_data = [source_id, ' [SEP] '.join(df.loc[ids]['text'].values)]
            for i in statis_feature:
                cur_data.append(df[i].sum())
        stat_data.append(cur_data)

    reply_stat_df = pd.DataFrame(columns=['tweet_id', 'reply_text'] + ['reply_' + s for s in statis_feature], data=stat_data)
    
    source_df.index = list(range(len(source_df)))
    source_df_reply = pd.merge(source_df, reply_stat_df, on='tweet_id', how='left')
    if data_type == 'dev' or data_type == 'train':
        source_df_reply = concat_label(data_type, source_df_reply)
    for i in ['reply_' + s for s in statis_feature]:
        source_df_reply[i] = source_df_reply.apply(lambda x: x[i] / x['reply_count'], axis=1)
    return  source_df_reply
def extract_stat_tweet_feat(istrain, df):
    # extract statistic features
    # reply_reply_count， reply_quote_count，quote_count
    statistic_features = ['reply_' + i for i in [ 'user_engagement','contributors','favourite_rate',
       'possibly_sensitive', 'possibly_sensitive_appealable','following_rate',
        'retweet_count', 'favorite_count', 'mentioned_url_num', 'id_num','question_mark',
       'followers_count', 'friends_count', 'listed_count', 'favourites_count',
       'statuses_count', 'has_url', 'senti_score','truncated', 'is_quote_status', 'favorited', 'retweeted', 'protected',
       'geo_enabled', 'verified', 'contributors_enabled', 'isweekday', 'is_translator', 'is_translation_enabled',
       'has_extended_profile', 'default_profile', 'default_profile_image', 'following', 'follow_request_sent', 'notifications']] + ['tweet_id','user_engagement', 'contributors',
       'possibly_sensitive', 'possibly_sensitive_appealable','following_rate',
        'retweet_count', 'favorite_count', 'mentioned_url_num', 'id_num','favourite_rate','question_mark',
       'followers_count', 'friends_count', 'listed_count', 'favourites_count',
       'statuses_count', 'has_url', 'senti_score','truncated', 'is_quote_status', 'favorited', 'retweeted', 'protected',
       'geo_enabled', 'verified', 'isweekday', 'reply_count','contributors_enabled', 'is_translator', 'is_translation_enabled','has_extended_profile', 'default_profile', 'default_profile_image', 'following', 'follow_request_sent', 'notifications']
    
    
    
    if istrain:
        stat_feat_df = df[statistic_features + ['label', 'tweet_id']]
        tweet_df = df[['tweet_id', 'text', 'reply_text', 'label']]
        tweet_df.index = df['tweet_id']
        # nonan_stat_feat_df = []
        # for _, cur_df in stat_feat_df.groupby('label'):
        #     nonan_stat_feat_df.append(cur_df.fillna(cur_df.mean()))
        # nonan_stat_feat_df = pd.concat(nonan_stat_feat_df)
        # nonan_stat_feat_df.index = nonan_stat_feat_df['tweet_id']
        # nonan_stat_feat_df = nonan_stat_feat_df.loc[tweet_df.index]
    else:
        stat_feat_df = df[statistic_features + ['tweet_id']]
        tweet_df = df[['tweet_id', 'text', 'reply_text']]
        tweet_df.index = df['tweet_id']
        # nonan_stat_feat_df = stat_feat_df.fillna(stat_feat_df.mean())
    # tweet_df = df.drop(columns=statistic_features)
    
    # convert into float
    # for col in ['tweet_count', 'followers_count', 'verified']:
    #     stat_feat_df[col] = stat_feat_df[col].apply(lambda x: float(x))
    # fill nan using corresponding mean
    return stat_feat_df, tweet_df
def get_tweet_stat_df(data_type):
    # split_source_reply(f'tweepy_data/original_data/{data_type}.data.txt')
    # with open(f'tweepy_data/original_data/{data_type}_source.txt', 'r') as f:
    #     content = f.readlines()
    # source_ids = [c.strip() for c in content]
    # with open(f'tweepy_data/original_data/{data_type}_reply.txt', 'r') as f:
    #     content = f.readlines()
    # reply_ids = [c.strip() for c in content]
    # merge_json(data_type, 'source', source_ids)
    # merge_json(data_type,'reply', reply_ids)
    # raw_files = [f'./tweepy_data/original_data/{data_type}.data.txt']
    # json_files = [f'./tweepy_data/objects/{data_type}_reply.json']
    # for raw_file, json_file in zip(raw_files, json_files):
    #     sort_by_time(raw_file, json_file)
    df = processing(data_type)
    if data_type == 'test' or data_type == 'covid':
        stat_feat_df, tweet_df = extract_stat_tweet_feat(False, df)
    else:
        stat_feat_df, tweet_df = extract_stat_tweet_feat(True, df)
    return  stat_feat_df, tweet_df
if __name__ == '__main__':
    stat_feat_df, tweet_df = get_tweet_stat_df(TYPE)
    print(f'tweepy_data/res/{TYPE}_stat_feat_df.csv')
    stat_feat_df.to_csv(f'tweepy_data/res/{TYPE}_stat_feat_df.csv', index=False)
    tweet_df.to_csv(f'tweepy_data/res/{TYPE}_tweet_df.csv', index=False)
    print('saving.')