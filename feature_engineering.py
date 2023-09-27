import logging

import pandas as pd

from utils import set_logger

'''
特征工程部分

除召回部分提供的特征外，主要的特征还包括点击时间与创建时间的时间差
'''

seed = 2020


def user_feature(df_click):

    df = df_click.groupby(['user_id']).agg({
        'click_created_ts_diff': [('click_created_ts_diff_mean', 'mean'),
                                  ('click_created_ts_diff_std', 'std')],
        'click_hour': [('click_hour_std', 'std')],
        'words_count': [('clicked_words_mean', 'mean'),
                        ('last_article_words', 'last')],
        'created_at_ts': [('last_article_created_time', 'last'),
                          ('clicked_article_created_time_max', 'max')],
        'click_timestamp': [('last_article_click_time', 'last'),
                            ('clicked_article_click_time_mean', 'mean')]
    }).droplevel(0, axis=1).reset_index()

    return df


def get_click(mode, df_article):
    df_click = pd.read_pickle(f'../user_data/data/{mode}/click.pkl')

    df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)
    df_click = df_click.merge(df_article, how='left')
    df_click['click_datetime'] = pd.to_datetime(df_click['click_timestamp'], unit='ms', errors='coerce')
    df_click['click_hour'] = df_click['click_datetime'].dt.hour
    df_click['click_created_ts_diff'] = df_click['click_timestamp'] - df_click['created_at_ts']

    return df_click


def main(mode='offline'):
    df_feature = pd.read_pickle(f'../user_data/data/{mode}/recall_v2.pkl')
    df_article = pd.read_csv('../tcdata/articles.csv')

    df_click = get_click(mode, df_article)

    df_temp = user_feature(df_click)
    df_feature = df_feature.merge(df_temp, how='left')

    df_feature = df_feature.merge(df_article, how='left')

    # df_feature['created_datetime'] = pd.to_datetime(df_feature['created_at_ts'], unit='ms')
    df_feature['created_lastcreated_tsdiff'] = df_feature['created_at_ts'] - df_feature['last_article_created_time']
    df_feature['created_lastclick_tsdiff'] = df_feature['created_at_ts'] - df_feature['last_article_click_time']
    df_feature['last_click_words_diff'] = df_feature['words_count'] - df_feature['last_article_words']

    for f in [['user_id'], ['article_id'], ['category_id'], ['user_id', 'category_id']]:
        df_temp = df_click.groupby(f).size().reset_index()
        df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]
        df_feature = df_feature.merge(df_temp, how='left')

    logging.info(f'Features: {df_feature.columns}')
    df_feature.to_pickle(f'../user_data/data/{mode}/feature.pkl')


if __name__ == '__main__':
    set_logger()
    main(mode='online')
