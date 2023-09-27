import os
import logging
from random import sample
import pandas as pd

from utils import set_logger

'''
合并训练与测试数据集后重新进行数据分割

为了使验证集与测试集的分布尽可能的一致，从点击超过两次的用户中随机选取50000名用户作为验证集。

由于测试集没有点击时间，而时间又是新闻点击预测的强特征，因此以用户的最后一次点击时间作为测试数据集的点击时间。
'''

def data_split(mode='offline', save=False):
    train_click = pd.read_csv("../tcdata/train_click_log.csv")
    test_click = pd.read_csv("../tcdata/testA_click_log.csv")

    logging.info(f'Data split, mode: {mode}')

    click = pd.concat([train_click, test_click])
    click.sort_values(['user_id', 'click_timestamp'], inplace=True)
    click = click.reset_index(drop=True)

    click['click_datetime'] = pd.to_datetime(click['click_timestamp'], unit='ms')
    click.rename(columns={'click_article_id': 'article_id'}, inplace=True)

    if mode == 'offline':
        # 从点击记录超过1次的用户中随机采样验证集
        click_count = click.user_id.value_counts()
        train_users = click_count[click_count > 1].index.to_list()
        val_users = sample(train_users, 50000)
        logging.info(f'val_users num: {len(set(val_users))}')

        query = click[click["user_id"].isin(val_users)][["user_id", "article_id", 'click_datetime']]
        query = query.groupby("user_id").tail(1)
        click = click[~click.index.isin(query.index)]

    elif mode == 'online':
        test_users = test_click['user_id'].unique()
        query = click[click["user_id"].isin(test_users)][["user_id", 'click_datetime']]
        query = query.groupby("user_id").tail(1)
        query['article_id'] = -1

    else:
        raise ValueError

    logging.info(
        f'Data split done;')

    if save:
        os.makedirs(f'../user_data/data/{mode}', exist_ok=True)
        click.to_pickle(f'../user_data/data/{mode}/click.pkl')
        query.to_pickle(f'../user_data/data/{mode}/query.pkl')
    else:
        return click, query


if __name__ == '__main__':
    log_dir = '../user_data/log'
    set_logger(log_dir)
    data_split(mode='online', save=True)