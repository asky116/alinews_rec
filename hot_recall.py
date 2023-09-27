import os
import logging
import pandas as pd
from tqdm import tqdm

from utils import evaluate_news, set_logger, gen_df
'''
根据新闻热度召回新闻、对其他召回的新闻进行评分

测试数据集的时间比较分散，因此需要保存各个时期的新闻热度。
线上服务是只需保存当前周期的新闻热度表。

fit
记录各周期的新闻热度表
以及用户历史记录表

recall
线下测试时根据测试数据集进行召回

single_recall
单用户召回
返回数据类型有
dict： 用于物品评分
dataframe： 测试召回
set： 方便与其他路召回结果进行合并

grade_items
根据热度评分
'''

class HotRecall(object):
    def __init__(self, mode='offline', period='H', recall_n=100, save=False):
        self.mode = mode
        self.save = save
        self.period = period
        self.rec_n = recall_n

        self.d_hot = None
        self.user_item_dict = None

    def fit(self, click):
        logging.info(f'Fit Hot_Recall, mode={self.mode}, period={self.period}')

        click['hot'] = 1
        click[f'click_{self.period}'] = click['click_datetime'].dt.to_period(self.period)

        self.user_item_dict = click.groupby('user_id')['article_id'].agg(list).to_dict()

        df_click = click[['article_id', f'click_{self.period}', 'hot']]
        d_hot = df_click.groupby(by=['article_id', f'click_{self.period}'])['hot'].count().reset_index()
        d_hot = d_hot.groupby(f'click_{self.period}').agg(list)
        d_hot.rename(columns={'article_id': 'article_list', 'hot': 'hot_list'}, inplace=True)

        self.d_hot = d_hot
        if self.save:
            os.makedirs(f'../user_data/sim/{self.mode}', exist_ok=True)
            d_hot.to_pickle(f'../user_data/sim/{self.mode}/{self.period}_hot.pkl')

    def recall(self, df_query):
        logging.info(f'Recall by HotItem, period: {self.period}')
        df_query[f'click_{self.period}'] = df_query['click_datetime'].dt.to_period(self.period)
        data_list = []

        for u_id, a_id, period in tqdm(df_query[['user_id', 'article_id', f'click_{self.period}']].values):

            rank = self.single_recall(period, user_id=u_id, form='df')
            df_tmp = gen_df(u_id, a_id, rank)
            data_list.append(df_tmp)

        df_data = pd.concat(data_list, sort=False)
        df_data = df_data.sort_values(['user_id', 'score'], ascending=[True, False]).reset_index(drop=True)
        if self.save:
            os.makedirs('../user_data/tmp/hot_recall', exist_ok=True)
            df_data.to_pickle(f'../user_data/tmp/hot_recall/0.pkl')
        return df_data

    def single_recall(self, period, user_id=None, form='dict'):
        items, hots = self.d_hot.loc[period].values if period in self.d_hot.index else ([], [])
        if form == 'dict':
            res = dict(zip(items, hots))
        else:
            res = pd.DataFrame({'article_id': items, 'score': hots})
            res = res.sort_values(by='score', ascending=False)

            hist_articles = self.user_item_dict.get(user_id, [])
            res = res[~res['article_id'].isin(hist_articles)][: self.rec_n]
            if form == 'set':
                res = set(res['article_id'].astype('int64').values)

        return res

    def grade_items(self, period, item_list):
        rank = self.single_recall(period)
        return [rank.get(item_id, 0) for item_id in item_list]


def main(mode='offline', period='H', save=True):
    log_dir = '../user_data/log'
    set_logger(log_dir)
    logging.info(f'Hot Recall，mode: {mode}')

    df_click = pd.read_pickle(f'../user_data/data/{mode}/click.pkl')
    # df_click['hot'] = 1
    # df_click[f'click_{period}'] = df_click['click_datetime'].dt.to_period(period)

    df_query = pd.read_pickle(f'../user_data/data/{mode}/query.pkl')
    df_query[f'click_{period}'] = df_query['click_datetime'].dt.to_period(period)

    hot_recall = HotRecall(mode=mode, period=period, save=save)
    hot_recall.fit(df_click)
    df_data = hot_recall.recall(df_query)

    if mode == 'offline':  # 计算召回指标
        evaluate_news(df_query, df_data)
    if save:
        df_data.to_pickle(f'../user_data/data/{mode}/recall_hot{period}.pkl')


if __name__ == "__main__":
    main(mode='offline', period='H', save=True)
