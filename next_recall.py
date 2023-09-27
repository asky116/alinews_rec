import logging
import os

import pandas as pd
from tqdm import tqdm

from utils import evaluate_news, set_logger, gen_df

'''
根据用户前后点击的两个新闻的联系召回新闻、对其他召回的新闻进行评分

保存 (article, next_article) 的出现次数以及用户的最后一次点击

fit
根据用户分组，统计新闻对的出现次数

recall
线下测试时根据测试数据集进行召回

single_recall
单用户召回。
返回数据类型有
dict： 用于物品评分
dataframe： 测试召回
set： 方便与其他路召回结果进行合并

grade_items
根据热度评分
'''

class NextRecall(object):
    def __init__(self, mode='offline', recall_n=10, save=False):
        self.mode = mode
        self.save = save
        self.rec_n = recall_n

        self.last_article = None
        self.next_article = None
        self.user_item_dict = None

    def fit(self, click):
        logging.info(f'Fit next_connect recall, mode={self.mode}')
        self.last_article = click.groupby(by='user_id')['article_id'].last()
        self.user_item_dict = click.groupby('user_id')['article_id'].agg(list).to_dict()

        temp = click.groupby(by='user_id')['article_id'].shift(-1)
        temp.name = 'next_article'
        temp = pd.concat([click, temp], axis=1)
        temp.dropna(inplace=True)
        temp = temp.astype({'next_article': 'int64'})
        temp['score'] = 1

        self.next_article = temp.groupby(by=['article_id', 'next_article'])['score'].size()

    def single_recall(self, user_id, form='dict'):
        # return df: {index:article_id, columns: score}
        article_id = self.last_article.get(user_id, default=None)
        try:
            res = self.next_article.loc[article_id].reset_index().rename(columns={'next_article': 'article_id'})
        except:
            res = pd.DataFrame({'article_id': [], 'score': []})
        if form == 'dict':
            res = dict(zip(res['article_id'], res['score']))
        else:
            hist_clicks = self.user_item_dict[user_id]
            res = res.sort_values(by='article_id', ascending=False)[: self.rec_n]
            res = res[~res['article_id'].isin(hist_clicks)]

            if form == 'set':
                res = set(res['next_article'].astype('int64').values)

        return res

    def recall(self, df_query):
        logging.info(f'Recall by NextRecall, mode={self.mode}')
        data_list = []
        for u_id, a_id in tqdm(df_query[['user_id', 'article_id']].values):

            temp = self.single_recall(user_id=u_id, form='df')
            temp = gen_df(u_id, a_id, temp)
            data_list.append(temp)

        df_data = pd.concat(data_list, sort=False)
        df_data = df_data.sort_values(['user_id', 'score'], ascending=[True, False]).reset_index(drop=True)
        if self.save:
            os.makedirs('../user_data/tmp/next_recall', exist_ok=True)
            df_data.to_pickle(f'../user_data/tmp/next_recall/0.pkl')
        return df_data

    def grade_items(self, user_id, items):
        rank = self.single_recall(user_id)
        return [rank.get(item, 0) for item in items]


def main(mode='offline', save=False):
    set_logger()
    logging.info(f'Next article recall, mode={mode}')

    df_click = pd.read_pickle(f'../user_data/data/{mode}/click.pkl')
    df_query = pd.read_pickle(f'../user_data/data/{mode}/query.pkl')

    next_recall = NextRecall(mode=mode, save=save)
    next_recall.fit(df_click)
    df_data = next_recall.recall(df_query)

    if mode == 'offline':  # 计算召回指标
        evaluate_news(df_query, df_data)
    if save:
        df_data.to_pickle(f'../user_data/data/{mode}/next_recall.pkl')


if __name__ == '__main__':
    main(mode='offline', save=False)
