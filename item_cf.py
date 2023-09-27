import math
import os
import pickle
import random
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import evaluate_news, set_logger, gen_df

'''
根据基于物品的协同过滤召回新闻、对其他召回的新闻进行评分

保存用户的历史记录、物品总的热度、物品相似度。
线上服务可每日更新

fit
根据用户热度与物品热度计算、更新物品相似度。

recall
线下测试时根据测试数据集进行召回

single_recall
单用户召回，由于新闻点击具有强烈的连续性，因此仅根据用户最后两个点击记录进行召回。
返回数据类型有
dict： 用于物品评分
dataframe： 测试召回
set： 方便与其他路召回结果进行合并

grade_items
根据热度评分
'''

random.seed(2020)


class ItemCF(object):
    def __init__(self, mode, recall_n=100, save=True):
        self.mode = mode
        self.save = save
        self.rec_n = recall_n

        self.user_item_dict = None
        self.item_cnt = None
        self.sim_dict = {}
        self.user_item_sim = {}

    def _cal_sim(self):
        for _, items in tqdm(self.user_item_dict.items()):
            for loc1, item in enumerate(items):
                self.sim_dict.setdefault(item, {})
                for loc2, relate_item in enumerate(items):
                    if item == relate_item:
                        continue
                    # 位置信息权重, 考虑文章的正向顺序点击和反向顺序点击
                    self.sim_dict[item].setdefault(relate_item, 0)
                    loc_alpha = 1.0 if loc2 > loc1 else 0.7
                    loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                    self.sim_dict[item][relate_item] += loc_weight / math.log(1 + len(items))

    def _update_sim(self):
        for item, relate_items in tqdm(self.sim_dict.items()):
            for relate_item, cij in relate_items.items():
                self.sim_dict[item][relate_item] = cij / math.sqrt(self.item_cnt[item] * self.item_cnt[relate_item])

    def single_recall(self, user_id, form='dict'):
        # 字典形式CF召回 user_id 的 item_id： score
        res = {}
        interacted_items = self.user_item_dict.get(user_id, [])
        interacted_items_2 = interacted_items[::-1][:2]

        for loc, item in enumerate(interacted_items_2):
            for related_item, wij in self.sim_dict[item].items():
                if related_item not in interacted_items:
                    res[related_item] = res.get(related_item, 0) + wij * (0.7 ** loc)

        if form != 'dict':
            res = pd.DataFrame({'article_id': res.keys(), 'score': res.values()})
            res = res.sort_values(by='score', ascending=False)[:self.rec_n]
            if form == 'set':
                res = set(res['article_id'].astype('int64').values)
        return res

    def fit(self, df_click):
        self.user_item_dict = df_click.groupby('user_id')['article_id'].agg(list).to_dict()
        self.item_cnt = df_click.groupby('article_id').size().to_dict()
        # 计算、更新、保存物品相似度字典
        self._cal_sim()
        self._update_sim()

        if self.save:
            os.makedirs(f'../user_data/sim/{self.mode}', exist_ok=True)
            with open(f'../user_data/sim/{self.mode}/itemcf_sim.pkl', 'wb') as f:
                pickle.dump(self.sim_dict, f)

    def recall(self, df_query, worker_id=0):
        logging.info('Recall by ItemCF')
        data_list = []
        for user_id, item_id in tqdm(df_query[['user_id', 'article_id']].values):
            if user_id not in self.user_item_dict:
                continue

            rank = self.single_recall(user_id, form='df')
            # if not rank:
            #     continue
            df_temp = gen_df(user_id, item_id, rank)
            data_list.append(df_temp)

        df_data = pd.concat(data_list, sort=False)
        df_data.sort_values(['user_id', 'score'], ascending=[True, False]).reset_index(drop=True)
        if self.save:
            os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
            df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')
        return df_data

    def grade_items(self, user_id, item_list):
        rank = self.single_recall(user_id)
        return [rank.get(item_id, 0) for item_id in item_list]


def main(mode='offline', save=True):
    log_dir = '../user_data/log'
    set_logger(log_dir)
    logging.info(f'itemcf recall，mode: {mode}')

    df_click = pd.read_pickle(f'../user_data/data/{mode}/click.pkl')
    df_query = pd.read_pickle(f'../user_data/data/{mode}/query.pkl')

    logging.info('Calculate and Save Item_Similarity ')
    item_cf = ItemCF(mode=mode, save=save)
    item_cf.fit(df_click)
    df_data = item_cf.recall(df_query)

    if mode == 'offline':  # 计算召回指标
        evaluate_news(df_query, df_data)

    if save:
        df_data.to_pickle(f'../user_data/data/{mode}/recall_itemcf.pkl')


if __name__ == '__main__':
    main(mode='offline', save=False)
