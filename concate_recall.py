import pandas as pd
from tqdm import tqdm

from item_cf import ItemCF
from hot_recall import HotRecall
from next_recall import NextRecall
from utils import set_logger

'''
根据协同过滤、小时为周期的热度、天为周期的热度进行多路召回，并对召回结果进行评分。
'''


def main(mode='offline', recall_n=100, save=True):
    df_click = pd.read_pickle(f'../user_data/data/{mode}/click.pkl')
    df_query = pd.read_pickle(f'../user_data/data/{mode}/query.pkl')
    df_query['click_hour'] = df_query['click_datetime'].dt.to_period('H')
    df_query['click_day'] = df_query['click_datetime'].dt.to_period('D')

    # 建立、训练召回模型
    item_cf = ItemCF(mode, recall_n=recall_n, save=False)
    hot_hour = HotRecall(mode, recall_n=recall_n, save=False)
    hot_day = HotRecall(mode, recall_n=recall_n, period='D')
    next_recall = NextRecall(mode)

    item_cf.fit(df_click)
    hot_hour.fit(df_click)
    hot_day.fit(df_click)
    next_recall.fit(df_click)

    data_list = []
    for u_id, a_id, hour, day in tqdm(df_query[['user_id', 'article_id', 'click_hour', 'click_day']].values):

        # 多路召回与合并
        cf_rec = item_cf.single_recall(u_id, form='set')
        hour_rec = hot_hour.single_recall(hour, user_id=u_id, form='set')
        day_rec = hot_day.single_recall(day, user_id=u_id, form='set')

        article_rec = list(set.union(cf_rec, hour_rec, day_rec))
        if not article_rec:
            continue
        if a_id != -1 and (a_id not in article_rec):
            continue

        # 对召回结果进行评分
        df_temp = pd.DataFrame({
            'user_id': u_id,
            'article_id': article_rec,
            'hour_hot': hot_hour.grade_items(hour, article_rec),
            'day_hot': hot_day.grade_items(day, article_rec),
            'cf_score': item_cf.grade_items(u_id, article_rec),
            'next_score': next_recall.grade_items(u_id, article_rec),
            'label': 0
        })

        df_temp.loc[df_temp['article_id'] == a_id, 'label'] = 1
        data_list.append(df_temp)

    df_data = pd.concat(data_list)
    if save:
        df_data.to_pickle(f'../user_data/data/{mode}/recall_v2.pkl')

if __name__ == '__main__':
    set_logger()
    main(mode='online', save=True)