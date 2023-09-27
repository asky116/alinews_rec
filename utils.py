import logging
import os
import pickle
from random import sample
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def evaluate(df, total):
    # 计算评价指标
    result = np.zeros(10)
    mark = np.array([5, 10, 20, 40, 50])

    gg = df.groupby(['user_id'])

    for _, g in tqdm(gg):
        try:
            item_id = g[g['label'] == 1]['article_id'].values[0]
        except Exception as e:
            continue

        predictions = g['article_id'].values.tolist()

        rank = predictions.index(item_id) + 1
        n = 10 - 2 * sum(mark >= rank)
        result[n:: 2] += 1
        result[n+1:: 2] += 1 / rank

    return result / total


def gen_sub_multitasking(test_users, prediction, all_articles, worker_id):
    # 推荐结果不足5个的话，随机抽取新闻补充到5个
    lines = []

    for test_user in test_users:
        g = prediction[prediction['user_id'] == test_user]
        g = g.head(5)
        items = g['article_id'].values.tolist()

        if len(set(items)) < 5:
            buchong = all_articles - set(items)
            buchong = sample(buchong, 5 - len(set(items)))
            items += buchong

        assert len(set(items)) == 5

        lines.append([test_user] + items)

    os.makedirs('../user_data/tmp/sub', exist_ok=True)

    with open(f'../user_data/tmp/sub/{worker_id}.pkl', 'wb') as f:
        pickle.dump(lines, f)


def gen_sub(prediction, n_split=5):
    # 多进程生成提交数据
    logging.info('Generating submit file...')
    prediction.sort_values(['user_id', 'pred'], inplace=True, ascending=[True, False])

    all_articles = set(prediction['article_id'].values)

    sub_sample = pd.read_csv('../tcdata/testA_click_log.csv')
    test_users = sub_sample.user_id.unique()

    total = len(test_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/sub'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    p = Pool(n_split)
    for i in range(0, total, n_len):
        part_users = test_users[i:i + n_len]
        p.apply_async(gen_sub_multitasking, args=(part_users, prediction, all_articles, i))

    p.close()
    p.join()

    lines = []
    for path, _, file_list in os.walk('../user_data/tmp/sub'):
        for file_name in file_list:
            with open(os.path.join(path, file_name), 'rb') as f:
                line = pickle.load(f)
                lines += line

    df_sub = pd.DataFrame(lines)
    df_sub.columns = [
        'user_id', 'article_1', 'article_2', 'article_3', 'article_4',
        'article_5'
    ]
    return df_sub


def set_logger(log_dir='../user_data/log', method='test'):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, method + '.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s; P%(process)d; %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(filename=log_file, encoding='utf-8', mode='a'),
                                  logging.StreamHandler()])


def evaluate_news(df_query, df_data):
    # 评测推荐结果
    # 命中率 hit, 与 MRR(Mean Reciprocal Rank)
    logging.info(f'Evaluate News!')
    evaluate_name = ['h5', 'm5', 'h10', 'm10', 'h20', 'm20', 'h40', 'm40', 'h50', 'm50']
    total = df_query.user_id.nunique()

    evaluate_result = evaluate(df_data, total)

    message = ', '.join([f'{name}:{value:.4f}' for name, value in zip(evaluate_name, evaluate_result)])
    logging.info(f'\n*** {message} ***')


def gen_df(u_id, i_id, df):
    # 根据召回结果生成待拼接的dataframe
    df['label'] = 0
    df['user_id'] = u_id
    df.loc[df['article_id'] == i_id, 'label'] = 1

    df.fillna(0, inplace=True)
    df['user_id'] = df['user_id'].astype('int')
    df['article_id'] = df['article_id'].astype('int')

    return df
