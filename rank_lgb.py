import gc
import os
import random
import warnings
import logging

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import gen_sub, set_logger, evaluate_news


'''
LGB做排序

召回的结果作为训练集
K折交叉验证

减少召回数量，可以减轻负样本过多的影响
'''

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)


def train_model(df_feature, df_query, feature_names):
    df_train = df_feature[df_feature['label'].notnull()]
    del df_feature
    gc.collect()

    ycol = 'label'

    model = lgb.LGBMClassifier(num_leaves=64,
                               max_depth=10,
                               learning_rate=0.05,
                               n_estimators=1000,
                               subsample=0.8,
                               colsample_bytree=0.8,
                               reg_alpha=0.5,
                               reg_lambda=0.5,
                               random_state=seed,
                               importance_type='gain',
                               n_jobs=-1,
                               metric=None)

    oof = []
    df_importance_list = []

    # 训练模型
    kfold = GroupKFold(n_splits=5)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train[feature_names], df_train[ycol], df_train['user_id'])
    ):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][ycol]

        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][ycol]

        logging.info(f'\nFold_{fold_id + 1} Training ================================\n')

        call_backs = [lgb.log_evaluation(period=100), lgb.early_stopping(stopping_rounds=50)]
        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              eval_metric='auc',
                              callbacks=call_backs)

        pred_val = lgb_model.predict_proba(X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        df_oof = df_train.iloc[val_idx][['user_id', 'article_id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        df_importance = pd.DataFrame({
            'feature_name': feature_names,
            'importance': lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        joblib.dump(model, f'../user_data/model/lgb{fold_id}.pkl')

    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    logging.info(f'\n{df_importance}')

    df_oof = pd.concat(oof)
    df_oof.sort_values(['user_id', 'pred'], inplace=True, ascending=[True, False])

    evaluate_news(df_query, df_oof)


def online_predict(df_test, feature_names):

    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0

    for fold_id in tqdm(range(5)):
        model = joblib.load(f'../user_data/model/lgb{fold_id}.pkl')
        pred_test = model.predict_proba(df_test[feature_names])[:, 1]
        prediction['pred'] += pred_test / 5

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv(f'../prediction_result/result.csv', index=False)


def main(mode='offline'):
    df_feature = pd.read_pickle(f'../user_data/data/{mode}/feature.pkl')
    feature_names = [
        'hour_hot',
        'day_hot',
        'cf_score',
        'next_score',
        'article_id_cnt',
        'category_id_cnt',
        'created_lastcreated_tsdiff',
        'created_lastclick_tsdiff',
        'user_id_category_id_cnt',
        'created_at_ts',
        'last_article_click_time',
        'click_hour_std',
        'words_count',
        'click_created_ts_diff_mean',
        'click_created_ts_diff_std',
        'last_click_words_diff',
        'clicked_words_mean',
        'clicked_article_created_time_max'
    ]
    feature_names.sort()

    if mode == 'offline':
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

        train_model(df_feature, df_query, feature_names)
    else:
        online_predict(df_feature, feature_names)


if __name__ == "__main__":
    log_dir = '../user_data/log'
    set_logger(log_dir)
    main(mode='online')
