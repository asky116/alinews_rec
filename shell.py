import concate_recall
import feature_engineering
import rank_lgb
from utils import set_logger


if __name__ == '__main__':
    recall_num = 50
    mode = 'offline'
    save = True

    set_logger()
    concate_recall.main(mode=mode, recall_n=recall_num, save=save)
    feature_engineering.main(mode=mode)
    rank_lgb.main(mode=mode)
