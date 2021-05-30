import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from logging import StreamHandler, DEBUG, INFO, Formatter, FileHandler, getLogger
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, ParameterGrid
import sys
sys.path.append('..')
from inputs import load_train_data, load_test_data

if __name__ == '__main__':
    MODEL_NAME = 'XGBOOST'
    logger = getLogger(MODEL_NAME)
    logger.setLevel(DEBUG)

    log_fmt = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    sh = StreamHandler()
    sh.setLevel(INFO)
    sh.setFormatter(log_fmt)
    logger.addHandler(sh)

    log_file = f'../logs/{MODEL_NAME}.log'
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(log_fmt)
    logger.addHandler(fh)

    train_x, train_y = load_train_data()
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    param_grid = {'random_state': [1],
                  'learning_rate': [0.1],
                  'n_estimators': [1000],
                  'early_stopping_rounds': [50],
                  'max_depth': [20*(i+1) for i in range(5)],
                  'min_child_weight': [0.1, 0.5, 1, 2, 3, 5],
                  'gamma': [0, 0.1, 0.5, 1, 2, 3, 5],
                  'alpha': [0, 0.1, 0.5, 1, 2, 3, 5],
                  'subsample': np.linspace(0.1, 1, 10),
                  'colsample_bytree': np.linspace(0.1, 1, 10),
                  # 'colsample_bylevel': np.linspace(0.1, 1, 10),
                  # 'colsample_bynode': np.linspace(0.1, 1, 10),
                  'lambda': [0, 0.1, 0.5, 1, 2, 3, 5]}
    min_score = 100
    min_params = None

    for params in tqdm(list(ParameterGrid(param_grid))):
        logger.info(f'param: {params}')

        lst_logloss_score = []
        all_preds = np.zeros(shape=train_y.shape[0])

        for tr_idx, va_idx in cv.split(train_x, train_y):
            early_stopping_rounds = params['early_stopping_rounds']
            params_ = {i: params[i] for i in params if i != 'early_stopping_rounds'}

            tr_x = train_x.iloc[tr_idx]
            va_x = train_x.iloc[va_idx]

            tr_y = train_y.iloc[tr_idx]
            va_y = train_y.iloc[va_idx]

            model = XGBClassifier(**params_)
            model.fit(tr_x, tr_y, eval_set=[(va_x, va_y)],
                      early_stopping_rounds=early_stopping_rounds)
            pred_proba = model.predict_proba(va_x)
            sc_logloss = log_loss(va_y, pred_proba)

            lst_logloss_score.append(sc_logloss)
            logger.debug(f'   logloss: {sc_logloss}')

        sc_logloss = np.mean(lst_logloss_score)
        if min_score > sc_logloss:
            min_score = sc_logloss
            min_params = params

        logger.info(f'logloss: {sc_logloss}')
        logger.info('current min score: {}, params: {}'.format(min_score, min_params))

    logger.info('minimum params: {}'.format(min_params))
    logger.info('minimum logloss: {}'.format(min_score))

    test_x = load_test_data()

    early_stopping_rounds = min_params['early_stopping_rounds']
    min_params_ = {i: min_params[i] for i in min_params if i != 'early_stopping_rounds'}

    model = XGBClassifier(**min_params)
    model.fit(train_x, train_y)
    preds_test = model.predict(test_x)

    df_submit = pd.read_csv('../inputs/sample_submission.csv', header=None)
    df_submit[1] = preds_test
    df_submit.to_csv('xgb.csv', header=False, index=False)
