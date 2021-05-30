import numpy as np
import pandas as pd
from tqdm import tqdm
from logging import StreamHandler, DEBUG, INFO, Formatter, FileHandler, getLogger
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
import warnings
import sys
sys.path.append('..')
from inputs import load_train_data, load_test_data

if __name__ == '__main__':
    warnings.simplefilter('ignore', UserWarning)

    MODEL_NAME = 'LOGISTIC_REGRESSION'
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
    param_grid = {'penalty': ['l2', 'none'],
                  'C': np.linspace(0.3, 2.1, 8).tolist(),
                  'fit_intercept': [True, False],
                  'class_weight': ['balanced', None],
                  'max_iter': [10000],
                  'random_state': [1]}
    min_score = 100
    min_params = None

    for params in tqdm(list(ParameterGrid(param_grid))):
        logger.info(f'param: {params}')

        lst_logloss_score = []
        all_preds = np.zeros(shape=train_y.shape[0])

        for tr_idx, va_idx in cv.split(train_x, train_y):
            tr_x = train_x.iloc[tr_idx]
            va_x = train_x.iloc[va_idx]

            tr_y = train_y.iloc[tr_idx]
            va_y = train_y.iloc[va_idx]

            model = LogisticRegression(**params)
            model.fit(tr_x, tr_y)
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

    model = LogisticRegression(**min_params)
    model.fit(train_x, train_y)
    preds_test = model.predict(test_x)

    df_submit = pd.read_csv('../inputs/sample_submission.csv', header=None)
    df_submit[1] = preds_test
    df_submit.to_csv('logistic.csv', header=False, index=False)
