import pandas as pd
import numpy as np


pd1 = pd.read_csv('submission_lgbm_10995.177929_random_state 42 boosting dart n_estimators 40000 max_depth 8 metric mae lambda_l2 4 learning_rate 0.05 drop_rate 2.5e-05_fold0_0.csv')
pd2 = pd.read_csv('submission_lgbm_10987.497519_random_state 42 boosting dart n_estimators 40000 max_depth 7 metric mae lambda_l2 4 learning_rate 0.05 drop_rate 2.5e-05_fold1_0.csv')
pd1['Predicted'] = 0.5 * pd1['Predicted'] + 0.5 * pd2['Predicted']
pd1.to_csv('final.csv', index=None)