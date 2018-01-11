import xgboost as xgb
from numpy import genfromtxt
import numpy as np
import datetime
import pandas as pd
from pyspark.mllib.util import MLUtils

product_labels = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
               'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']


def runXGB(train_X, train_y, seed_val=0):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 8
    param['silent'] = 1
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 10
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = 250

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds,2)
    return model

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("Reading Data..")
    X_train = MLUtils.loadLibSVMFile(sc, 'X_train.csv').repartition()
    Y_train = MLUtils.loadLibSVMFile(sc, 'Y_train.csv').repartition()
    X_test = MLUtils.loadLibSVMFile(sc, 'X_test.csv').repartition()

    print("Building model..")
    model = runXGB(X_train, Y_train, seed_val=0)

    print("Predicting..")
    xgtest = xgb.DMatrix(X_test)
    preds = model.predict(xgtest)

    print(datetime.datetime.now() - start_time)

    print("Getting the top products..")
    target_cols = np.array(product_labels)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)[:, :7]
    test_id = np.array(pd.read_csv("test_final.csv", usecols=['ncodpers'])['ncodpers'])
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    out_df.to_csv('test_submission.csv', index=False)
    print(datetime.datetime.now() - start_time)
    print "XXXXX"