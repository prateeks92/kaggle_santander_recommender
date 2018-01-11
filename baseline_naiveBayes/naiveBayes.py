import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
from sklearn.naive_bayes import BernoulliNB
import time
import gzip
import warnings
warnings.filterwarnings("ignore")

from dataset import SantanderDataset
from average_precision import mapk

dataset_root = ''
dataset = SantanderDataset(dataset_root)

def train_bnb_model(msg):
    msg_copy = msg.copy()
    msg_copy['train'] = True
    if not 'month' in msg_copy.keys():
        msg_copy['month'] = msg_copy['train_month']
    ret = dataset.get_data(msg_copy)
    input_data, output_data = ret[0:2]
    bnb = BernoulliNB(alpha=1e-2)
    bnb.partial_fit(input_data, output_data, classes = range(24))
    return bnb

def create_prediction(bnb, msg):
    msg_copy = msg.copy()
    msg_copy['train'] = False
    if not 'month' in msg_copy.keys():
        msg_copy['month'] = msg_copy['eval_month']
    ret = dataset.get_data(msg_copy)
    input_data, output_data, previous_products = ret
    rank = bnb.predict_proba(input_data)
    filtered_rank = np.equal(previous_products, 0) * rank
    predictions = np.argsort(filtered_rank, axis=1)
    predictions = predictions[:,::-1][:,0:7]
    return predictions, output_data


def naive_bayes_workflow(msg):
    if type(msg['eval_month']) is not list:
        msg['eval_month'] = [msg['eval_month']]
    bnb = train_bnb_model(msg)
    scores = []
    for month in msg['eval_month']:
        msg_copy = msg.copy()
        msg_copy['month'] = month
        predictions, output_data = create_prediction(bnb, msg_copy)
        score = mapk(output_data, predictions)
        scores.append(score)
    
    return scores, bnb


def create_submission(filename, msg, 
                        verbose=False):
    test_month = 17
    ret = naive_bayes_workflow(msg)
    scores = ret[0]
    bnb = ret[1]
    msg['month'] = test_month
    predictions, output_data = create_prediction(bnb, msg)
    if verbose: print 'Creating text...'
    text='ncodpers,added_products\n'
    for i, ncodpers in enumerate(dataset.eval_current[dataset.eval_current.fecha_dato == test_month].ncodpers):
        text += '%i,' % ncodpers
        for j in predictions[i]:
            text += '%s ' % dataset.product_columns[j]
        text += '\n'
    if verbose: print 'Writing to file...'
    with open(dataset_root + 'submissions/%s.csv' % filename, 'w') as f:
        f.write(text)
    
    return scores


start_time = time.time()
msg = {'train_month': range(1,5),
       'eval_month': 5,
      'input_columns': dataset.categorical_columns,
      'use_product': True,
      'use_change': False}
print "Generating Prediction"
print naive_bayes_workflow(msg)[0]
print time.time()-start_time


start_time = time.time()
msg = {'train_month': [ 6,8,11,12,14,15],
       'eval_month': [5,16],
      'input_columns': ['ind_empleado','pais_residencia','age','indrel',
                        'indresi','indext','canal_entrada','renta'],
      'use_product': True,
      'use_change': True}
print "Creating Submission File"
print create_submission('NaiveBayes',msg)
print time.time()-start_time