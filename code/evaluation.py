import os
import datetime
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

import read_data
SAVE_PATH = read_data.read_config()['savepath']

def save_results(results):
    save_path = os.path.abspath(SAVE_PATH)
    results_dir = datetime.date.today().strftime("%Y-%m-%d")
    save_path = os.path.join(save_path, results_dir)
    save_file_pck = os.path.join(save_path, 'results.pck')
    save_file_txt = os.path.join(save_path, 'results.txt')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(save_file_pck, 'wb') as f:
        pickle.dump(results, f)

    with open(save_file_txt, 'w') as f:
        for k, v in results.items():
            f.write(k + '\n')
            f.write('='*10 + '\n')
            f.write(str(v))
            f.write('\n' + '='*10 + '\n' + '\n')

def experiment_code(data_name, feats_name, clf_name):
    return "{}_{}_{}".format(data_name, feats_name, clf_name)


def select_classifier(clf_name):
    if clf_name == 'LogR':
        return LogisticRegression(solver='liblinear')
    elif clf_name == 'LinSVM':
        return LinearSVC()

def get_preds(clf, X, y):
    np.random.seed(42)
    preds = cross_val_predict(clf, X, y, cv=10)
    return preds

def evaluate_preds(labels, preds):
    class_report = classification_report(labels, preds, output_dict=True)

    cols = ['precision', 'recall', 'f1-score', 'support']
    r = pd.DataFrame(data=class_report).T
    r.columns = cols
    r[cols[:-1]] = r[cols[:-1]] * 100
    r = r.round(2)
    return r

def predict_evaluate(clf, X, y):
    preds = get_preds(clf, X, y)
    result = evaluate_preds(y, preds)
    return result
