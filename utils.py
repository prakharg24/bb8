import json
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle
import torch
import scipy.stats as stats

def disparity_score(ytrue, ypred):
    cm = confusion_matrix(ytrue,ypred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)
    all_acc = list(cm.diagonal())
    # print(all_acc)
    return max(all_acc) - min(all_acc)

def randomness_score(ypred, num_classes):
    yexp = [len(ypred)/num_classes for i in range(num_classes)]
    yobs = [np.sum(ypred==i) for i in range(num_classes)]
    chi_sq_test = stats.chisquare(f_obs=yobs, f_exp=yexp)

    if chi_sq_test.pvalue > 0.05:
        return True
    else:
        return False

def getScore(results):
    acc = results['accuracy']
    disp = results['disparity']
    ad = 2*acc['gender']*(1-disp['gender']) + 4*acc['age']*(1-disp['age']**2) + 10*acc['skin_tone']*(1-disp['skin_tone']**5)
    return ad

def create_submission(results, submission_name, submission_filename):
    submission = {
        'submission_name': submission_name,
        'score': getScore(results),
        'metrics': results
    }

    print("Submission Score : ", submission['score'])
    with open(submission_filename, "w") as f:
        json.dump(submission, f, indent=4)

def load_state_dict(model, fname):
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if 'fc' in name:
            continue
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
