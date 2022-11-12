import json
import numpy as np
from sklearn.metrics import confusion_matrix

def disparity_score(ytrue, ypred):
    cm = confusion_matrix(ytrue,ypred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    all_acc = list(cm.diagonal())
    return max(all_acc) - min(all_acc)

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
