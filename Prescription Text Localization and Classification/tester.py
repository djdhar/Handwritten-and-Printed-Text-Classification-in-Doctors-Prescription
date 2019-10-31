import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from joblib import dump, load
import pickle

import pdb

clf = load("data.joblib")
mpred = [[40,113,0.353982300884956,	0.453097345132743,	0.008849557522124]]
dj = clf.predict(mpred)
print(dj[0])