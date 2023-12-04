import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    PrecisionRecallDisplay,
    RocCurveDisplay
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold,
    RepeatedKFold,
    cross_val_score
)
from xgboost import XGBClassifier

level = 6
data_type = 'relative'

df = pd.read_csv(f'feature-table-{data_type}-se-{level}.tsv',
                 sep='\t', index_col=0, header=1).T

df_metadata = pd.read_csv('metadata.tsv',
             sep='\t', index_col=0)

df_metadata['host_sex'] = [1 if i=='male' else 0 for i in df_metadata['host_sex']]
df_metadata['health_state']= [1 if i=='Parkinsons' else 0 for i in df_metadata['health_state']]

# feature selection
# discard less prevalent features
filt = (df>0).sum(axis=0)/len(df.index)>0.05
df_filt = df.loc[:,filt]


fig, ax = plt.subplots(figsize=(4,3))
bins=np.arange(0, 1.01, 0.05)
plt.hist((df>0).sum(axis=0)/len(df),
         bins=bins,
         label='Prevalence < 5%')
plt.hist((df_filt>0).sum(axis=0)/len(df),
         bins=bins,
         label='Prevalence >= 5%')
plt.xlabel('Feature prevalence', fontsize=12)
plt.ylabel('Feature count', fontsize=12)
plt.legend(edgecolor='w')
plt.tight_layout()
plt.savefig(f'figures/count_vs_prevalence_{level}.png', dpi=300)

X = pd.concat([df_filt, df_metadata[['host_Age', 'host_sex']]], axis=1)
y = df_metadata['health_state'][X.index]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=763)

ini_clf = dict()
ini_clf['RandomForest'] = RandomForestClassifier(n_jobs=-1)
ini_clf['AdaBoost'] = AdaBoostClassifier()
ini_clf['CatBoost'] = CatBoostClassifier(thread_count=-1, verbose=False)
ini_clf['LightGBM'] = LGBMClassifier(n_jobs=-1, verbose=-1, importance_type='gain')
ini_clf['XGBoost'] = XGBClassifier()

params = dict()

np.random.seed(763)
estimators = np.random.randint(50, 500, 50)

# defining hyperparameter search space
params['RandomForest'] = {
    'n_estimators': estimators,
    'max_depth': np.arange(3,10),
    'min_samples_leaf': np.arange(5,20,2)
}

params['AdaBoost'] = {
    'n_estimators': estimators,
    'learning_rate': np.logspace(-4, 0, 10),
}

params['CatBoost'] = {
    'iterations': estimators,  
    'depth': np.arange(3,10),
    'learning_rate': np.logspace(-4, 0, 10),
}

params['LightGBM'] = {
    'n_estimators': estimators,  
    'max_depth': np.arange(3,10),
    'learning_rate': np.logspace(-4, 0, 10),
}

params['XGBoost'] = {
    'n_estimators': estimators,  
    'max_depth': np.arange(3,10),
    'learning_rate': np.logspace(-4, 0, 10),
}


# loop over the different algorithms and record the scores
clf = dict()
repeated_score = dict()

random_state = 763
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
rcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=random_state)

clfs = ['RandomForest', 'AdaBoost', 'CatBoost', 'LightGBM', 'XGBoost']

for classifier in clfs:
    clf[classifier] = RandomizedSearchCV(ini_clf[classifier],
                                         params[classifier],
                                         cv=cv,
                                         random_state=random_state,
                                         n_jobs=-1)
    
    clf[classifier].fit(X=X_train, y=y_train)

    repeated_score[classifier] = cross_val_score(clf[classifier],
                                                 X=X_train,
                                                 y=y_train,
                                                 cv=rcv,
                                                 n_jobs=-1)
    

# plot the ROC and PR curves
fig, (ax_roc, ax_pr) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
for c in clfs:
    RocCurveDisplay.from_estimator(clf[c], X_test, y_test, ax=ax_roc, name=c)
    PrecisionRecallDisplay.from_estimator(clf[c], X_test, y_test, ax=ax_pr, name=c)
    
ax_roc.set_title('Receiver Operating Characteristic (ROC) curves')
ax_roc.grid(linestyle='--')

ax_pr.set_title('Precision-Recall curves')
ax_pr.grid(linestyle='--')
plt.savefig(f'figures/ROC_and_PR_curves_{data_type}_{level}.png', dpi=300)

acc = dict()
p_r_f = defaultdict(lambda: defaultdict(int))

# test the results (evaluation)
# get confusion matrix
for c in clf:
    y_pred = clf[c].predict(X_test)
    acc[c] = accuracy_score(y_true=y_test, y_pred=y_pred)
    for avg in ['macro','micro','weighted']:
        *p_r_f[avg][c], _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, average=avg)


for avg in ['macro','micro','weighted']:
    df_a = pd.DataFrame(acc, index=['Accuracy'])
    df_prf = pd.DataFrame(p_r_f[avg], index=['Precision', 'Recall', 'F-measure'])
    df_aprf = pd.concat([df_a, df_prf])
    df_aprf.T.to_csv(f'tables/aprf_{avg}_{data_type}_{level}.tsv', sep='\t')


# vmin and vmax for colorbar
vmax = 0
vmin = np.inf
for c in clf:
    y_pred = clf[c].predict(X_test)
    matrix = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel() # tn, fp, fn, tp
    if max(matrix) > vmax:
        vmax = max(matrix)
    if min(matrix) < vmin:
        vmin = min(matrix)

nrows = 2
ncols = 3
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,7))

idx = 0
for row in np.arange(nrows):
    for col in np.arange(ncols):
        if idx >= len(clfs):
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            for spine in (ax[row, col].spines).keys():
                ax[row, col].spines[spine].set_visible(False)
        else:
            cm = np.random.random((1,1))
            im = ax[row, col].imshow(cm, vmin=vmin, vmax=vmax)
            
            ConfusionMatrixDisplay.from_estimator(clf[clfs[idx]],
                                                  X_test,
                                                  y_test,
                                                  colorbar=False,
                                                  ax=ax[row, col],
                                                  im_kw=dict(vmin=vmin, vmax=vmax))
            ax[row, col].set_title(clfs[idx])
            idx+=1

# fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.75, 0.15, 0.05, 0.3])
# [a, b, c, d]
# (a,b) is the point in southwest corner of the rectangle which we create.
# c represents width and d represents height of the respective rectangle.

fig.colorbar(im, cax=cbar_ax)
plt.savefig(f'figures/confusion_matrix_{data_type}_{level}.png', dpi=300)

df = pd.DataFrame(repeated_score)
df.to_csv(f'tables/repeated_score_{data_type}_{level}.tsv', sep='\t')

fig, ax = plt.subplots(figsize=(5,4))
plt.boxplot(df,
            patch_artist=True,
            boxprops=dict(facecolor = 'lightblue'))

plt.xticks([i + 1 for i in range(len(df.columns))],
           df.columns,
           rotation=30,
           ha='right')

plt.ylabel('Repeated Cross-Validation Scores', fontsize=12)
plt.grid(ls='--', axis='y')
plt.tight_layout()
plt.savefig(f'figures/repeated_cv_boxplot_{data_type}_{level}.png', dpi=300)