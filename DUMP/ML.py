import umap 
import pandas as pd

starData = pd.read_csv('ALL_AND_FINAL_EXTRA.csv')
starData = starData.drop_duplicates(subset='SPECID')
flameExists = starData['flameMask']

# lowAgeMask = starData['AGE'] < 10
# midAgeMask = starData['AGE'] >= 10 & starData['AGE'] <13.5
# highAgeMask = starData['AGE'] >= 13.5

X_train = starData[flameExists]
X_pred = starData[~flameExists]

usefulColumns = [    
    'ALPHA_FE','TEFF','LOGG', 'FEH', 'GAIA_G', 'GAIA_BP', 'GAIA_RP', 'PARALLAX', 'MK_COMB', 'age_flame'
]
X_train = X_train[usefulColumns]
X_train = X_train.dropna()
lowAgeMask = X_train['age_flame'] < 10
midAgeMask = (X_train['age_flame'] > 10) & (X_train['age_flame'] < 13.5)
lowAgeX = X_train[lowAgeMask]
lowAgeY = X_train[lowAgeMask]['age_flame']
midAgeX = X_train[midAgeMask]
midAgeY = X_train[midAgeMask]['age_flame']
lowAgeX =lowAgeX.drop(['age_flame'], axis=1)
midAgeX =midAgeX.drop(['age_flame'], axis=1)
# ages = X_train['age_flame']
Y_train = X_train['age_flame']
X_train = X_train.drop(['age_flame'], axis=1)
# Prediction
X_pred = X_pred[usefulColumns]
X_pred = X_pred.drop(['age_flame'], axis=1)
X_pred = X_pred.dropna()

# print(X_pred)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# X_tr, X_te, y_tr, y_te = train_test_split(X_pred, lowAgeY, test_size=0.2, random_state=42)

# model = RandomForestRegressor(
#     n_estimators=100,
#     n_jobs=-1,    
#     verbose=1      
# )

# model.fit(X_train, Y_train)

# joblib.dump(model, 'OUTPUT\\model_ALL.pkl')
# model = joblib.load('OUTPUT\\model_ALL.pkl')
# yp = model.predict(X_pred)

# yyAgesNotInFlame = starData[~flameExists]['AGE']

# # plt.hexbin(yp, midAgeY, gridsize=32)
# # plt.colorbar()
# # plt.show()
# plt.hist(yyAgesNotInFlame, bins=20, alpha=0.5, color='red')
# plt.hist(yp, bins=20, alpha=0.5, color='green')
# plt.show()
