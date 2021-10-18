import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets.base import Bunch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

v = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/glass source classification dataset.csv') 
v.head()
v
#loading the dataset as data frame 

#number of missing values 

v.isnull().sum() 

#handling missing values
from sklearn.impute import SimpleImputer
impute  = SimpleImputer(missing_values= np.nan,strategy = 'mean')
impute.fit(v[['Ca']])
v['Ca'] = impute.transform(v[['Ca']])
v.isnull().sum()

#finding catagorical features
v.info()

v['Type'].unique()

# Transform the Type_desc column
Type_enc = pd.get_dummies(v['Type'])
v = v.drop('Type',axis = 1)
v = v.join(Type_enc)

enc = LabelEncoder()
v['Ba'] = enc.fit_transform(v['Ba'])
v['Fe'] = enc.fit_transform(v['Fe'])
print(v[['Fe']].head(10))
print(v[['Ba']].head(10))
v.head()

v.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)     
v.drop(["a"], axis=1, inplace=True)
v.head()
v.shape
v.info()

# task 6 and 7
data = v.iloc[:213, 0:9] .to_numpy()
target = v.iloc[:213,9:].to_numpy() # task7. Splitting the dataset into features and labels
feature_names = ['RI', 	'Na', 	'Mg', 	'Al', 	'Si', 	'K', 	'Ca', 	'Ba', 	'Fe' 	] 
target_names = ['building_window glass', 	'container glass', 	'headlamp glass', 	'tableware glass', 	'vehicle_window glass']
D = Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)
X_train, X_test, y_train, y_test = train_test_split(D.data, D.target,random_state=1)                                                   
print(X_train.shape)
print(X_test.shape)
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)



