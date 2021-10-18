
# id 17301131
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets.base import Bunch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Import the dependencies for logistic regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#loading the dataset as data frame 
v = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/glass source classification dataset.csv') 




#data preprocess
#number of missing values 

v.isnull().sum() 

#handling missing values
from sklearn.impute import SimpleImputer
impute  = SimpleImputer(missing_values= np.nan,strategy = 'mean')
impute.fit(v[['Ca']])
v['Ca'] = impute.transform(v[['Ca']])
v.isnull().sum()

#encoding catagorical features

enc = LabelEncoder()
v['Type'] = enc.fit_transform(v['Type'])
v['Ba'] = enc.fit_transform(v['Ba'])
v['Fe'] = enc.fit_transform(v['Fe'])

v.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)     
v.drop(["a"], axis=1, inplace=True)

                                               
#scaling data
X = v.drop("Type", axis=1)
y= v["Type"]
X_trai, X_tes, y_train, y_test= train_test_split(X, y, stratify=y, test_size=0.2)                                                 
scaler = MinMaxScaler()
scaler.fit(X_trai)
X_train = scaler.transform(X_trai)
X_test = scaler.transform(X_tes)


#pre pca
from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)
SVC1 = svc.score(X_test, y_test)
print("Training accuracy of the model is {:.2f}".format(svc.score(X_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(SVC1))
predictions = svc.predict(X_test)
print(predictions)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train, y_train)
RFC1 = rfc.score(X_test, y_test)
print("The Training accuracy of the model is {:.2f}".format(rfc.score(X_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(RFC1))

from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=100000)
nnc.fit(X_train, y_train)
MLP1 = nnc.score(X_test, y_test)
print("The Training accuracy of the model is {:.2f}".format(nnc.score(X_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(MLP1))
predictions = nnc.predict(X_test)
print(predictions)

from sklearn.decomposition import PCA 
pca = PCA(n_components=4)
data = v.iloc[:214, :].to_numpy()
principal_components= pca.fit_transform(data)
principal_df = pd.DataFrame(data=principal_components, columns=["principle component 1", "principle component 2","principle component 3", "principle component 4"])
main_df=pd.concat([principal_df, v[["Type"]]], axis=1)

#post pca
X = main_df.drop("Type", axis=1)
y= main_df["Type"]
X_train, X_test, y_train, y_test= train_test_split(X, y, stratify=y, test_size=0.2) 


from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)
SVC2 = svc.score(X_test, y_test)
print("Training accuracy of the model is {:.2f}".format(svc.score(X_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(SVC2))
predictions = svc.predict(X_test)
print(predictions)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train, y_train)
RFC2 = rfc.score(X_test, y_test)
print("The Training accuracy of the model is {:.2f}".format(rfc.score(X_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(RFC2))


from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=100000)
nnc.fit(X_train, y_train)
MLP2 = nnc.score(X_test, y_test)
print("The Training accuracy of the model is {:.2f}".format(nnc.score(X_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(MLP2))
predictions = nnc.predict(X_test)
print(predictions)



#bar chart
import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

# set height of bar
svcc = [SVC1, SVC2]
rfcc = [RFC1, RFC2]
mlpp = [MLP1, MLP2]

# Set position of bar on X axis
br1 = np.arange(len(svcc))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, IT, color ='r', width = barWidth,
		edgecolor ='grey', label ='svcc')
plt.bar(br2, ECE, color ='g', width = barWidth,
		edgecolor ='grey', label ='rfcc')
plt.bar(br3, CSE, color ='b', width = barWidth,
		edgecolor ='grey', label ='mlpp')

# Adding Xticks
plt.xlabel('Models', fontweight ='bold', fontsize = 15)
plt.ylabel('Accuracy in test dataset', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(svcc))],
		['pre-PCA', 'post-PCA'])

plt.legend()
plt.show()




