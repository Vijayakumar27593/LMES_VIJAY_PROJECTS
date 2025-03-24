"""import libraries"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

#To read the csv file
df = pd.read_csv("car_dataset_india.csv")
print(df.head())

#finding data information like columns name, int or float or object, null values count
df.info()
#describe for all coumns total, mean like...
df.describe()

# #finding nan columns
nan_column = df['Service_Cost'].isna()
print(nan_column)

#finding sum of nan columns
nan_column_sum = df['Service_Cost'].isna().sum()
print(nan_column_sum)

#filling up the null values from previous value
df['Service_Cost'] = df['Service_Cost'].ffill()
print(df.head())

#convert categorical data into numerical data
le = LabelEncoder()
for column in ['Brand','Model','Fuel_Type','Transmission']:
    df[column] = le.fit_transform(df[column])
print(df.head())

#standardize the value for accuracy
sc = StandardScaler()
for column in ['Year','Price','Engine_CC','Service_Cost']:
          df[column] = sc.fit_transform(df[[column]])
print(df)

"""_______supervised learning______"""

x = df.drop(columns=['Car_ID','Price'])
y = df['Price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('Accuracy Score:',r2)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

param_grid = {
    'fit_intercept': [True,False],

}

grid_search = GridSearchCV(
    estimator= model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=1
)
grid_search.fit(x_train,y_train)

print('Best param',grid_search.best_params_)
print('Best Score',grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred_1 = best_model.predict(x_test)

r2 = r2_score(y_test, y_pred_1)
print('Accuracy Score-2:',r2)


# plt.figure(figsize=(10, 5))
# # sns.scatterplot(x=df["Engine_CC"], y=df["Price"], alpha=0.6, color="blue")
# # plt.title("Engine Capacity vs. Price")
# # plt.show()

plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
plt.title('Supervised learning')
plt.show()

"""__________unsupervised learning______________"""
from sklearn.cluster import KMeans

df_car = pd.read_csv("car_dataset_india.csv",index_col ='Car_ID')

plt.figure(1 , figsize = (10 , 5) )
plt.title('Unsupervised Learning')
sns.scatterplot(
    data=df_car,
    x="Mileage",
    y="Price",
    hue="Brand",
    size="Seating_Capacity",
    palette="Set2"
)
plt.show()


import numpy as np
from sklearn import preprocessing
X = df_car.iloc[:,[5,6]].values
X_norm = preprocessing.normalize(X)

X_final = pd.DataFrame(X_norm)

def elbow_plot(data,clusters):
    inertia = []
    for n in range(1, clusters):
        algorithm = KMeans(
            n_clusters=n,
            init="k-means++",
            random_state=125,
        )
        algorithm.fit(data)
        inertia.append(algorithm.inertia_)
    # Plot
    plt.plot(np.arange(1 , clusters) , inertia , 'o')
    plt.plot(np.arange(1 , clusters) , inertia , alpha = 0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Unsupervised Learning')
    plt.show()

elbow_plot(X_norm,10)

model = KMeans(n_clusters=3,random_state=42)
model.fit_predict(X_final)
labels = model.labels_


# sns.scatterplot(X_norm[:0],X_norm[:1],c=fitted_model,cmap='viridis',marker='*',label='KMeans',alpha=0.9)
sns.scatterplot(data=X_final, hue = labels, palette="Set2")
plt.show()

