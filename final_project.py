import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("car_dataset_india.csv")
# print(df.head())

df['Service_Cost'] = df['Service_Cost'].ffill()
# print(df.head())

le = LabelEncoder()
for column in ['Brand','Model','Fuel_Type','Transmission']:
    df[column] = le.fit_transform(df[column])

# print(df.head())

sc = StandardScaler()
for column in ['Year','Price','Engine_CC','Service_Cost']:
          df[column] = sc.fit_transform(df[[column]])
# print(df)

"""supervised learning"""

x = df.drop(columns=['Car_ID','Price'])
y = df['Price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(x_train)
# X_test_scaled = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print(r2)

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["Engine_CC"], y=df["Price"], alpha=0.6, color="blue")
plt.title("Engine Capacity vs. Price")
plt.show()

"""-------------------------------------------------"""
"""unsupervised learning"""
from sklearn.cluster import KMeans

# x = df.columns['Engine_CC','Price']
# print(x)


# model = KMeans(n_clusters=3,random_state=42)
# fitted_model = model.fit_predict(x)
#
# plt.scatter(x[:0],x[:6],c=fitted_model,cmap='viridis',marker='*',label='KMeans',alpha=0.9)
# plt.show()

