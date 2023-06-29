import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Outcome = 1 Diabet
# Outcome = 0 Healthy
data = pd.read_csv("diabetes.csv")
data.head()

diabetes = data[data.Outcome == 1]
healthy_people = data[data.Outcome == 0]


# Let's make an example drawing just by looking at glourose for now:
# At the end of our program, our machine learning model will make a prediction by looking not only at glucose, but also at all other data..
plt.scatter(healthy_people.Age, healthy_people.Glucose, color="green", label="healthy", alpha = 0.4)
plt.scatter(diabetes.Age, diabetes.Glucose, color="red", label="diabet", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()



# let's determine the x and y axes
y = data.Outcome.values
x_raw_data = data.drop(["Outcome"],axis=1)   
# We remove the outcome column(dependent variable) and leave only independent variables
# Because the KNN algorithm will group within x values..


# we are doing normalization - we are updating all of them so that the values in the x_ham_ Decal are only between 0 and 1
# If we do not normalize in this way, high numbers will crush small numbers and may mislead the KNN algorithm!
x = (x_raw_data - np.min(x_raw_data.values))/(np.max(x_raw_data.values)-np.min(x_raw_data.values))

# before
print("Raw data before normalization:\n")
print(x_raw_data.head().values)


# after 
print("\n\n\nThe data that we will provide to artificial intelligence for training after normalization:\n")
print(x.head())
    

# we separate our test data with our train data
# our train data will be used to learn how the system distinguishes between a healthy person and a sick person
# if our test data is, let's see if our machine learning model can accurately distinguish between sick and healthy people
# it will be used for testing...
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=1)

# we are creating our knn model.
knn = KNeighborsClassifier(n_neighbors = 6) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("Verification test result of our test data for K=3 ", knn.score(x_test, y_test))



# what should k be ?
# let's determine the best k value..
sayac = 1
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors = k)
    knn_yeni.fit(x_train,y_train)
    print(sayac, "  ", "Accuracy rate: %", knn_yeni.score(x_test,y_test)*100)
    sayac += 1
    


# For a new patient forecast:
from sklearn.preprocessing import MinMaxScaler

# we are doing normalization - we used MinMax scaler to make normalization faster...
sc = MinMaxScaler()
sc.fit_transform(x_raw_data.values)
new_prediction = knn.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
new_prediction[0]
if new_prediction==1:
    print("Diabet")
else:
    print("Healthy")



