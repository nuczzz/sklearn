from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd

#1.normalization: x'=(x-min)/(max-min), x"=x'*(mx-mi)+mi

#2.standard: x'=(x-avg)/std
#for every column, 'avg' is average of value, and 'std' is standard deviation


def preprocessing_demo():
	#get data from file
	data = pd.read_csv("data.txt")
	data = data.iloc[:, :3]
	
	#standard
	transfer = StandardScaler()
	data_new = transfer.fit_transform(data)
	print(data_new)


if __name__ == "__main__":
	preprocessing_demo()
