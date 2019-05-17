from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#1.standard
#2.normalization: x'=(x-min)/(max-min), x"=x'*(mx-mi)+mi


def preprocessing_demo():
	#get data from file
	data = pd.read_csv("data.txt")
	data = data.iloc[:, :3]
	
	#
	transfer = MinMaxScaler()
	data_new = transfer.fit_transform(data)
	print(data_new)


if __name__ == "__main__":
	preprocessing_demo()
