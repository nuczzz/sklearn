from sklearn.datasets import load_iris


def get_iris_info():
	iris_info = load_iris()
	print("iris datasets: \n", iris_info)
	print("iris datasets description: \n", iris_info["DESCR"])
	print("iris datasets field name: \n", iris_info.feature_names)
	print("iris datasets filed: \n", iris_info.data, iris_info.data.shape)


if __name__ == "__main__":
	get_iris_info()
