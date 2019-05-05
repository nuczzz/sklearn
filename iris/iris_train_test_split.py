from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def iris_train_test_split():
	iris_info = load_iris()
	x_train, x_test, y_train, y_test = train_test_split(iris_info.data, iris_info.target, test_size=0.2, random_state=22)
	print("x_train: \n", x_train)


if __name__ == "__main__":
	iris_train_test_split()

