from sklearn.datasets import load_iris


def data_sets_demo():
    # get data set
    iris = load_iris()
    print(iris)

    return None


if __name__ == "__main__":
    data_sets_demo()
