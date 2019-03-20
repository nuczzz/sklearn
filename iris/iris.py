from sklearn.datasets import load_iris


def data_sets_demo():
    """
    load iris data set return Bunch data,
    and Bunch is implemented from dict.
    Bunch has fields:
    DESCR: description of data set
    data: n*n array of data
    target: array of target(label)
    target_names: array of target names
    feature_names: feature names
    """
    iris = load_iris()

    # print("iris descr: \n", iris.DESCR)
    print("iris descr: \n", iris["DESCR"])

    print("iris data: \n", iris.data)

    print("iris target: \n", iris.target)

    print("iris target names: \n", iris.target_names)

    print("iris feature names: \n", iris.feature_names)

    return None


if __name__ == "__main__":
    data_sets_demo()
