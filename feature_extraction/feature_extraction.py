from sklearn.feature_extraction import DictVectorizer


def dict_feature_extraction():
	data = [{"city": "beijing", "temperature": 100},
		{"city": "shanghai", "temperature": 60},
		{"city": "shenzhen", "temperature": 30}]
	
	# new default transform instance
	transform_default = DictVectorizer() #default param, sparse matrix will be returned
	data_default = transform_default.fit_transform(data)
	print("sparse matrix:")
	print(data_default)

	# new normal transform instance
	transform_normal = DictVectorizer(sparse=False) #normal matrix will be returned
	data_normal = transform_normal.fit_transform(data)
	print("\nnormal matrix:")
	print(data_normal)
	

if __name__ == "__main__":
	dict_feature_extraction()
