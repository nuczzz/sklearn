from sklearn.feature_extraction.text import CountVectorizer


def text_count_vectorizer():
	text = ["life is short, I like python", "life is too long, I dilike python"]
	
	#CountVectorizer(stop_word=[])
	#create a new sparse transform instance
	transform_default = CountVectorizer()
	data_default = transform_default.fit_transform(text)
	print(data_default)
	print("\n")
	print(data_default.toarray())
	

if __name__ == "__main__":
	text_count_vectorizer()

