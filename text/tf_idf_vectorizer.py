from sklearn.feature_extraction.text import TfidfVectorizer
#tf: term frequency(=word_count/total_count)
#idf: inverse document frequency(=lg(total_file/word_file))
#tf-idf: importance of some word(=tf*idf)
#API: sklearn.feature_extraction.text.TfidfVectorizer(stop_word=[])

def tf_idf():
	text = ["life is short, I like python", "life is too long, I dilike python"]
	
	#create a new sparse transform instance
	transform_default = TfidfVectorizer()
	data_default = transform_default.fit_transform(text)
	print(data_default)
	print("\n")
	print(data_default.toarray())
	print("\n")
	print(transform_default.get_feature_names())


if __name__ == "__main__":
	tf_idf()
