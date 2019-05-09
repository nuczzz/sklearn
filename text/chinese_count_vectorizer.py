#coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer
import jieba


def chinese_count_vectorizer():
	text = ["精确模式，试图将句子最精确地切开，适合文本分析;",
		"全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；",
		"搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词"]
	data = []
	for sentence in text:
		data.append(cut_word(sentence))

	transform = CountVectorizer()
	data_new = transform.fit_transform(data)
	print(data_new.toarray())
	print("\n")
	print(transform.get_feature_names())


def cut_word(text):
	"""
	我爱北京天安门 -> 我 爱 北京 天安门
	"""
	return " ".join(list(jieba.cut(text)))


if __name__ == "__main__":
	chinese = "我爱北京天安门"
	#print(cut_word(chinese))
	chinese_count_vectorizer()
