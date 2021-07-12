import io
from itertools import islice

import nltk
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import wordnet, pos_tag
from nltk.stem.lancaster import LancasterStemmer
from tqdm import tqdm


def load_vectors(fname, limit):  # будем работать с Word2Vec, поэтому нам нужна функция загрузки векторов
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(islice(fin, limit), total=limit):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


def get_wordnet_pos(treebank_tag):  # для лемматизации необходимо разметить слова
    my_switch = {
        'J': wordnet.wordnet.ADJ,
        'V': wordnet.wordnet.VERB,
        'N': wordnet.wordnet.NOUN,
        'R': wordnet.wordnet.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.wordnet.NOUN


def my_lemmatizer(sent):  # функция лемматизации текста
    lemmatizer = WordNetLemmatizer()
    tokenized_sent = sent.split()
    pos_tagged = [(word, get_wordnet_pos(tag))
                  for word, tag in pos_tag(tokenized_sent)]
    return ' '.join([lemmatizer.lemmatize(word, tag)
                     for word, tag in pos_tagged])


def text2vec(text):  # функция перевода текста в вектор
    words = text.split()
    return sum(list(map(lambda w: np.array(list(vecs.get(w, zero))), words))) / len(words)


# загружаем необходимые словари/данные
nltk.download('stopwords')
sw_eng = set(stopwords.words('english'))

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

vecs = load_vectors('crawl-300d-2M.vec', 100000)
zero = sum(vecs.values()) / len(vecs)

stemmer = LancasterStemmer()


class Document:
    def __init__(self, title, text, title_words, text_words):
        self.title = title
        self.text = text
        self.title_words = title_words  # включаем в документ обработанные данные
        self.text_words = text_words

    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text[:150] + ' ...']


index = []
df = pd.read_csv('../../../../../Tinkoff.Generation/search/articles1.csv')


def build_index():
    # считывает сырые данные и строит индекс
    for i in range(df.shape[0]):
        index.append(Document(df['title'][i],
                              df['content'][i][:150],
                              prep(df['title'][i]),
                              prep(df['content'][i][:150])))


def score(query, document):
    # возвращает скор для пары запрос-документ
    # больше -- релевантнее

    title = document.title_words
    text = document.text_words
    query = prep(query)

    # чтобы положение слова из запроса оказывало влияние
    # на релевантность документа, будем суммировать индексы включений
    title_sum = 0
    text_sum = 0
    for word in query.split():
        if word in title.split():
            title_sum += list(title.split()).index(word)
        if word in text.split():
            text_sum += list(text.split()).index(word)

    title = text2vec(title)
    text = text2vec(text)
    query = text2vec(query)

    # важно, чтобы запрос соответствовал документу, поэтому учитываем "похожесть" векторов данных
    corr1 = np.linalg.norm(title - query)
    corr2 = np.linalg.norm(text - query)

    # делим 1 на суммы, чтобы документы, где слова из запроса встречаются раньше, были релевантнее
    return 1 / (title_sum + 1) + 1 / (text_sum + 1) + 0.2 * corr1 + 0.05 * corr2


def prep(text):  # функция обработки текстовых данных
    # (очиска от знаков -> лемматизация -> стемминг -> очистка от stop words)
    text = ' '.join(re.findall(r'\w+', text.lower()))
    text = my_lemmatizer(text)
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    text = ' '.join([word for word in text.split() if not word in sw_eng])

    return text


def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    query = prep(query)
    inv_index = {x: [] for x in query.split()}  # строим инвертированный индекс
    for doc_id, doc in enumerate(index):
        for word in query.split():
            if (word in doc.title_words.split()) or (word in doc.text_words.split()):
                inv_index[word].append(doc_id)

    # отбор документов по инв. индексу (пересекаем множества слов из запросов)
    lst = [set(lst) for lst in inv_index.values()]
    if len(lst) > 0:
        lst = lst[0].intersection(*lst)
        candidates = [index[i] for i in lst]

        return candidates[:50]
    else:
        return []
