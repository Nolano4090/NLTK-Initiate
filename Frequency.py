import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

stop_words = stopwords.words('english')
for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's', '``', '\'', '\"', '[', ']', 'asiaing.com', 'ma', '-']:
    stop_words.append(w)


def read_file(path):  # 读取文件
    file = open(path)
    text = file.read()
    file.close()
    return text


def generate_corpus(text):  # 英文文本分词
    sentences = nltk.sent_tokenize(text)
    corpus = nltk.word_tokenize(str(sentences))
    return corpus


def word_tagging(corpus):  # 词性标注
    tagged_words = nltk.pos_tag(corpus)
    return tagged_words


def generate_filtered_words(corpus):  # 去除停用词
    filtered_words = [word for word in corpus if word.lower() not in stop_words]
    return filtered_words


def remove_suffix_prefix(filtered_words):  # 去除前后缀
    for i in range(len(filtered_words)):
        filtered_words[i] = filtered_words[i].strip()
        filtered_words[i] = filtered_words[i].strip('\'')
        filtered_words[i] = filtered_words[i].strip('\n')
        filtered_words[i] = filtered_words[i].strip('\\n')
        filtered_words[i] = filtered_words[i].strip('\\')
        filtered_words[i] = filtered_words[i].strip('/')
    return filtered_words


def get_wordnet_pos(tag):  # 泛化单词的词性为四类
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatized_words(tagged_words):  # 词形还原
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word_tag in tagged_words:
        wordnet_pos = get_wordnet_pos(word_tag[1]) or wordnet.NOUN
        lemmas.append(lemmatizer.lemmatize(word_tag[0], pos=wordnet_pos))  # 词形还原
    return lemmas


def WordCloudGeneration(src, freq_dist):  # 词云生成

    # 生成词云
    Word_C = WordCloud(
        background_color="white",  # 设置背景为白色，默认为黑色
        width=1500,  # 设置图片的宽度
        height=960,  # 设置图片的高度
        margin=10  # 设置图片的边缘
    ).generate(str(freq_dist.most_common(30)))

    # 绘制图片
    plt.imshow(Word_C)

    # 消除坐标轴
    plt.axis("off")

    # 展示图片
    plt.show()

    # 保存图片
    Word_C.to_file(src + '_wordcloud.jpg')