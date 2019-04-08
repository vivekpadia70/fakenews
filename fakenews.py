from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import pickle
import nltk

option = webdriver.ChromeOptions()
option.add_argument("--incognito")

browser = webdriver.Chrome(executable_path='./chromedriver', options=option)

browser.get('https://www.google.com/webhp?tbm=nws')

queryString = "India attacked Pakistan"

browser.find_element_by_xpath("//input[@title='Search']").send_keys(queryString, Keys.ENTER)

headings = browser.find_elements_by_xpath("//div[@class='g']//h3")

sentences = []

for heading in headings:
    sentences.append(heading.text)

print(sentences)


def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    for synset in synsets1:
        scores = [wn.path_similarity(synset, ss) for ss in synsets2]
        if [x for x in scores if x is not None] == []:
            return 0

        best_score = max([x for x in scores if x is not None])
        if best_score is not None:
            score += best_score
            count += 1

    if count == 0:
        score = 0
        print('oops')
    else:
        score /= count
    return score * 100


focus_sentence = queryString

def symmetric_sentence_similarity(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2

def dialogue_act_features(post):
    features={}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

def check_sentence_type(post):
    f = open('nltk_sentence_type_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier.classify(dialogue_act_features(post))
    
for sentence in sentences:
    print(check_sentence_type(focus_sentence))
    print("similarity = %f" % symmetric_sentence_similarity(focus_sentence, sentence) + '%')

