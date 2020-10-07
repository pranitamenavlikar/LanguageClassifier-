# Decision tree algorithm implemented by taking reference from Russel and Norvig (page 702)
import pickle
import sys

import math
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re


# DECISION TREE _____________________________________________________________________________________________________
class Node:
    __slots__ = "value", "true", "false", "parent"

    def __init__(self, value):
        self.value = value
        self.true = None
        self.false = None
        self.parent = None

    def obtainParent(self):
        return self.parent

    def getLink(self, Node):
        if Node.true == self:
            return "true"
        elif Node.false == self:
            return "false"


class buildDT:
    __slots__ = "root"

    def __init__(self, root, true, false):
        self.root = root
        root.true = true
        true.parent = root
        root.false = false
        false.parent = root


# FEATURE_SELECTION ___________________________________________________________________________________________________
def features(line, dict):
    # words = line.split()
    # words = line.replace('|', ' ').replace('-', ' ').replace(';', ' ').replace(':', ' ').replace('(', ' ').split()
    # words = re.split(r'\W+', line)
    words = line.replace('|', ' ').split(' ')
    # print(words[0])

    dict = {'average_word_length': '0', 'words_with_ij': '0', 'presenceOf_het_de': '0', 'dutch_vowelCombinations': '0',
            'english_stopwords': '0', 'long_word': '0', 'words_with_z': '0', 'eng_Thirdperson': '0',
            'dutch_stopwords': '0',
            'Language': words[0]}

    words = words[1:16]

    sum = 0
    for word in words:
        # print(word)
        sum += len(word)

    avg = sum / 15
    if avg > 4.8:
        # dict = {'average_word_length': 'False'}
        dict['average_word_length'] = False
    else:
        # dict = {'average_word_length': 'True'}
        dict['average_word_length'] = True
    # average word length _____________________________________________________________________________

    flagij = 0
    for word in words:
        if 'ij' in word:
            flagij = 1

    if flagij == 1:
        # print("Here")
        dict['words_with_ij'] = False
    else:
        dict['words_with_ij'] = True

    # presence of ij ____________________________________________________________________________________

    joint_vowels = ['ae', 'ai', 'au', 'ei', 'eu', 'ie', 'ij', 'oe', 'oi', 'ou', 'ui']
    flagvo = 0
    for word in words:
        for a in joint_vowels:
            if a in word:
                flagvo = 1
                break

    if flagvo == 1:
        # print("Here ae ai au")
        dict['dutch_vowelCombinations'] = False
    else:
        dict['dutch_vowelCombinations'] = True
    # joint vowels ______________________________________________________________________________________

    long = 0
    for word in words:
        # print(word)
        long = len(word)
        if long > 13:
            dict['long_word'] = False
        else:
            dict['long_word'] = False

    #  count long word ____________________________________________________________________________________

    words_for_the = ['het', 'de']
    flaghet = 0
    for word in words:
        for a in words_for_the:
            if a == word:
                flaghet = 1
                break

    if flaghet == 1:
        # print("Here het de")
        dict['presenceOf_het_de'] = False
    else:
        dict['presenceOf_het_de'] = True

    # het_de presence _____________________________________________________________________________________

    countz = 0
    flagz = 0
    words_for_the = ['het', 'de']
    for word in words:
        if 'z' in word:
            countz = countz + 1
        for a in words_for_the:
            if a == word:
                flagz = 1
                break

    if countz >= 2:
        # print("Here")
        dict['words_with_z'] = False
    elif countz < 2:
        if flagz == 1:
            dict['words_with_z'] = False
        else:
            dict['words_with_z'] = True

    # presence of z ____________________________________________________________________________________

    dwords = ['aan', 'af', 'al', 'alles', 'als', 'altijd', 'andere', 'ben', 'bij', 'daar', 'dan', 'dat', 'der',
              'deze', 'die', 'dit', 'doch', 'doen', 'door', 'dus', 'een', 'eens', 'en', 'er', 'ge', 'geen', 'geweest',
              'haar', 'heb', 'hebben', 'heeft', 'hem', 'hier', 'hij', 'hoe', 'hun', 'iemand', 'iets', 'ik',
              'ja', 'je', 'kan', 'kon', 'kunnen', 'maar', 'me', 'meer', 'men', 'met', 'mij', 'mijn', 'moet', 'na',
              'naar', 'niet', 'niets', 'nog', 'nu', 'om', 'omdat', 'ons', 'ook', 'op', 'reeds', 'te', 'tegen', 'toch',
              'toen',
              'tot', 'u', 'uit', 'uw', 'van', 'veel', 'voor', 'waren', 'wat', 'wel', 'werd', 'wezen', 'wie', 'wij',
              'wil', 'worden', 'zal', 'ze', 'zei', 'zelf', 'zich', 'zij', 'zijn', 'zo', 'zonder', 'zou', 'ij']
    # print(swords)

    flagwo = 0
    for word in words:
        if word in dwords:
            flagwo = 1
            break

    if flagwo == 1:
        # print("there in stopword")
        dict['dutch_stopwords'] = False
    else:
        dict['dutch_stopwords'] = True

    # dutch stopword presence _____________________________________________________________________________________

    eng_thirdperson = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                       "you'd", 'your', 'yours', 'yourself', 'yourselves', 'it', "it's", 'its', 'itself', 'they',
                       'them',
                       'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                       'these', 'those', 'am',
                       'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
                       'an',
                       'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                       'with',
                       'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                       'to']
    # print(swords)

    flagtp = 0
    for word in words:
        if word in eng_thirdperson:
            flagtp = 1
            break

    if flagtp == 1:
        # print("there in stopword")
        dict['eng_Thirdperson'] = True
    else:
        dict['eng_Thirdperson'] = False

    # dutch stopword presence _____________________________________________________________________________________

    # swords = stopwords.words('english')
    swords = ['from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
              'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
              'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
              't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
              've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
              "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
              "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
              "weren't", 'won', "won't", 'wouldn', "wouldn't", 'is', 'are', 'was', 'him', 'her', 'he', 'she',
              'herself', 'himself', "her's", "he's", 'the', 'a']
    # print(swords)

    flagwo = 0
    for words in words:
        if words in swords:
            flagwo = 1
            break

    if flagwo == 1:
        # print("there in stopword")
        dict['english_stopwords'] = True
    else:
        dict['english_stopwords'] = False

    # stopword presence _____________________________________________________________________________________

    # print(dict)
    # CLASS LABEL FILLED _______________________________________________________________________

    # print(dict)
    return dict


# FEATURE_SELECTION ___________________________________________________________________________________________________


def pluralityValues(data):
    # arr = data.to_numpy()
    # print("Arr",arr[0][8])
    labels = data['Language'].to_numpy()
    # for i in range(len(arr)):
    labels, counts = np.unique(labels, return_counts=True)
    # print("Labels", labels)
    # print("Count", counts)
    max = counts[0]
    index = 0
    for i in range(len(counts)):
        if counts[i] > max:
            max = counts[i]
            index = i
    getClass = Node(labels[i])
    return getClass


def read_file(path, to_pickle):
    # data = pd.read_csv(path, header=None,
    #                    names=['attri1', 'attri2', 'attri3', 'attri4', 'attri5', 'attri6', 'attri7', 'attri8',
    #                           'classLabel'])
    # # print(data)
    #
    # # print(data.classLabel)
    # attributes = ['attri1', 'attri2', 'attri3', 'attri4', 'attri5', 'attri6', 'attri7', 'attri8']
    # independentattri = data[attributes]
    # dependentattri = data['classLabel']
    # # print(independentattri)
    # # print(dependentattri)

    colnames = ['average_word_length', 'words_with_ij', 'presenceOf_het_de', 'dutch_vowelCombinations',
                'english_stopwords', 'long_word', 'words_with_z', 'eng_Thirdperson', 'dutch_stopwords', 'Language']

    file = open(path, encoding="utf8")
    attribute_cols = {}
    i = 0

    data = pd.DataFrame(columns=colnames)

    for line in file:
        vals = line.split('\n')
        # print(vals[0])
        # average_word_length(vals[0],dict)
        # words_with_ij(vals[0],dict)
        # presenceOf_het_de(vals[0],dict)
        # dutch_vowelCombinations(line,dict)
        # english_stopwords(line,dict)
        # language_class(line,dict)
        attribute_cols = features(line, attribute_cols)
        # data.loc['y'] = pandas.Series({'a': 1, 'b': 5, 'c': 2, 'd': 3})
        data = data.append(attribute_cols, ignore_index=True)

    columns = data.columns
    # print("Columns", columns)
    height = 0  # start level
    val = decisiontree(data, columns, data, height)
    print("Result", val.value)
    print(type(val))

    # pickle_root = open("root.pickle", "wb")
    # pickle.dump(val, pickle_root)
    # pickle_root.close()

    result = [1, val]
    pickle_deci = open(to_pickle, "wb")
    pickle.dump(result, pickle_deci)
    pickle_deci.close()


def decisiontree(data, attributes, parentdata, height):
    if len(data) == 0:
        return pluralityValues(parentdata)
    elif len(data) == sameClassification(data):
        return pluralityValues(data)
    elif len(attributes) == 0 or len(attributes) == 1:
        return pluralityValues(data)
    else:
        selectedAttri = infogain(data)
        print("Selected attribue: ", selectedAttri)
        root = Node(selectedAttri)
        featureTrue = data[data[selectedAttri] == True]
        featureTrue = featureTrue.drop(columns=selectedAttri)
        # print(featureTrue)
        attributesr = featureTrue.columns
        featureFalse = data[data[selectedAttri] == False]
        featureFalse = featureFalse.drop(columns=selectedAttri)
        # print(featureFalse)
        attributesl = featureFalse.columns
        rightTrue = decisiontree(featureTrue, attributesr, data, height + 1)
        rightTree = rightTrue
        leftFalse = decisiontree(featureFalse, attributesl, data, height + 1)
        leftTree = leftFalse
        # print("parent " + selectedAttri + " = True " + rightTrue + " height : " + str(height))
        # print("parent " + selectedAttri + " = False " + leftFalse + " height : " + str(height))
        buildDT(root, rightTree, leftTree)
        #     return 0
        # return selectedAttri
        return root


def sameClassification(data):
    labels = data['Language'].to_numpy()
    # for i in range(len(arr)):
    labels, counts = np.unique(labels, return_counts=True)
    # print("Labels", labels)
    # print("Count", counts)
    flag = False
    index = 0
    for i in range(len(counts)):
        if counts[i] == len(data):
            flag = True
            index = i
            break

    if flag:
        # print("Count:", counts[index])
        return counts[index]
    else:
        # print("Count:", counts[0])
        return counts[0]


def infogain(data):
    counta = len(data[data['Language'] == 'en'])
    countb = len(data[data['Language'] == 'nl'])
    mainEntropy = entropy(counta, countb)

    countTA = 0
    countTB = 0
    countFA = 0
    countFB = 0
    num = len(data.columns)
    maxims = {}

    for attri in data.columns:
        if attri != 'Language':
            countTA = len(data[(data['Language'] == 'en') & (data[attri] == True)])
            countTB = len(data[(data['Language'] == 'nl') & (data[attri] == True)])
            countFA = len(data[(data['Language'] == 'en') & (data[attri] == False)])
            countFB = len(data[(data['Language'] == 'nl') & (data[attri] == False)])

            cntTrue = countTA + countTB
            cntFalse = countFA + countFB
            total = cntTrue + cntFalse
            # print("Count:", cntTrue, cntFalse)
            thisAttributeEntropy = (cntTrue / (counta + countb)) * entropy(countTA, countTB) + \
                                   (cntFalse / (counta + countb)) * entropy(countFA, countFB)

            diff = mainEntropy - thisAttributeEntropy
            maxims[attri] = diff

    # print(maxims)
    val = findMax(maxims)
    print("Val", val)
    # index = val + 1
    # col_name = "attri"+str(index)
    return val


def entropy(counta, countb):
    if counta == 0:
        ent = 0
    elif countb == 0:
        ent = 0
    else:
        x = (counta / (counta + countb) * math.log(counta / (counta + countb), 2))
        y = (countb / (counta + countb) * math.log(countb / (counta + countb), 2))
        ent = -(x + y)
    return ent


def findMax(dict):
    # list = dict.keys()
    if len(dict) != 0:
        sendKey = list(dict.keys())[0]
        # print("First Key:",max)
        max = dict.get(sendKey)
        # sendKey = list[0]
        for i, (k, v) in enumerate(dict.items()):
            print(i, k, v)
            if v > max:
                max = v
                sendKey = k

        # index = 0
        # for b in range(len(list)):
        #     if list[b] > max:
        #         max = list[b]
        #         index = b
        return sendKey
    else:
        return ''


# DECISION TREE END_____________________________________________________________________________________________________

# ADABOOST ___________________________________________________________________________________________________________

def adaboost(data, attributes):
    print(data.shape)
    no_of_samples = data.shape[0]
    ini_weight = 1 / no_of_samples
    print(data)
    print("Initial Weight:", ini_weight)
    weight_vals = []
    for x in range(no_of_samples):
        weight_vals.append(ini_weight)

    data['Weight'] = weight_vals
    # print(data)
    h = []
    z = []
    # value = data['words_with_ij'].iloc
    # print(value)
    attributes = data.columns
    print("Columns", attributes)

    for i in range(len(attributes) - 2):
        rootnode = decisiontreeAda(data, attributes, data, 0)
        print(rootnode)
        h.append(rootnode)
        error = 0
        correct = 0
        wrong = 0

        for i in range(no_of_samples):
            # print(data[rootnode].loc(i))
            if (data.loc[i, rootnode] and data.loc[i, 'Language'] == 'nl') or (
                    data.loc[i, rootnode] == False and data.loc[i, 'Language'] == 'en'):
                wrong += 1
                error = error + (data.loc[i, 'Weight'])
                # print("Error", error)

        for j in range(no_of_samples):
            # print(data.loc[j, 'Weight'], "beforeekta", j)
            if (data.loc[j, rootnode] and data.loc[j, 'Language'] == 'en') or (
                    data.loc[j, rootnode] == False and data.loc[j, 'Language'] == 'nl'):
                correct += 1
                data.loc[j, 'Weight'] = ((data.loc[j, 'Weight'] * error) / (1 - error))
                # print(data.loc[j, 'Weight'], "afterekta", j)

        weights_sum = data.loc[:, 'Weight'].sum()
        for k in range(no_of_samples):
            data.loc[k, 'Weight'] = data.loc[k, 'Weight'] / weights_sum

        z.append(math.log((1 - error) / error))
    return h, z


def decisiontreeAda(data, attributes, parentdata, height):
    if len(data) == 0:
        return pluralityValues(parentdata)
    elif len(data) == sameClassification(data):
        return pluralityValues(data)
    elif len(attributes) == 0 or len(attributes) == 1 or len(attributes) == 2:
        return pluralityValues(data)
    else:
        selectedAttri = infogainAda(data)
        print("Selected attribue: ", selectedAttri)
        # featureTrue = data[data[selectedAttri] == True]
        # featureTrue = featureTrue.drop(columns=selectedAttri)
        # attributesr = featureTrue.columns
        # featureFalse = data[data[selectedAttri] == False]
        # featureFalse = featureFalse.drop(columns=selectedAttri)
        # attributesl = featureFalse.columns
        # rightTrue = decisiontreeAda(featureTrue, attributesr, data, height + 1)
        # leftFalse = decisiontreeAda(featureFalse, attributesl, data, height + 1)

        # print("parent " + selectedAttri + " = True " + rightTrue + " height : " + str(height))
        # print("parent " + selectedAttri + " = False " + leftFalse + " height : " + str(height))
        return selectedAttri

def infogainAda(data):
    # counta = data[data['Language'] == 'en'].loc[:, 'Weight'].sum()
    # countb = data[data['Language'] == 'nl'].loc[:, 'Weight'].sum()
    # mainEntropy = entropy(counta, countb)

    countTA = 0
    countTB = 0
    countFA = 0
    countFB = 0
    num = len(data.columns)
    maxims = {}

    for attri in data.columns:
        if attri != 'Language' and attri != 'Weight':
            countTA = data[(data['Language'] == 'en') & (data[attri] == True)].loc[:, 'Weight'].sum()
            countTB = data[(data['Language'] == 'nl') & (data[attri] == True)].loc[:, 'Weight'].sum()
            countFA = data[(data['Language'] == 'en') & (data[attri] == False)].loc[:, 'Weight'].sum()
            countFB = data[(data['Language'] == 'nl') & (data[attri] == False)].loc[:, 'Weight'].sum()

            misClassified = countTB + countFA
            total = data.shape[0]

            # weight = misClassified / total
            # cntTrue = countTA + countTB
            # cntFalse = countFA + countFB
            # total = cntTrue + cntFalse
            # # print("Count:", cntTrue, cntFalse)
            # thisAttributeEntropy = (cntTrue / (counta + countb)) * entropy(countTA, countTB) + \
            #                        (cntFalse / (counta + countb)) * entropy(countFA, countFB)
            #
            # diff = mainEntropy - thisAttributeEntropy
            maxims[attri] = misClassified
    print(maxims)
    val = findMin(maxims)
    # print("Val", val)
    # index = val + 1
    # col_name = "attri"+str(index)
    return val



def read_fileAda(path, to_pickle):
    # colnames = ['average_word_length', 'words_with_ij', 'presenceOf_het_de', 'dutch_vowelCombinations',
    #             'english_stopwords', 'words_with_z', 'dutch_stopwords', 'Language']
    colnames = ['average_word_length', 'words_with_ij', 'presenceOf_het_de', 'dutch_vowelCombinations',
                'english_stopwords', 'long_word', 'words_with_z', 'eng_Thirdperson', 'dutch_stopwords', 'Language']

    file = open(path, encoding="utf8")
    attribute_cols = {}
    i = 0

    data = pd.DataFrame(columns=colnames)

    for line in file:
        vals = line.split('\n')
        attribute_cols = features(line, attribute_cols)
        # data.loc['y'] = pandas.Series({'a': 1, 'b': 5, 'c': 2, 'd': 3})
        data = data.append(attribute_cols, ignore_index=True)

    columns = data.columns
    print("Columns", columns)
    hypotheses, hypo_weights = adaboost(data, columns)
    print(hypotheses)
    print(hypo_weights)
    result = [2, hypotheses, hypo_weights]
    pickle_ada = open(to_pickle, "wb")
    pickle.dump(result, pickle_ada)
    pickle_ada.close()


def findMin(dict):
    # list = dict.keys()
    if len(dict) != 0:
        sendKey = list(dict.keys())[0]
        # print("First Key:",max)
        min = dict.get(sendKey)
        # sendKey = list[0]
        for i, (k, v) in enumerate(dict.items()):
            # print(i, k, v)
            if v < min:
                min = v
                sendKey = k

        # index = 0
        # for b in range(len(list)):
        #     if list[b] > max:
        #         max = list[b]
        #         index = b
        return sendKey
    else:
        return ''


# ADABOOST END ______________________________________________________________________________________________________

if __name__ == '__main__':
    # path = "C:\\Users\\prani\\Downloads\\train.dat"
    # path = "C:\\Users\\prani\\Desktop\\Spring2020\\FoundationsOfArtificialIntelligence\\dttree.csv"
    # path = "C:\\Users\\prani\\Downloads\\train.dat"

    path = sys.argv[1]
    to_pickle = sys.argv[2]
    algorithm = sys.argv[3]
    # path = "C:\\Users\\prani\\Downloads\\Code.dat"

    if algorithm == 'dt':
        print("DECISION TREE CODE")
        read_file(path, to_pickle)
    elif algorithm == 'ada':
        print("ADA BOOST CODE")
        read_fileAda(path, to_pickle)
