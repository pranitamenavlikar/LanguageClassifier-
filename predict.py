import pickle
import re
import sys

from sklearn.metrics import confusion_matrix

import pandas as pd

from modelDecisionTree import Node


def featuresFill(line, dict):
    # words = line.split()
    # words = line.replace('|', ' ').replace('-', ' ').replace(';', ' ').replace(':', ' ').replace('(', ' ').split()
    # words = re.split(r'\W+', line)
    words = line.replace('|', ' ').split(' ')
    # print(words[0])

    dict = {'average_word_length': '0', 'words_with_ij': '0', 'presenceOf_het_de': '0', 'dutch_vowelCombinations': '0',
            'english_stopwords': '0', 'long_word': '0', 'words_with_z': '0', 'eng_Thirdperson': '0',
            'dutch_stopwords': '0'}

    words = words[1:16]

    sum = 0
    for word in words:
        # print(word)
        sum += len(word)

    avg = sum / 15
    if avg > 4.8:
        # dict = {'average_word_length': 'False'}
        dict['average_word_length'] = 'False'
    else:
        # dict = {'average_word_length': 'True'}
        dict['average_word_length'] = 'True'
    # average word length _____________________________________________________________________________

    flagij = 0
    for word in words:
        if 'ij' in word:
            flagij = 1

    if flagij == 1:
        # print("Here")
        dict['words_with_ij'] = 'False'
    else:
        dict['words_with_ij'] = 'True'

    # presence of ij ____________________________________________________________________________________


    long = 0
    for word in words:
        # print(word)
        long = len(word)
        if long > 13:
            dict['long_word'] = False
        else:
            dict['long_word'] = False

    #  count long word ____________________________________________________________________________________

    joint_vowels = ['ae', 'ai', 'au', 'ei', 'eu', 'ie', 'ij', 'oe', 'oi', 'ou', 'ui']
    flagvo = 0
    for word in words:
        for a in joint_vowels:
            if a in word:
                flagvo = 1
                break

    if flagvo == 1:
        # print("Here ae ai au")
        dict['dutch_vowelCombinations'] = 'False'
    else:
        dict['dutch_vowelCombinations'] = 'True'
    # joint vowels ______________________________________________________________________________________


    words_for_the = ['het', 'de']
    flaghet = 0
    for word in words:
        for a in words_for_the:
            if a == word:
                flaghet = 1
                break

    if flaghet == 1:
        # print("Here het de")
        dict['presenceOf_het_de'] = 'False'
    else:
        dict['presenceOf_het_de'] = 'True'

    # het_de presence _____________________________________________________________________________________

    # gender_pronouns = ['him', 'her', 'he', 'she', 'herself', 'himself', "her's", "he's", 'the', 'a']
    # flaggp = 0
    # for word in words:
    #     for a in gender_pronouns:
    #         if a == word:
    #             flaggp = 1
    #             break
    #
    # if flagvo == 1:
    #     # print("Here ae ai au")
    #     dict['gender_Pronouns'] = 'True'
    # else:
    #     dict['gender_Pronouns'] = 'False'
    # pronouns_gender ______________________________________________________________________________________

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
              "you'd", 'your', 'yours', 'yourself', 'yourselves', 'it', "it's", 'its', 'itself', 'they', 'them',
              'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
              'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
              'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
              'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to']
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
    swords =  ['from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
              'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
              'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
              't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
              've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
              "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
              "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
              "weren't", 'won', "won't", 'wouldn', "wouldn't", 'is', 'are', 'was','him', 'her', 'he', 'she',
               'herself', 'himself', "her's", "he's", 'the', 'a' ]
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


def traversetree(root, data):
    # print("Helooooo",type(data[root.value]))
    if root.true is None or root.false is None:
        #print("none?",root.value)
        return root.value
    elif data[root.value] == True:
        return traversetree(root.true, data)
    elif data[root.value] == False:
        return traversetree(root.false, data)


def testDecisionTree(root, path):
    # colnames = ['average_word_length', 'words_with_ij', 'presenceOf_het_de', 'dutch_vowelCombinations',
    #             'english_stopwords', 'words_with_z', 'dutch_stopwords','same_vowels', 'Language']
    colnames = ['average_word_length', 'words_with_ij', 'presenceOf_het_de', 'dutch_vowelCombinations',
                'english_stopwords', 'long_word', 'words_with_z', 'eng_Thirdperson', 'dutch_stopwords', 'Language']
    file = open(path, encoding="utf8")
    attribute_cols = {}
    i = 0
    data = pd.DataFrame(columns=colnames)
    for line in file:
        vals = line.split('\n')
        attribute_cols = featuresFill(line, attribute_cols)
        # data.loc['y'] = pandas.Series({'a': 1, 'b': 5, 'c': 2, 'd': 3})
        data = data.append(attribute_cols, ignore_index=True)

    #print(data)
    predicted = []
    for x in range(data.shape[0]):
        label = traversetree(root, data.loc[x])
        #print(label)
        if label == None:
            label = 'en'
        print(label)
        predicted.append(label)
        # print("\n")

    # train = ['nl','en','en','nl','en','en','nl','en','nl','en','en','en','nl','en','nl','nl','nl','nl','en','nl']
    # print("Accuracy",confusion_matrix(train, predicted))
    test = ['nl', 'en', 'en', 'nl', 'en', 'en', 'nl', 'en', 'en', 'en', 'en', 'en', 'nl', 'en', 'nl', 'nl', 'nl', 'nl',
            'nl', 'en', 'nl']
    # print('Result:')
    # print('Predicted deci:', predicted)
    # print('Expected deci:', test)


def checkensemble(hypo, hypoweight, data):
    counten = 0
    countnl = 0
    for i in range(len(hypo)):
        attribute = hypo[i]
        if data[attribute] == 'True':
            counten = counten + hypoweight[i]
        elif data[attribute] == 'False':
            countnl = countnl + hypoweight[i]
    if counten > countnl:
        return 'en'
    else:
        return 'nl'


def testAdaboost(hypo, hypoweights, path):
    # colnames = ['average_word_length', 'words_with_ij', 'presenceOf_het_de', 'dutch_vowelCombinations',
    #             'english_stopwords', 'words_with_z', 'dutch_stopwords','same_vowels','Language']

    colnames = ['average_word_length', 'words_with_ij', 'presenceOf_het_de', 'dutch_vowelCombinations',
                'english_stopwords', 'long_word', 'words_with_z', 'eng_Thirdperson', 'dutch_stopwords', 'Language']

    file = open(path, encoding="utf8")
    attribute_cols = {}
    i = 0

    data = pd.DataFrame(columns=colnames)

    for line in file:
        vals = line.split('\n')
        attribute_cols = featuresFill(line, attribute_cols)
        # data.loc['y'] = pandas.Series({'a': 1, 'b': 5, 'c': 2, 'd': 3})
        data = data.append(attribute_cols, ignore_index=True)

    #print(data)
    predicted = []
    test = ['nl', 'en', 'en', 'nl', 'en', 'en', 'nl', 'en', 'en', 'en', 'en', 'en', 'nl', 'en', 'nl', 'nl', 'nl', 'nl',
            'nl', 'en', 'nl']


    for x in range(data.shape[0]):
        label = checkensemble(hypo, hypoweight, data.loc[x])
        print(label)
        predicted.append(label)

    # print('Result:')
    # print('Predicted:', predicted)
    # print('Expected:', test)


if __name__ == '__main__':
    #hypothesis = "C:\\Users\\prani\\PycharmProjects\\AI4_2\\ada.pickle"
    hypothesis = sys.argv[1]
    test_file_path = sys.argv[2]

    pickle_in = open(hypothesis, "rb")
    pkl_list = pickle.load(pickle_in)
    if pkl_list[0] == 1:
        rootNode = pkl_list[1]
        testDecisionTree(rootNode, test_file_path)
    elif pkl_list[0] == 2:
        hypoattri = pkl_list[1]
        hypoweight = pkl_list[2]
        testAdaboost(hypoattri, hypoweight, test_file_path)
