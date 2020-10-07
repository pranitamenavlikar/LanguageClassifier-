# LanguageClassifier
A machine learning model using 'Decision Tree' and 'AdaBoost' algorithms for classifying short sentences as English or Dutch

## How to use?
A] Train_model.py -  
This model takes the labelled training data and the algorithm to be checked, 'dt' for Decision tree, 'ada' for Adaboost.  
It is the model's training file, output is the pickled files for the respective models.  
B] Predict.py - 
This is the prediction file.It takes  
1. test data file  
2. If the first parameter is   
1 -> decision tree  
2 -> ada boost  
3. The second parameter here,
If checked for 'Decision tree', root node is received
If checked for 'Adaboost', pair of hypothesis attributes and weight is received  

C]/data folder containes different sizes' training files and a temporary test file


## Curated Features  
Text classification depends a lot on the choice of Parameters used for
classification.
For my code I have selected the following features :
#### 1. 'Average_word_length' -  
English tends to have small words for a sentence . The average word
length is 5, here I discovered getting accurate distinguishing factor as 4.8
for average of sum of length of all words.
#### 2. 'Words_with_ij' - 
Many dutch words have ‘ij’ in them while no english word has ‘ij’
consecutively. Hence, this proved to be an important parameter while
selecting the best attribute. While studying the differences between the
two languages , I came across this difference.
#### 3. 'presenceOf_het_de' -  
Dutch uses ‘het’ and ‘de’ instead of the and from the high usage of ‘the’ in
english , we can certainly say the ‘het’ and ‘de’ are also frequently used.
#### 4. 'dutch_vowelCombinations' -  
Dutch language is known for this vowel combination of english languages.
When you search a dutch document for vowel pairs like ‘ao’ ‘eu’ ‘ae’, etc.
you’ll find plenty of words having these combinations but not many english
words have it.
#### 5. 'English_stopwords' -  
This is a list of common english words i searched for and found lists given
by people for the ease of language classification.
#### 6. 'Long_word' -  
Many dutch words have length greater than 12 on an average but same is
not the case with english. There are hardly any words greater than 12 so if
there exists one or more, the language has a high probability of being
dutch.
#### 7. 'Words_with_z' -   
Many dutch words have the alphabet ‘z’ in it but not many english words
have z in them .
#### 8. 'eng_Thirdperson' -  
Dutch has use of only two genders to specify - I, me, myself and third
person. This attribute includes all the pronouns especially third person
pronouns which are common english words.
#### 9. 'Dutch_stopwords' -  
This is a list of common dutch words I searched for and found lists given
by people for the ease of language classification.
#### 10. 'Same_vowel' -  
Rarely words having consecutive use of ‘i’, ‘u’, ‘e’, ‘a’ vowels is found in english.
In dutch its very common.

## Model Results  
Both algorithms are implemented using Russel and Norvig’s algorithms and numpy dataframes.  

#### Decision Tree :  
Selected attribue: dutch_stopwords ...  
Selected attribue: eng_Thirdperson ...  
Selected attribue: english_stopwords ...  
Selected attribue: average_word_length ...  
Selected attribue: dutch_vowelCombinations ....  
Selected attribue: words_with_ij ....  
……..  
Result dutch_stopwords  
<class '__main__.Node'>   

The above output had a large depth of the tree, I have taken only a few to show the kind of variation in choice of attributes. Dutch common words was a common attribute in both the models , hence quite a powerful differentiator.  
The info_gain and entropy are the most important features here. Element with highest info gain i.e. least entropy is always selected. 
This goes on till all the attributes are evaluated, else the label for class is assigned . That is the final class label.  

#### Adaboost :  
Hypothesis -
['dutch_stopwords', 'eng_Thirdperson', 'english_stopwords',
'dutch_vowelCombinations', 'words_with_z', 'long_word', 'words_with_ij',
'long_word', 'dutch_stopwords']  

Hypothesis weights  
[4.318890638310396, 3.005780703917874, 2.0581296095434234,
2.7943072659444805, 3.793948074282655, 5.188223084067835,
9.315208425756609, 12.936772697230193, 16.292217866459556]  
This is a boosting algorithm so the feature with minimum weight is selected and
high errored attributes are boosted by giving them higher weights.
The weights are combined and the best classifier is chosen for the selection.
Dutch words, english stopwords, words with ‘ij’, words with ‘z’ were proven to be
the features for best classifiers.  

K = 9 performed the best out of other K options. There was quite a variation of
choice of features , hence no attribute was too strong in my selection. Hence the
model is quite explored for all features as all features were possibly good
dominating features.  

Accuracy achieved for both the algorithms,  
DT - 86 - 93 %  
ADA - 80- 90 %  

