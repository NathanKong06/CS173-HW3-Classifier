import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import math
import string
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#Stop words 
nltk.download('punkt')
nltk.download('stopwords')

#Instantiate all dictionaries
fear_dictionary = {}
anger_dictionary = {}
surprise_dictionary = {}
disgust_dictionary = {}
sadness_dictionary = {}
joy_dictionary = {}

#Read CSV File
df = pd.read_csv(r"CS173-published-sheet - Sheet1.csv")
#Remove all newline characters
df = df.replace({r'\n':' '},regex = True) 

def reset_w_c():
    w = []
    c = []
    #Create W and C lists and initialize to 1
    for vocab_size in range(len(unique_words)):
        empty = []
        for d in range(5):
            empty.append(1)
        w.append(empty)
        c.append(empty)
    return w,c

def fix_sen_tokenize(text): #Sentence tokenization
    if isinstance(text,float): #If blank cell (NaN)
        text = ''
    return sent_tokenize(text)

def fix_tokenize(text): #Word tokenizes sentence
    if isinstance(text,float): #If blank cell (NaN)
        text = ''
    return word_tokenize(text)
  
def sigmoid(feature,weights): 
    try:
        return (1.0/ (1.0 + math.exp(-(weights[0]*feature[0] + weights[1]*feature[1] + weights[2]*feature[2]+weights[3]*feature[3]+weights[4]*feature[4]))))
    except: #Handle overflow issues
        return 1- (1.0/ (1.0 + math.exp(weights[0]*feature[0] + weights[1]*feature[1] + weights[2]*feature[2]+weights[3]*feature[3]+weights[4]*feature[4])))
    
def pos_loss(pos_feature,weights):
    temp = sigmoid(pos_feature,weights)-1
    return [i*temp for i in weights]

def neg_loss(neg_feature,weights):
    temp = sigmoid(neg_feature,weights)
    return [i*temp for i in weights]

def w_loss(pos_feature,neg_features,weights):
    pos_temp = sigmoid(pos_feature,weights)-1
    pos = [i*pos_temp for i in pos_feature]
    neg_sum = [0,0,0,0,0]
    neg = []
    for neg_feature in neg_features:
        neg_temp = sigmoid(neg_feature,weights)
        neg.append([i*neg_temp for i in neg_feature])
        neg_sum = [x + y for x,y in zip(neg_sum,neg[-1])]
    return [x+y for x,y in zip(neg_sum,pos)]

def stochastic_gradient_descent(w,c,rate):
    total_loss = 0.0
#     for j in range(100):
    for i,pos_examples in enumerate(positive_examples_num):
        temp_list = [sublist for sublist in [negative_examples_num[i][(idx)*5:(idx+1)*5] for idx in range(len(negative_examples_num[i]))]]
        temp_list = [sublist for sublist in temp_list if sublist]
        for j,example in enumerate(pos_examples):
            target = pos_examples[j][0]
            pos = pos_examples[j][1]
            #pos_examples holds [target word,positive word]
            #temp_list holds 5 instances of [target word, negative word]
            
#             neg = [temp_list[j][0][1],temp_list[j][1][1],temp_list[j][2][1],temp_list[j][3][1],temp_list[j][4][1]]
#             neg_1 = temp_list[j][0][1]
#             neg_2 = temp_list[j][1][1]
#             neg_3 = temp_list[j][2][1]
#             neg_4 = temp_list[j][3][1]
#             neg_5 = temp_list[j][4][1]
            
            #Get the actual list from W and C Matrices
            w_target = w[target]
            c_pos = c[pos]
            c_neg = [c[temp_list[j][0][1]],c[temp_list[j][1][1]],c[temp_list[j][2][1]],c[temp_list[j][3][1]],c[temp_list[j][4][1]]]
            c_neg_1 = c[temp_list[j][0][1]]
            c_neg_2 = c[temp_list[j][1][1]]
            c_neg_3 = c[temp_list[j][2][1]]
            c_neg_4 = c[temp_list[j][3][1]]
            c_neg_5 = c[temp_list[j][4][1]]
            
            #Perform loss calculations
            w_target = [x-y for x,y in zip(w_target,[i*rate for i in  w_loss(c_pos,c_neg,w_target)])]
            c_pos = [x-y for x,y in zip(c_pos,[i*rate for i in pos_loss(c_pos,w_target)])]
            c_neg_1 = [x-y for x,y in zip(c_neg_1,[i*rate for i in neg_loss(c_neg_1,w_target)])]
            c_neg_2 = [x-y for x,y in zip(c_neg_2,[i*rate for i in neg_loss(c_neg_2,w_target)])]
            c_neg_3 = [x-y for x,y in zip(c_neg_3,[i*rate for i in neg_loss(c_neg_3,w_target)])]
            c_neg_4 = [x-y for x,y in zip(c_neg_4,[i*rate for i in neg_loss(c_neg_4,w_target)])]
            c_neg_5 = [x-y for x,y in zip(c_neg_5,[i*rate for i in neg_loss(c_neg_5,w_target)])]
            
            #Calculate loss
            total_loss += -math.log(sigmoid(c_pos,w_target))
            total_loss += -math.log(sigmoid(c_neg_1,w_target))
            total_loss += -math.log(sigmoid(c_neg_2,w_target))
            total_loss += -math.log(sigmoid(c_neg_3,w_target))
            total_loss += -math.log(sigmoid(c_neg_4,w_target))
            total_loss += -math.log(sigmoid(c_neg_5,w_target))

            #Update W and C Matrices
            w[target] = w_target
            c[pos] = c_pos
            c[temp_list[j][0][1]] = c_neg_1
            c[temp_list[j][1][1]] = c_neg_2
            c[temp_list[j][2][1]] = c_neg_3
            c[temp_list[j][3][1]] = c_neg_4
            c[temp_list[j][4][1]] = c_neg_5
    return total_loss/len(target_words)

pd.set_option('display.max_rows',None)
#Instantiate all lists
Sadness_Sentences_T = []
Joy_Sentences_T = []
Sadness_Joy_Sentences_T = []
Sadness_Joy_Fear_Sentences_T = []
Sadness_Sentences_WT = []
Joy_Sentences_WT = []
Sadness_Joy_Sentences_WT = []
Sadness_Joy_Fear_Sentences_WT = []

Sadness_Sentences = df['Sadness Sentences']
Joy_Sentences = df['Joy Sentences']
Sadness_Joy_Sentences = df['Sadness + Joy Sentences']
Sadness_Joy_Fear_Sentences = df['Sadness + Joy + Fear Sentences']
   
for i in range(len(Sadness_Sentences)): #Tokenize all document into sentences
    Sadness_Sentences_T.append(fix_sen_tokenize(Sadness_Sentences[i]))
    Joy_Sentences_T.append(fix_sen_tokenize(Joy_Sentences[i]))
    Sadness_Joy_Sentences_T.append(fix_sen_tokenize(Sadness_Joy_Sentences[i]))
    Sadness_Joy_Fear_Sentences_T.append(fix_sen_tokenize(Sadness_Joy_Fear_Sentences[i]))

#NaN Locations: Sadness+Joy+Fear 7, Sadness+Joy+Fear 48 
Sadness_Joy_Fear_Sentences_T.remove([]) #Removes NaN at Sadness_Joy_Fear_Sentences_T[7]
Sadness_Joy_Fear_Sentences_T.remove([]) #Removes NaN at Sadness_Joy_Fear_Sentences_T[48]

#Flatten out array
Sadness_Sentences_T = [element for sublist in Sadness_Sentences_T for element in sublist]
Joy_Sentences_T = [element for sublist in Joy_Sentences_T for element in sublist]
Sadness_Joy_Sentences_T = [element for sublist in Sadness_Joy_Sentences_T for element in sublist]
Sadness_Joy_Fear_Sentences_T = [element for sublist in Sadness_Joy_Fear_Sentences_T for element in sublist]

#Remove all punctuation and any tabs, whitespace, and new lines
for i in range(len(Sadness_Sentences_T)): #From https://www.geeksforgeeks.org/python-remove-punctuation-from-string/ and https://stackoverflow.com/questions/10711116/strip-spaces-tabs-newlines-python
    Sadness_Sentences_T[i] = ' '.join(Sadness_Sentences_T[i].translate(str.maketrans("","",string.punctuation)).split())
for i in range(len(Joy_Sentences_T)):
    Joy_Sentences_T[i] = ' '.join(Joy_Sentences_T[i].translate(str.maketrans("","",string.punctuation)).split())
for i in range(len(Sadness_Joy_Sentences_T)):
    Sadness_Joy_Sentences_T[i] = ' '.join(Sadness_Joy_Sentences_T[i].translate(str.maketrans("","",string.punctuation)).split())
for i in range(len(Sadness_Joy_Fear_Sentences_T)):
    Sadness_Joy_Fear_Sentences_T[i] = ' '.join(Sadness_Joy_Fear_Sentences_T[i].translate(str.maketrans("","",string.punctuation)).split())
    
#Word tokenization of all sentences
for i in range(len(Sadness_Sentences_T)):
    Sadness_Sentences_WT.append(fix_tokenize(Sadness_Sentences_T[i]))
for i in range(len(Joy_Sentences_T)):
    Joy_Sentences_WT.append(fix_tokenize(Joy_Sentences_T[i]))
for i in range(len(Sadness_Joy_Sentences_T)):
    Sadness_Joy_Sentences_WT.append(fix_tokenize(Sadness_Joy_Sentences_T[i]))
for i in range(len(Sadness_Joy_Fear_Sentences_T)):
    Sadness_Joy_Fear_Sentences_WT.append(fix_tokenize(Sadness_Joy_Fear_Sentences_T[i]))

#Adds sentences with Multiple Emotions into single emotion categories
Sadness_Sentences_WT += Sadness_Joy_Sentences_WT
Joy_Sentences_WT += Sadness_Joy_Sentences_WT
Sadness_Sentences_WT += Sadness_Joy_Fear_Sentences_WT
Joy_Sentences_WT += Sadness_Joy_Fear_Sentences_WT

Combined_Sentences_WT = Sadness_Sentences_WT + Joy_Sentences_WT

#Remove all non word elements
Combined_Sentences_WT = [[word for word in inner if re.compile(r'^[a-zA-Z]+$').match(word)] for inner in Combined_Sentences_WT]
    
#Create a dictionary for unique words and assign to a unique integer
unique_words = {}
index = 0
for sentence in Combined_Sentences_WT:
    for word in sentence:
        if word.lower() not in unique_words:
            unique_words[word.lower()] = index
            index += 1
            
#Remove all non alphabetic words like apostrophes from https://stackoverflow.com/questions/40362857/how-to-remove-non-alphanumeric-characters-in-a-dictionary-using-python
unique_words = {key: value for key, value in unique_words.items() if key.isalpha()}

print("|V|:",len(unique_words))

w = []
c = []
#Create W and C lists and initialize to 1
for vocab_size in range(len(unique_words)):
    empty = []
    for d in range(5):
        empty.append(1)
    w.append(empty)
    c.append(empty)
    
#Create Weighted Unigram Probabilities
unigram_probabilities = {}

#Initialize counts of all unique words to 0
for sentence in Combined_Sentences_WT:
    for i in sentence:
        if i.isalpha():
            unigram_probabilities[i.lower()] = 0
#Count number of occurence of word   
for sentence in Combined_Sentences_WT:
    for i in sentence:
        if i.isalpha():
            unigram_probabilities[i.lower()] += 1
            
#Total number of words in the corpus (Only words, no punctuation)
total = sum(unigram_probabilities.values())  
for word in unigram_probabilities:
    unigram_probabilities[word] = math.pow(unigram_probabilities[word],0.75)/math.pow(total,0.75)
    #Calculate weighted probabilities using alpha = .75
  
#Sort dictionary  by largest values first  
unigram_probabilities = sorted(unigram_probabilities.items(), key=lambda x:x[1], reverse = True)
unigram_probabilities = dict(unigram_probabilities)

positive_examples = []
negative_examples = []
target_words = []
repeated_target_words = []
positive_and_target_blacklist = []

#Generate positive examples
for sentence in Combined_Sentences_WT:
    for i,word in enumerate(sentence):
        if (word.lower() not in repeated_target_words) and (word.lower() in unique_words):
            repeated_target_words.append(word.lower())
            positive_sentence = []
            target_pos_sentence = []
            target_pos_sentence.append(word.lower())
            target_words.append(word)
            if (i == 0):
                positive_sentence.append([word,sentence[1]])
                positive_sentence.append([word,sentence[2]])
                target_pos_sentence.append(sentence[1])
                target_pos_sentence.append(sentence[2])
                positive_examples.append(positive_sentence)
                
            elif (i == 1):
                positive_sentence.append([word,sentence[0]])
                positive_sentence.append([word,sentence[2]])
                positive_sentence.append([word,sentence[3]]) 
                target_pos_sentence.append(sentence[0])
                target_pos_sentence.append(sentence[2])
                target_pos_sentence.append(sentence[3])
                positive_examples.append(positive_sentence)
            elif (i == len(sentence)-1):  
                positive_sentence.append([word,sentence[len(sentence)-2]])
                positive_sentence.append([word,sentence[len(sentence)-3]])     
                target_pos_sentence.append(sentence[len(sentence)-2])
                target_pos_sentence.append(sentence[len(sentence)-3])
                positive_examples.append(positive_sentence)
            elif (i == len(sentence)-2):  
                positive_sentence.append([word,sentence[len(sentence)-1]])
                positive_sentence.append([word,sentence[len(sentence)-3]])
                positive_sentence.append([word,sentence[len(sentence)-4]])   
                target_pos_sentence.append(sentence[len(sentence)-1])
                target_pos_sentence.append(sentence[len(sentence)-3])
                target_pos_sentence.append(sentence[len(sentence)-4])   
                positive_examples.append(positive_sentence)
            else:
                positive_sentence.append([word,sentence[i-1]])
                positive_sentence.append([word,sentence[i-2]])
                positive_sentence.append([word,sentence[i+1]]) 
                positive_sentence.append([word,sentence[i+2]]) 
                target_pos_sentence.append(sentence[i-1])
                target_pos_sentence.append(sentence[i-2])
                target_pos_sentence.append(sentence[i+1])  
                target_pos_sentence.append(sentence[i+2])  
                positive_examples.append(positive_sentence)
            positive_and_target_blacklist.append(target_pos_sentence)

#Generate Negative Examples
for i,examples in enumerate(positive_examples):
    negative_sentence = []
    for key in unigram_probabilities:
        if key in positive_and_target_blacklist[i] or key.lower() in positive_and_target_blacklist[i]:
            continue
        negative_sentence.append([target_words[i],key])
        if len(negative_sentence) == (len(examples)*5):
            break
    negative_examples.append(negative_sentence)
    
#Section 2
print("Target Word:", target_words[10])
print("Positive Target Word Pair:", positive_examples[10])
print("Negative Target Word Pair:", negative_examples[10])

#Convert positive_examples and negative_examples to their corresponding numbers from unique_words dictionary
positive_examples_num = []
negative_examples_num = []

for examples in positive_examples:
    temp_example = []
    for example in examples:
        temp_example.append([unique_words[example[0].lower()],unique_words[example[1].lower()]])
    positive_examples_num.append(temp_example)
    
for examples in negative_examples:
    temp_example = []
    for example in examples:
        temp_example.append([unique_words[example[0].lower()],unique_words[example[1].lower()]])
    negative_examples_num.append(temp_example)
   
rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]    
for rate in rates:      
    print("Rate",rate,"Validation Loss:",stochastic_gradient_descent(w,c,rate))
    w,c = reset_w_c()
#Set W and C
stochastic_gradient_descent(w,c,.00001)

combined_w_c = [x+y for x,y in zip(w,c)]
selected_words = ["happy","happily","delighted","rekindle","laughed","joy","wrong","lonely","dying","agony","suffering","weeps","weeping"]
selected_w_c_1 = [w[unique_words["happy"]],w[unique_words["happily"]],w[unique_words["delighted"]],w[unique_words["rekindle"]],w[unique_words["laughed"]],w[unique_words["joy"]],w[unique_words["wrong"]],w[unique_words["lonely"]],w[unique_words["dying"]],w[unique_words["agony"]],w[unique_words["suffering"]],w[unique_words["weeps"]],w[unique_words["weeping"]]]
selected_w_c_2 = [c[unique_words["happy"]],c[unique_words["happily"]],c[unique_words["delighted"]],c[unique_words["rekindle"]],c[unique_words["laughed"]],c[unique_words["joy"]],c[unique_words["wrong"]],c[unique_words["lonely"]],c[unique_words["dying"]],c[unique_words["agony"]],c[unique_words["suffering"]],c[unique_words["weeps"]],c[unique_words["weeping"]]]
selected_w_c = [x+y for x,y in zip(selected_w_c_1,selected_w_c_2)]

w_2d= TSNE(n_components=2).fit_transform(combined_w_c)
plt.figure(figsize=(6,6))
plt.scatter(w_2d[:,0],w_2d[:,1])
for i,word in enumerate(target_words):
    plt.annotate(word,xy=(w_2d[i,0],w_2d[i,1]))
plt.show()

w_2d= TSNE(n_components=2).fit_transform(selected_w_c)
plt.figure(figsize=(6,6))
plt.scatter(w_2d[:,0],w_2d[:,1])
for i,word in enumerate(selected_words):
    plt.annotate(word,xy=(w_2d[i,0],w_2d[i,1]))
plt.show()
