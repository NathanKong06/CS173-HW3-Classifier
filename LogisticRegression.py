import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from nltk.stem import PorterStemmer
import math

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

filepath = "/Users/nathan/Desktop/data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t', keep_default_na=False)
joy_list = list(emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'joy')].word)
sad_list = list(emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'sadness')].word)


def fix_tokenize(text): #Word tokenizes sentence
    if isinstance(text,float): #If blank cell (NaN)
        text = ''
    return word_tokenize(text)
  
def num_joy(text, lexicons, index):
    joy = 0
    lexs=0
    for word in text:
        if word in joy_list:
            joy += 1
    for lex in lexicons[index]:
        if lex in joy_list:
            lexs+=1
    return max(joy,lexs)
    
def num_sad(text, lexicons,index):
    sad = 0
    lexs = 0
    for word in text:
        if word in sad_list:
            sad += 1 
    for lex in lexicons[index]:
        if lex in sad_list:
            lexs+=1
    return max(sad,lexs)
          
def sigmoid(feature,weights,b): #Section 3.1
    try:
        return (1.0/ (1.0 + math.exp(-(weights[0]*feature[0] + weights[1]*feature[1] + weights[2]*feature[2] + b))))
    except: #Handle overflow issues
        return 1- (1.0/ (1.0 + math.exp(weights[0]*feature[0] + weights[1]*feature[1] + weights[2]*feature[2] + b)))
    
def loss(feature,weights,b,y):
    if y == 1:
        return -math.log(sigmoid(feature,weights,b))
    else:
        if (1-sigmoid(feature,weights,b) == 0): #If 0 due to rounding error, return small number
            return 1e-100
        return -math.log(1-sigmoid(feature,weights,b))
#     return -(y*math.log(sigmoid(feature,weights,b)) + (1-y)*math.log(1-sigmoid(feature,weights,b)))
pd.set_option('display.max_rows',None)
#Instantiate all lists
Sadness_Sentences_T = []
Joy_Sentences_T = []
Sadness_Joy_Sentences_T = []
Sadness_Joy_Fear_Sentences_T = []
Sadness_Lexicons_T = []
Joy_Lexicons_T = []
Sadness_Joy_Lexicons_T = []
Sadness_Joy_Fear_Lexicons_T = []

Sadness_Lexicons = df['Sadness Lexicons']
Sadness_Sentences = df['Sadness Sentences']
Joy_Lexicons = df['Joy Lexicons']
Joy_Sentences = df['Joy Sentences']
Sadness_Joy_Lexicons = df['Sadness + Joy Lexicons']
Sadness_Joy_Sentences = df['Sadness + Joy Sentences']
Sadness_Joy_Fear_Lexicons = df['Sadness + Joy + Fear Lexicons']
Sadness_Joy_Fear_Sentences = df['Sadness + Joy + Fear Sentences']
   
#Tokenize all Sentences and Lexicons
for i in range(len(Sadness_Sentences)):
    Sadness_Sentences_T.append(fix_tokenize(Sadness_Sentences[i]))
    Joy_Sentences_T.append(fix_tokenize(Joy_Sentences[i]))
    Sadness_Joy_Sentences_T.append(fix_tokenize(Sadness_Joy_Sentences[i]))
    Sadness_Joy_Fear_Sentences_T.append(fix_tokenize(Sadness_Joy_Fear_Sentences[i]))
    Sadness_Lexicons_T.append(fix_tokenize(Sadness_Lexicons[i]))
    Joy_Lexicons_T.append(fix_tokenize(Joy_Lexicons[i]))
    Sadness_Joy_Lexicons_T.append(fix_tokenize(Sadness_Joy_Lexicons[i]))
    Sadness_Joy_Fear_Lexicons_T.append(fix_tokenize(Sadness_Joy_Fear_Lexicons[i]))
    
#NaN Locations: Sadness+Joy+Fear 7, Sadness+Joy+Fear 48

#Take first 30 elements only
Sadness_Sentences_T = Sadness_Sentences_T[:30]
Joy_Sentences_T = Joy_Sentences_T[:30]
Sadness_Joy_Sentences_T = Sadness_Joy_Sentences_T[:30]
Sadness_Joy_Fear_Sentences_T = Sadness_Joy_Fear_Sentences_T[:30]    
Sadness_Lexicons_T = Sadness_Lexicons_T [:30]
Joy_Lexicons_T = Joy_Lexicons_T [:30]
Sadness_Joy_Lexicons_T = Sadness_Joy_Lexicons_T [:30]
Sadness_Joy_Fear_Lexicons_T= Sadness_Joy_Fear_Lexicons_T [:30] 

#Remove all NaN from rows 1-30 from arrays
Sadness_Joy_Fear_Sentences_T.remove([]) #Removes NaN at Sadness_Joy_Fear_Sentences_T[7]
Sadness_Joy_Fear_Lexicons_T.remove([]) #Removes NaN at Sadness_Joy_Fear_Lexicons_T[7]

#Adds sentences with Multiple Emotions into single emotion categories
Sadness_Sentences_T += Sadness_Joy_Sentences_T
Joy_Sentences_T += Sadness_Joy_Sentences_T
Sadness_Sentences_T += Sadness_Joy_Fear_Sentences_T
Joy_Sentences_T += Sadness_Joy_Fear_Sentences_T
Sadness_Lexicons_T += Sadness_Joy_Lexicons_T
Joy_Lexicons_T+= Sadness_Joy_Lexicons_T
Sadness_Lexicons_T += Sadness_Joy_Fear_Lexicons_T
Joy_Lexicons_T += Sadness_Joy_Fear_Lexicons_T

#Remove commas in lexicons
for document in Sadness_Lexicons_T:
    while ',' in document: 
        document.remove(',')
        
for document in Joy_Lexicons_T:
    while ',' in document: 
        document.remove(',')

#Stemm all words
for i in range(len(Sadness_Sentences_T)):
    for j in range(len(Sadness_Sentences_T[i])):
        Sadness_Sentences_T[i][j] = PorterStemmer().stem(Sadness_Sentences_T[i][j])
        
for i in range(len(Joy_Sentences_T)):
    for j in range(len(Joy_Sentences_T[i])):
        Joy_Sentences_T[i][j] = PorterStemmer().stem(Joy_Sentences_T[i][j])

Features= []
Features_Y = []

for i,document in enumerate(Sadness_Sentences_T):
    Feature = [-1,-1,-1]
    Feature[0] = num_joy(document,Sadness_Lexicons_T,i)
    Feature[1] = num_sad(document,Sadness_Lexicons_T,i)
    Feature[2] = len(document)
    Features.append(Feature)
    Features_Y.append("0")
    
for i,document in enumerate(Joy_Sentences_T):
    Feature = [-1,-1,-1]
    Feature[0] = num_joy(document,Joy_Lexicons_T,i)
    Feature[1] = num_sad(document,Joy_Lexicons_T,i)
    Feature[2] = len(document)
    Features.append(Feature)
    Features_Y.append("1")
    
#Section 2.2
print(Features)
#Section 3.2
print((loss(Features[0],[0,0,0],0,0)+loss(Features[math.floor((len(Features)-1)/2)],[0,0,0],0,1)/2))

Sadness_Sentences_T_V= []
Joy_Sentences_T_V = []
Sadness_Joy_Sentences_T_V = []
Sadness_Joy_Fear_Sentences_T_V = []
Sadness_Lexicons_T_V = []
Joy_Lexicons_T_V = []
Sadness_Joy_Lexicons_T_V = []
Sadness_Joy_Fear_Lexicons_T_V = []

Sadness_Lexicons_V = df['Sadness Lexicons']
Sadness_Sentences_V = df['Sadness Sentences']
Joy_Lexicons_V = df['Joy Lexicons']
Joy_Sentences_V = df['Joy Sentences']
Sadness_Joy_Lexicons_V = df['Sadness + Joy Lexicons']
Sadness_Joy_Sentences_V = df['Sadness + Joy Sentences']
Sadness_Joy_Fear_Lexicons_V = df['Sadness + Joy + Fear Lexicons']
Sadness_Joy_Fear_Sentences_V = df['Sadness + Joy + Fear Sentences']

#Tokenize all Sentences and Lexicons
for i in range(len(Sadness_Sentences_V)):
    Sadness_Sentences_T_V.append(fix_tokenize(Sadness_Sentences_V[i]))
    Joy_Sentences_T_V.append(fix_tokenize(Joy_Sentences_V[i]))
    Sadness_Joy_Sentences_T_V.append(fix_tokenize(Sadness_Joy_Sentences_V[i]))
    Sadness_Joy_Fear_Sentences_T_V.append(fix_tokenize(Sadness_Joy_Fear_Sentences_V[i]))
    Sadness_Lexicons_T_V.append(fix_tokenize(Sadness_Lexicons_V[i]))
    Joy_Lexicons_T_V.append(fix_tokenize(Joy_Lexicons_V[i]))
    Sadness_Joy_Lexicons_T_V.append(fix_tokenize(Sadness_Joy_Lexicons_V[i]))
    Sadness_Joy_Fear_Lexicons_T_V.append(fix_tokenize(Sadness_Joy_Fear_Lexicons_V[i]))
    
#Take first 31-40 elements only
Sadness_Sentences_T_V = Sadness_Sentences_T_V[30:40]
Joy_Sentences_T_V = Joy_Sentences_T_V[30:40]
Sadness_Joy_Sentences_T_V = Sadness_Joy_Sentences_T_V[30:40]
Sadness_Joy_Fear_Sentences_T_V = Sadness_Joy_Fear_Sentences_T_V[30:40]  
Sadness_Lexicons_T_V = Sadness_Lexicons_T_V [30:40]
Joy_Lexicons_T_V = Joy_Lexicons_T_V [30:40]
Sadness_Joy_Lexicons_T_V = Sadness_Joy_Lexicons_T_V [30:40]
Sadness_Joy_Fear_Lexicons_T_V= Sadness_Joy_Fear_Lexicons_T_V [30:40]

Sadness_Sentences_T_V += Sadness_Joy_Sentences_T_V
Joy_Sentences_T_V += Sadness_Joy_Sentences_T_V
Sadness_Sentences_T_V += Sadness_Joy_Fear_Sentences_T_V
Joy_Sentences_T_V += Sadness_Joy_Fear_Sentences_T_V
Sadness_Lexicons_T_V += Sadness_Joy_Lexicons_T_V
Joy_Lexicons_T_V+= Sadness_Joy_Lexicons_T_V
Sadness_Lexicons_T_V += Sadness_Joy_Fear_Lexicons_T_V
Joy_Lexicons_T_V += Sadness_Joy_Fear_Lexicons_T_V

#Remove commas in lexicons
for document in Sadness_Lexicons_T_V:
    while ',' in document: 
        document.remove(',')
        
for document in Joy_Lexicons_T_V:
    while ',' in document: 
        document.remove(',')

#Stemm all words
for i in range(len(Sadness_Sentences_T_V)):
    for j in range(len(Sadness_Sentences_T_V[i])):
        Sadness_Sentences_T_V[i][j] = PorterStemmer().stem(Sadness_Sentences_T_V[i][j])
        
for i in range(len(Joy_Sentences_T_V)):
    for j in range(len(Joy_Sentences_T_V[i])):
        Joy_Sentences_T_V[i][j] = PorterStemmer().stem(Joy_Sentences_T_V[i][j])

Features_V = []
Features_V_Y = []

for i,document in enumerate(Sadness_Sentences_T_V):
    Feature = [-1,-1,-1]
    Feature[0] = num_joy(document,Sadness_Lexicons_T_V,i)
    Feature[1] = num_sad(document,Sadness_Lexicons_T_V,i)
    Feature[2] = len(document)
    Features_V.append(Feature)
    Features_V_Y.append("0")
    
for i,document in enumerate(Joy_Sentences_T_V):
    Feature = [-1,-1,-1]
    Feature[0] = num_joy(document,Joy_Lexicons_T_V,i)
    Feature[1] = num_sad(document,Joy_Lexicons_T_V,i)
    Feature[2] = len(document)
    Features_V.append(Feature)
    Features_V_Y.append("1")
    
def stochastic_gradient_descent(training_features,training_y,validation_features,validation_y,rate):
    weight = [0,0,0] #Initialize weight to 0,0,0
    b = 0 #Initialize bias to 0
    
    for index in range(1000): #Loop many times
        for i in range(len(training_features)):
            prob = sigmoid(training_features[i],weight,b) #Predicted probs 
            gradient_bias = prob - float(training_y[i])
            gradient_weights = []
            for element in range(len(weight)):
                gradient = training_features[i][element] * (prob - float(training_y[i]))
                gradient_weights.append(gradient)
            b = b - (rate * gradient_bias)
            for element in range(len(weight)):
                weight[element] = weight[element] - rate * gradient_weights[element]
        loss_counter = 0
        for document in range(len(validation_features)):
            loss_counter += loss(validation_features[document],weight,b,float(validation_y[document]))
    return loss_counter/len(validation_features) #Average loss

rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for rate in rates:
    print("Rate",rate,"Validation Loss:",stochastic_gradient_descent(Features,Features_Y,Features_V,Features_V_Y,rate))
    
Sadness_Sentences_T_T= []
Joy_Sentences_T_T = []
Sadness_Joy_Sentences_T_T = []
Sadness_Joy_Fear_Sentences_T_T = []
Sadness_Lexicons_T_T = []
Joy_Lexicons_T_T = []
Sadness_Joy_Lexicons_T_T = []
Sadness_Joy_Fear_Lexicons_T_T = []

Sadness_Lexicons_Test = df['Sadness Lexicons']
Sadness_Sentences_Test = df['Sadness Sentences']
Joy_Lexicons_Test = df['Joy Lexicons']
Joy_Sentences_Test = df['Joy Sentences']
Sadness_Joy_Lexicons_Test = df['Sadness + Joy Lexicons']
Sadness_Joy_Sentences_Test = df['Sadness + Joy Sentences']
Sadness_Joy_Fear_Lexicons_Test = df['Sadness + Joy + Fear Lexicons']
Sadness_Joy_Fear_Sentences_Test = df['Sadness + Joy + Fear Sentences']

#Tokenize all Sentences and Lexicons
for i in range(len(Sadness_Sentences_Test)):
    Sadness_Sentences_T_T.append(fix_tokenize(Sadness_Sentences_Test[i]))
    Joy_Sentences_T_T.append(fix_tokenize(Joy_Sentences_Test[i]))
    Sadness_Joy_Sentences_T_T.append(fix_tokenize(Sadness_Joy_Sentences_Test[i]))
    Sadness_Joy_Fear_Sentences_T_T.append(fix_tokenize(Sadness_Joy_Fear_Sentences_Test[i]))
    Sadness_Lexicons_T_T.append(fix_tokenize(Sadness_Lexicons_Test[i]))
    Joy_Lexicons_T_T.append(fix_tokenize(Joy_Lexicons_Test[i]))
    Sadness_Joy_Lexicons_T_T.append(fix_tokenize(Sadness_Joy_Lexicons_Test[i]))
    Sadness_Joy_Fear_Lexicons_T_T.append(fix_tokenize(Sadness_Joy_Fear_Lexicons_Test[i]))
    
#Take last 10 eleme ts
Sadness_Sentences_T_T = Sadness_Sentences_T_T[-10:]
Joy_Sentences_T_T = Joy_Sentences_T_T[-10:]
Sadness_Joy_Sentences_T_T = Sadness_Joy_Sentences_T_T[-10:]
Sadness_Joy_Fear_Sentences_T_T = Sadness_Joy_Fear_Sentences_T_T[-10:]
Sadness_Lexicons_T_T = Sadness_Lexicons_T_T [-10:]
Joy_Lexicons_T_T = Joy_Lexicons_T_T[-10:]
Sadness_Joy_Lexicons_T_T = Sadness_Joy_Lexicons_T_T [-10:]
Sadness_Joy_Fear_Lexicons_T_T= Sadness_Joy_Fear_Lexicons_T_T [-10:]

Sadness_Joy_Fear_Sentences_T_T.remove([]) #Removes NaN at Sadness_Joy_Fear_Sentences_T[48]
Sadness_Joy_Fear_Lexicons_T_T.remove([]) #Removes NaN at Sadness_Joy_Fear_Lexicons_T[48]

Sadness_Sentences_T_T += Sadness_Joy_Sentences_T_T
Joy_Sentences_T_T += Sadness_Joy_Sentences_T_T
Sadness_Sentences_T_T += Sadness_Joy_Fear_Sentences_T_T
Joy_Sentences_T_T += Sadness_Joy_Fear_Sentences_T_T
Sadness_Lexicons_T_T += Sadness_Joy_Lexicons_T_T
Joy_Lexicons_T_T += Sadness_Joy_Lexicons_T_T
Sadness_Lexicons_T_T += Sadness_Joy_Fear_Lexicons_T_T
Joy_Lexicons_T_T += Sadness_Joy_Fear_Lexicons_T_T

#Remove commas in lexicons
for document in Sadness_Lexicons_T_T:
    while ',' in document: 
        document.remove(',')
        
for document in Joy_Lexicons_T_T:
    while ',' in document: 
        document.remove(',')

#Stemm all words
for i in range(len(Sadness_Sentences_T_T)):
    for j in range(len(Sadness_Sentences_T_T[i])):
        Sadness_Sentences_T_T[i][j] = PorterStemmer().stem(Sadness_Sentences_T_T[i][j])
        
for i in range(len(Joy_Sentences_T_T)):
    for j in range(len(Joy_Sentences_T_T[i])):
        Joy_Sentences_T_T[i][j] = PorterStemmer().stem(Joy_Sentences_T_T[i][j])

Features_T = []
Features_T_Y = []

for i,document in enumerate(Sadness_Sentences_T_T):
    Feature = [-1,-1,-1]
    Feature[0] = num_joy(document,Sadness_Lexicons_T_T,i)
    Feature[1] = num_sad(document,Sadness_Lexicons_T_T,i)
    Feature[2] = len(document)
    Features_T.append(Feature)
    Features_T_Y.append("0")
    
for i,document in enumerate(Joy_Sentences_T_T):
    Feature = [-1,-1,-1]
    Feature[0] = num_joy(document,Joy_Lexicons_T_T,i)
    Feature[1] = num_sad(document,Joy_Lexicons_T_T,i)
    Feature[2] = len(document)
    Features_T.append(Feature)
    Features_T_Y.append("1")
    
def stochastic_gradient_descent_weight_bias(training_features,training_y,rate):
    weight = [0,0,0] #Initialize weight to 0,0,0
    b = 0 #Initialize bias to 0
    
    for index in range(1000): #Loop many times
        for i in range(len(training_features)):
            prob = sigmoid(training_features[i],weight,b) #Predicted probs 
            gradient_bias = prob - float(training_y[i])
            gradient_weights = []
            for element in range(len(weight)):
                gradient = training_features[i][element] * (prob - float(training_y[i]))
                gradient_weights.append(gradient)
            b = b - (rate * gradient_bias)
            for element in range(len(weight)):
                weight[element] = weight[element] - rate * gradient_weights[element]
    return weight,b

weight_and_bias = stochastic_gradient_descent_weight_bias(Features,Features_Y,rate=.00001)

w1 = weight_and_bias[0]
b1 = weight_and_bias[1]
predicted = []
for document in Features_T:
    probability = sigmoid(document,w1,b1)
    if probability > .5:
        predicted.append('1')
    else:
        predicted.append('0')

print(confusion_matrix(Features_T_Y,predicted))
print(classification_report(Features_T_Y,predicted))
