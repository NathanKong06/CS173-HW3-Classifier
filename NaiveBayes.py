import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
df = pd.read_csv(r"/CS173-published-sheet - Sheet1.csv")
#Remove all newline characters
df = df.replace({r'\n':''},regex = True) 

def fix_tokenize(text): #Word tokenizes sentence
    if isinstance(text,float): #If blank cell (NaN)
        text = ''
    return word_tokenize(text)

def classifier(Sentence_T):
    fear_prob = pfear#Probability of fear
    for word in Sentence_T:
        for i,mle in enumerate(fear_mle_words_list):
            if word == mle[0]: #If word is in fear MLE
                fear_prob = fear_prob * mle[1] #multiply with probability
                break
        else: #Word is not in fear MLE
            fear_prob = fear_prob * (1/(fear_num_words + fear_unique_words))#Multily by 1/(#words in training set + # of vocabulary)
    anger_prob = panger
    for word in Sentence_T:
        for i,mle in enumerate(anger_mle_words_list):
            if word == mle[0]: #If word is in MLE
                anger_prob = anger_prob * mle[1]
                break
        else:
            anger_prob = anger_prob * (1/(anger_num_words + anger_unique_words))
    surprise_prob = psurprise
    for word in Sentence_T:
        for i,mle in enumerate(surprise_mle_words_list):
            if word == mle[0]: #If word is in MLE
                surprise_prob = surprise_prob * mle[1]
                break
        else:
            surprise_prob = surprise_prob * (1/(surprise_num_words + surprise_unique_words))
    disgust_prob = pdisgust
    for word in Sentence_T:
        for i,mle in enumerate(disgust_mle_words_list):
            if word == mle[0]: #If word is in MLE
                disgust_prob = disgust_prob * mle[1]
                break
        else:
            disgust_prob = disgust_prob * (1/(disgust_num_words + disgust_unique_words))
    sadness_prob = psadness
    for word in Sentence_T:
        for i,mle in enumerate(sadness_mle_words_list):
            if word == mle[0]: #If word is in MLE
                sadness_prob = sadness_prob * mle[1]
                break
        else:
            sadness_prob = sadness_prob * (1/(sadness_num_words + sadness_unique_words))
    joy_prob = pjoy
    for word in Sentence_T:
        for i,mle in enumerate(joy_mle_words_list):
            if word == mle[0]: #If word is in MLE
                joy_prob = joy_prob * mle[1]
                break
        else:
            joy_prob = joy_prob * (1/(joy_num_words + joy_unique_words))
    emotion = max(fear_prob,anger_prob,surprise_prob,disgust_prob,sadness_prob,joy_prob) #Find most likely emotion
    if emotion == fear_prob:
        return "Fear"
    elif emotion == anger_prob:
        return "Anger"
    elif emotion == surprise_prob:
        return "Surprise"
    elif emotion == disgust_prob:
        return "Disgust"
    elif emotion == sadness_prob:
        return "Sadness"
    elif emotion == joy_prob:
        return "Joy"
        
pd.set_option('display.max_rows',None)
#Instantiate all lists
Sadness_Sentences_T = []
Joy_Sentences_T = []
Fear_Sentences_T = []
Anger_Sentences_T = []
Surprise_Sentences_T = []
Disgust_Sentences_T = []
Sadness_Joy_Sentences_T = []
Fear_Anger_Sentences_T = []
Surprise_Disgust_Sentences_T = []
Sadness_Joy_Fear_Sentences_T = []

Sadness_Sentences_T_Testing = []
Joy_Sentences_T_Testing = []
Fear_Sentences_T_Testing = []
Anger_Sentences_T_Testing = []
Surprise_Sentences_T_Testing = []
Disgust_Sentences_T_Testing = []
Sadness_Joy_Sentences_T_Testing = []
Fear_Anger_Sentences_T_Testing = []
Surprise_Disgust_Sentences_T_Testing = []
Sadness_Joy_Fear_Sentences_T_Testing = []

Sadness_Lexicons = df['Sadness Lexicons']
Sadness_Sentences = df['Sadness Sentences']
Joy_Lexicons = df['Joy Lexicons']
Joy_Sentences = df['Joy Sentences']
Fear_Lexicons = df['Fear Lexicons']
Fear_Sentences = df['Fear Sentences']
Anger_Lexicons = df['Anger Lexicons']
Anger_Sentences = df['Anger Sentences']
Surprise_Lexicons = df['Surprise Lexicons']
Surprise_Sentence = df['Surprise Sentence']
Disgust_Lexicons = df['Disgust Lexicons']
Disgust_Sentences = df['Disgust Sentences']
Sadness_Joy_Lexicons = df['Sadness + Joy Lexicons']
Sadness_Joy_Sentences = df['Sadness + Joy Sentences']
Fear_Anger_Lexicons = df['Fear + Anger Lexicons']
Fear_Anger_Sentences = df['Fear + Anger Sentences']
Surprise_Disgust_Lexicons = df['Surprise + Disgust Lexicons']
Surprise_Disgust_Sentences = df['Surprise + Disgust Sentences']
Sadness_Joy_Fear_Lexicons = df['Sadness + Joy + Fear Lexicons']
Sadness_Joy_Fear_Sentences = df['Sadness + Joy + Fear Sentences']

#Tokenize all Sentences
for i in range(len(Sadness_Sentences)):
    Sadness_Sentences_T.append(fix_tokenize(Sadness_Sentences[i]))
    Joy_Sentences_T.append(fix_tokenize(Joy_Sentences[i]))
    Fear_Sentences_T.append(fix_tokenize(Fear_Sentences[i]))
    Anger_Sentences_T.append(fix_tokenize(Anger_Sentences[i]))
    Surprise_Sentences_T.append(fix_tokenize(Surprise_Sentence[i]))
    Disgust_Sentences_T.append(fix_tokenize(Disgust_Sentences[i]))
    Sadness_Joy_Sentences_T.append(fix_tokenize(Sadness_Joy_Sentences[i]))
    Fear_Anger_Sentences_T.append(fix_tokenize(Fear_Anger_Sentences[i]))
    Surprise_Disgust_Sentences_T.append(fix_tokenize(Surprise_Disgust_Sentences[i]))
    Sadness_Joy_Fear_Sentences_T.append(fix_tokenize(Sadness_Joy_Fear_Sentences[i]))
    
#Section 2.2
print("Sadness: ",Sadness_Sentences_T[0])
print("Joy: ",Joy_Sentences_T[0])
print("Fear: ",Fear_Sentences_T[0])
print("Anger: ",Anger_Sentences_T[0])
print("Surprise: ",Surprise_Sentences_T[0])
print("Disgust: ",Disgust_Sentences_T[0])
print("Sadness + Joy: ",Sadness_Joy_Sentences_T[0])
print("Fear + Anger: ",Fear_Anger_Sentences_T[0])
print("Surprise + Disgust: ",Surprise_Disgust_Sentences_T[0])
print("Sadness + Joy + Fear: ",Sadness_Joy_Fear_Sentences_T[0])

#NaN Locations: Surprise 11, Surprise+Disgust 7, Sadness+Joy+Fear 7, Surprise+Disgust 19, Surprise+Disgust 48, Sadness+Joy+Fear 48

#Take first 30 elements only
Sadness_Sentences_T = Sadness_Sentences_T[:30]
Joy_Sentences_T = Joy_Sentences_T[:30]
Fear_Sentences_T = Fear_Sentences_T[:30]
Anger_Sentences_T = Anger_Sentences_T[:30]
Surprise_Sentences_T = Surprise_Sentences_T[:30]
Disgust_Sentences_T = Disgust_Sentences_T[:30]
Sadness_Joy_Sentences_T = Sadness_Joy_Sentences_T[:30]
Fear_Anger_Sentences_T = Fear_Anger_Sentences_T[:30]
Surprise_Disgust_Sentences_T = Surprise_Disgust_Sentences_T[:30]
Sadness_Joy_Fear_Sentences_T = Sadness_Joy_Fear_Sentences_T[:30]    

#Remove all NaN from rows 1-30 from arrays
Surprise_Sentences_T.remove([]) #Removes NaN at Surprise_Sentences_T[11]
Sadness_Joy_Fear_Sentences_T.remove([]) #Removes NaN at Sadness_Joy_Fear_Sentences_T[7]
Surprise_Disgust_Sentences_T.remove([]) #Removes NaN at Surprise_Disgust_Sentences_T[7]
Surprise_Disgust_Sentences_T.remove([]) #Removes NaN at Surprise_Disgust_Sentences_T[19]

#Adds sentences with Multiple Emotions into single emotion categories
Sadness_Sentences_T += Sadness_Joy_Sentences_T
Joy_Sentences_T += Sadness_Joy_Sentences_T
Fear_Sentences_T += Fear_Anger_Sentences_T
Anger_Sentences_T += Fear_Anger_Sentences_T
Surprise_Sentences_T += Surprise_Disgust_Sentences_T
Disgust_Sentences_T += Surprise_Disgust_Sentences_T
Sadness_Sentences_T += Sadness_Joy_Fear_Sentences_T
Joy_Sentences_T += Sadness_Joy_Fear_Sentences_T
Fear_Sentences_T += Sadness_Joy_Fear_Sentences_T

#Section 3.1
total = len(Sadness_Sentences_T) + len(Joy_Sentences_T) + len(Fear_Sentences_T) + len(Anger_Sentences_T) + len(Surprise_Sentences_T) + len(Disgust_Sentences_T)
print("Probability of Fear: ", len(Fear_Sentences_T) / total )
print("Probability of Anger: ", len(Anger_Sentences_T) / total )
print("Probability of Surprise: ", len(Surprise_Sentences_T) / total )
print("Probability of Disgust: ", len(Disgust_Sentences_T) / total )
print("Probability of Sadness: ", len(Sadness_Sentences_T) / total )
print("Probability of Joy: ", len(Joy_Sentences_T) / total )

pfear =  len(Fear_Sentences_T) / total
panger = len(Anger_Sentences_T) / total
psurprise = len(Surprise_Sentences_T) / total
pdisgust = len(Disgust_Sentences_T) / total 
psadness = len(Sadness_Sentences_T) / total
pjoy = len(Joy_Sentences_T) / total

#Flatten out all 6 emotion arrays
Sadness_Sentences_T = [element for sublist in Sadness_Sentences_T for element in sublist]
Joy_Sentences_T = [element for sublist in Joy_Sentences_T for element in sublist]
Fear_Sentences_T = [element for sublist in Fear_Sentences_T for element in sublist]
Anger_Sentences_T = [element for sublist in Anger_Sentences_T for element in sublist]
Surprise_Sentences_T = [element for sublist in Surprise_Sentences_T for element in sublist]
Disgust_Sentences_T = [element for sublist in Disgust_Sentences_T for element in sublist]

###############################################################################################################
for word in Fear_Sentences_T: #Get Frequency of every unique word and store in dictionary
    if word in fear_dictionary: #From https://www.geeksforgeeks.org/find-frequency-of-each-word-in-a-string-in-python/
        fear_dictionary[word] += 1
    else:
        fear_dictionary.update({word: 1})
    
#Sort dictionary in order
fear_dictionary = dict(sorted(fear_dictionary.items(), key=lambda item: item[1]))
#From https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value

#Convert dictionary to have MLE estimates by dividing number of times it appears by total number of words
fear_num_words = len(Fear_Sentences_T)
fear_unique_words = len(fear_dictionary.keys())

#Convert Frequency to MLE with Laplace Smoothing
for word in fear_dictionary:
    fear_dictionary[word] = (fear_dictionary[word] + 1)/(fear_num_words+fear_unique_words)
#Convert dictionary to list named mle_words_list
fear_mle_words_list = list(fear_dictionary.items())
fear_mle_words_list.reverse()
###############################################################################################################
for word in Anger_Sentences_T: #Get Frequency of every unique word and store in dictionary
    if word in anger_dictionary: #From https://www.geeksforgeeks.org/find-frequency-of-each-word-in-a-string-in-python/
        anger_dictionary[word] += 1
    else:
        anger_dictionary.update({word: 1})
    
#Sort dictionary in order
anger_dictionary = dict(sorted(anger_dictionary.items(), key=lambda item: item[1]))
#From https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value

#Convert dictionary to have MLE estimates by dividing number of times it appears by total number of words
anger_num_words = len(Anger_Sentences_T)
anger_unique_words = len(anger_dictionary.keys())

#Convert Frequency to MLE with Laplace Smoothing
for word in anger_dictionary:
    anger_dictionary[word] = (anger_dictionary[word] + 1)/(anger_num_words+anger_unique_words)

#Convert dictionary to list named mle_words_list
anger_mle_words_list = list(anger_dictionary.items())
anger_mle_words_list.reverse()
###############################################################################################################
for word in Surprise_Sentences_T: #Get Frequency of every unique word and store in dictionary
    if word in surprise_dictionary: #From https://www.geeksforgeeks.org/find-frequency-of-each-word-in-a-string-in-python/
        surprise_dictionary[word] += 1
    else:
        surprise_dictionary.update({word: 1})
    
#Sort dictionary in order
surprise_dictionary = dict(sorted(surprise_dictionary.items(), key=lambda item: item[1]))
#From https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value

#Convert dictionary to have MLE estimates by dividing number of times it appears by total number of words
surprise_num_words = len(Surprise_Sentences_T)
surprise_unique_words = len(surprise_dictionary.keys())

#Convert Frequency to MLE with Laplace Smoothing
for word in surprise_dictionary:
    surprise_dictionary[word] = (surprise_dictionary[word] + 1)/(surprise_num_words+surprise_unique_words)

#Convert dictionary to list named mle_words_list
surprise_mle_words_list = list(surprise_dictionary.items())
surprise_mle_words_list.reverse()
###############################################################################################################
for word in Disgust_Sentences_T: #Get Frequency of every unique word and store in dictionary
    if word in disgust_dictionary: #From https://www.geeksforgeeks.org/find-frequency-of-each-word-in-a-string-in-python/
        disgust_dictionary[word] += 1
    else:
        disgust_dictionary.update({word: 1})
    
#Sort dictionary in order
disgust_dictionary = dict(sorted(disgust_dictionary.items(), key=lambda item: item[1]))
#From https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value

#Convert dictionary to have MLE estimates by dividing number of times it appears by total number of words
disgust_num_words = len(Disgust_Sentences_T)
disgust_unique_words = len(disgust_dictionary.keys())

#Convert Frequency to MLE with Laplace Smoothing
for word in disgust_dictionary:
    disgust_dictionary[word] = (disgust_dictionary[word] + 1)/(disgust_num_words+disgust_unique_words)

#Convert dictionary to list named mle_words_list
disgust_mle_words_list = list(disgust_dictionary.items())
disgust_mle_words_list.reverse()
###############################################################################################################
for word in Sadness_Sentences_T: #Get Frequency of every unique word and store in dictionary
    if word in sadness_dictionary: #From https://www.geeksforgeeks.org/find-frequency-of-each-word-in-a-string-in-python/
        sadness_dictionary[word] += 1
    else:
        sadness_dictionary.update({word: 1})
    
#Sort dictionary in order
sadness_dictionary = dict(sorted(sadness_dictionary.items(), key=lambda item: item[1]))
#From https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value

#Convert dictionary to have MLE estimates by dividing number of times it appears by total number of words
sadness_num_words = len(Sadness_Sentences_T)
sadness_unique_words = len(sadness_dictionary.keys())

#Convert Frequency to MLE with Laplace Smoothing
for word in sadness_dictionary:
    sadness_dictionary[word] = (sadness_dictionary[word] + 1)/(sadness_num_words+sadness_unique_words)

#Convert dictionary to list named mle_words_list
sadness_mle_words_list = list(sadness_dictionary.items())
sadness_mle_words_list.reverse()
###############################################################################################################
for word in Joy_Sentences_T: #Get Frequency of every unique word and store in dictionary
    if word in joy_dictionary: #From https://www.geeksforgeeks.org/find-frequency-of-each-word-in-a-string-in-python/
        joy_dictionary[word] += 1
    else:
        joy_dictionary.update({word: 1})
    
#Sort dictionary in order
joy_dictionary = dict(sorted(joy_dictionary.items(), key=lambda item: item[1]))
#From https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value

#Convert dictionary to have MLE estimates by dividing number of times it appears by total number of words
joy_num_words = len(Joy_Sentences_T)
joy_unique_words = len(joy_dictionary.keys())

#Convert Frequency to MLE with Laplace Smoothing
for word in joy_dictionary:
    joy_dictionary[word] = (joy_dictionary[word] + 1)/(joy_num_words+joy_unique_words)

#Convert dictionary to list named mle_words_list
joy_mle_words_list = list(joy_dictionary.items())
joy_mle_words_list.reverse()
###############################################################################################################
#Section 3.2
print("Fear MLE: ", fear_mle_words_list)
print("Anger MLE: ", anger_mle_words_list)
print("Surprise MLE: ", surprise_mle_words_list)
print("Disgust MLE: ", disgust_mle_words_list)
print("Sadness MLE: ", sadness_mle_words_list)
print("Joy MLE: ", joy_mle_words_list)

Sentence = "As she hugged her daughter goodbye on the first day of college, she felt both sad to see her go and joyful knowing that she was embarking on a new and exciting chapter in her life."

Sentence_T = fix_tokenize(Sentence)

Sentence_T = [word for word in Sentence_T if word not in stopwords.words('english')]

###############################################################################################################
fear_prob = pfear#Probability of fear
for word in Sentence_T:
    for i,mle in enumerate(fear_mle_words_list):
        if word == mle[0]: #If word is in fear MLE
            fear_prob = fear_prob * mle[1] #multiply with probability
            break
    else: #Word is not in fear MLE
        fear_prob = fear_prob * (1/(fear_num_words + fear_unique_words))#Multily by 1/(#words in training set + # of vocabulary)
###############################################################################################################
anger_prob = panger
for word in Sentence_T:
    for i,mle in enumerate(anger_mle_words_list):
        if word == mle[0]: #If word is in MLE
            anger_prob = anger_prob * mle[1]
            break
    else:
        anger_prob = anger_prob * (1/(anger_num_words + anger_unique_words))
###############################################################################################################
surprise_prob = psurprise
for word in Sentence_T:
    for i,mle in enumerate(surprise_mle_words_list):
        if word == mle[0]: #If word is in MLE
            surprise_prob = surprise_prob * mle[1]
            break
    else:
        surprise_prob = surprise_prob * (1/(surprise_num_words + surprise_unique_words))
        
###############################################################################################################
disgust_prob = pdisgust
for word in Sentence_T:
    for i,mle in enumerate(disgust_mle_words_list):
        if word == mle[0]: #If word is in MLE
            disgust_prob = disgust_prob * mle[1]
            break
    else:
        disgust_prob = disgust_prob * (1/(disgust_num_words + disgust_unique_words))
        
###############################################################################################################
sadness_prob = psadness
for word in Sentence_T:
    for i,mle in enumerate(sadness_mle_words_list):
        if word == mle[0]: #If word is in MLE
            sadness_prob = sadness_prob * mle[1]
            break
    else:
        sadness_prob = sadness_prob * (1/(sadness_num_words + sadness_unique_words))
        
###############################################################################################################
joy_prob = pjoy
for word in Sentence_T:
    for i,mle in enumerate(joy_mle_words_list):
        if word == mle[0]: #If word is in MLE
            joy_prob = joy_prob * mle[1]
            break
    else:
        joy_prob = joy_prob * (1/(joy_num_words + joy_unique_words))

#Section 3.3
print("Probability the sentence is Fear: ", fear_prob)
print("Probability the sentence is Anger: ", anger_prob)
print("Probability the sentence is Surprise: ", surprise_prob)
print("Probability the sentence is Disgust: ", disgust_prob)
print("Probability the sentence is Sadness: ", sadness_prob)
print("Probability the sentence is Joy: ", joy_prob)

Sadness_Sentences_Testing = df['Sadness Sentences']
Joy_Sentences_Testing = df['Joy Sentences']
Fear_Sentences_Testing = df['Fear Sentences']
Anger_Sentences_Testing = df['Anger Sentences']
Surprise_Sentence_Testing = df['Surprise Sentence']
Disgust_Sentences_Testing = df['Disgust Sentences']
Sadness_Joy_Sentences_Testing = df['Sadness + Joy Sentences']
Fear_Anger_Sentences_Testing = df['Fear + Anger Sentences']
Surprise_Disgust_Sentences_Testing = df['Surprise + Disgust Sentences']
Sadness_Joy_Fear_Sentences_Testing = df['Sadness + Joy + Fear Sentences']

#Tokenize
for i in range(len(Sadness_Sentences_Testing)):
    Sadness_Sentences_T_Testing.append(fix_tokenize(Sadness_Sentences_Testing[i]))
    Joy_Sentences_T_Testing.append(fix_tokenize(Joy_Sentences_Testing[i]))
    Fear_Sentences_T_Testing.append(fix_tokenize(Fear_Sentences_Testing[i]))
    Anger_Sentences_T_Testing.append(fix_tokenize(Anger_Sentences_Testing[i]))
    Surprise_Sentences_T_Testing.append(fix_tokenize(Surprise_Sentence_Testing[i]))
    Disgust_Sentences_T_Testing.append(fix_tokenize(Disgust_Sentences_Testing[i]))
    Sadness_Joy_Sentences_T_Testing.append(fix_tokenize(Sadness_Joy_Sentences_Testing[i]))
    Fear_Anger_Sentences_T_Testing.append(fix_tokenize(Fear_Anger_Sentences_Testing[i]))
    Surprise_Disgust_Sentences_T_Testing.append(fix_tokenize(Surprise_Disgust_Sentences_Testing[i]))
    Sadness_Joy_Fear_Sentences_T_Testing.append(fix_tokenize(Sadness_Joy_Fear_Sentences_Testing[i]))

#Take last 10 elements only
Sadness_Sentences_T_Testing = Sadness_Sentences_T_Testing[-10:]
Joy_Sentences_T_Testing= Joy_Sentences_T_Testing[-10:]
Fear_Sentences_T_Testing = Fear_Sentences_T_Testing[-10:]
Anger_Sentences_T_Testing = Anger_Sentences_T_Testing[-10:]
Surprise_Sentences_T_Testing = Surprise_Sentences_T_Testing[:-10]
Disgust_Sentences_T_Testing = Disgust_Sentences_T_Testing[-10:]
Sadness_Joy_Sentences_T_Testing = Sadness_Joy_Sentences_T_Testing[-10:]
Fear_Anger_Sentences_T_Testing = Fear_Anger_Sentences_T_Testing[-10:]
Surprise_Disgust_Sentences_T_Testing = Surprise_Disgust_Sentences_T_Testing[-10:]
Sadness_Joy_Fear_Sentences_T_Testing = Sadness_Joy_Fear_Sentences_T_Testing[-10:] 

Sadness_Joy_Fear_Sentences_T_Testing.remove([]) #Removes NaN at Sadness_Joy_Fear_Sentences_T_Testing[48]
Surprise_Disgust_Sentences_T_Testing.remove([]) #Removes NaN at Surprise_Disgust_Sentences_T_Testing[48]

#Adds sentences with Multiple Emotions into single emotion categories
Sadness_Sentences_T_Testing += Sadness_Joy_Sentences_T_Testing
Joy_Sentences_T_Testing += Sadness_Joy_Sentences_T_Testing
Fear_Sentences_T_Testing += Fear_Anger_Sentences_T_Testing
Anger_Sentences_T_Testing += Fear_Anger_Sentences_T_Testing
Surprise_Sentences_T_Testing += Surprise_Disgust_Sentences_T_Testing
Disgust_Sentences_T_Testing += Surprise_Disgust_Sentences_T_Testing
Sadness_Sentences_T_Testing += Sadness_Joy_Fear_Sentences_T_Testing
Joy_Sentences_T_Testing += Sadness_Joy_Fear_Sentences_T_Testing
Fear_Sentences_T_Testing += Sadness_Joy_Fear_Sentences_T_Testing

###############################################################################################################
predicted = []
actual = []
for sentence in Sadness_Sentences_T_Testing:
    sentence = [word for word in sentence if word not in stopwords.words('english')] #Remove stopwords
    predicted.append(classifier(sentence))
     
for i in range(len(Sadness_Sentences_T_Testing)):
    actual.append("Sadness")
###############################################################################################################  
for sentence in Joy_Sentences_T_Testing:
    sentence = [word for word in sentence if word not in stopwords.words('english')]
    predicted.append(classifier(sentence))
     
for i in range(len(Joy_Sentences_T_Testing)):
    actual.append("Joy")
###############################################################################################################  
for sentence in Fear_Sentences_T_Testing:
    sentence = [word for word in sentence if word not in stopwords.words('english')]
    predicted.append(classifier(sentence))
     
for i in range(len(Fear_Sentences_T_Testing)):
    actual.append("Fear")
###############################################################################################################  
for sentence in Anger_Sentences_T_Testing:
    sentence = [word for word in sentence if word not in stopwords.words('english')]
    predicted.append(classifier(sentence))
     
for i in range(len(Anger_Sentences_T_Testing)):
    actual.append("Anger")
###############################################################################################################  
for sentence in Surprise_Sentences_T_Testing:
    sentence = [word for word in sentence if word not in stopwords.words('english')]
    predicted.append(classifier(sentence))
     
for i in range(len(Surprise_Sentences_T_Testing)):
    actual.append("Surprise")
###############################################################################################################  
for sentence in Disgust_Sentences_T_Testing:
    sentence = [word for word in sentence if word not in stopwords.words('english')]
    predicted.append(classifier(sentence))
     
for i in range(len(Disgust_Sentences_T_Testing)):
    actual.append("Disgust")
matrix = confusion_matrix(actual,predicted)
#Section 4.1
print("Confusion Matrix: ")
print(matrix)
#Section 4.2
print("Precision, Recall, F1 Score, Accuracy, etc. : ")
print(classification_report(actual,predicted))
