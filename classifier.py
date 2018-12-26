from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import sys

train = [
    ["John had a twisted ankle so I asked him to wait", "low-priority"],
    ["Woman had acne, told her to wait as it wasn't priority", "low-priority"],
    ["The client could not explain the problem", "low-priority"],
    ["A patient called to ask about information", "low-priority"],
    ["The woman who called said she had a slight headache last night", "low-priority"],
    ["Clients were not sure if they still had the symptoms", "low-priority"],
    ["My patient thought she was sick, she was not", "low-priority"],
    ["There was this woman who scratched her arm and wanted to come in", "low-priority"],
    ["A few bruises on their legs, nothing to be worried about", "low-priority"],
    ["Somehow she came in without an appointment", "low-priority"],
    ["Explained the patient that how he feels is not important right now", "low-priority"],

    ["Told patient to buy Aspirin for headache", "prescription"],
    ["Patient was dying, gave him paracetamol", "prescription"],
    ["Treated severe cold with extra dose of C vitamin", "prescription"],
    ["Ibuprofen was prescribed to the patient with some pains", "prescription"],
    ["A 0.5ml injection of epinephrine was administered to the upper right leg", "prescription"],
    ["Wrote a prescription for multiple drugs, including nafazol, caffetin and andol", "prescription"],
    ["Prescribed 2000mg of Diasepam for suicide, twice a day", "prescription"],
    ["Client came in with skin rashes, gave him a bottle of rakija", "prescription"],
    ["Stomach problems to be solved with Ranitidin", "prescription"],
    ["Janice has to take 2 pills of Paracetamol every morning", "prescription"],
]

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = stopwords.words('english')

for i in range(len(train)):
    train[i][0] = word_tokenize(train[i][0])
    train[i][0] = [word for word in train[i][0] if word.isalpha()]
    train[i][0] = [w for w in train[i][0] if not w in stop_words]

print(train[:100])

cl = NaiveBayesClassifier(train)

text = [
    ("Alek was told to wait in the pong room until further notice"),
    ("I gave a bottle of rakija to Stano"),
    ("Everybody needs more drugs"),
    ("The patient was instructed to leave the premises"),
    ("There was no reason for this appointment"),
    ("2mg pills for bone strength, calcium"),
    ("Paracetamol is to be taken twice a day"),
    ("Dance took a pill in Ibiza")
]

print("============================================================")

for sentence in text:
    print(sentence)
    print("Classification: " + str(cl.classify(sentence)) + '\n')