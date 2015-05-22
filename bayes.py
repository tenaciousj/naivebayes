# Name: Jeanette Pranin (jrp338), Nishant Subramani (nso155), Jaiveer Kothari (jvk383)
# Date: 05/20/2015
# Description: Naive Bayes Classifier
#
#

import math, os, pickle, re, random

class Bayes_Classifier:

   def __init__(self, pos_doc_file = "positive.p",
                     neg_doc_file = "negative.p",
                     num_doc_pos_file = "num_doc_pos.txt",
                     num_doc_neg_file = "num_doc_neg.txt",
                     j = -1):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""

      self.positive_words = {}
      self.negative_words = {}

      #for presence
      self.num_doc_pos = 0
      self.num_doc_neg = 0

      #If the pickled files exist, then load the dictionaries into memory.
      if os.path.exists(pos_doc_file):
         self.positive_words = pickle.load(open(pos_doc_file, "rb"))
      if os.path.exists(num_doc_pos_file):
         self.num_doc_pos = int(pickle.load(open(num_doc_pos_file, "rb")))

      if os.path.exists(neg_doc_file):
         self.negative_words = pickle.load(open(neg_doc_file, "rb"))
      if os.path.exists(num_doc_neg_file):
         self.num_doc_neg = int(pickle.load(open(num_doc_neg_file, "rb")))

      #If the pickled files do not exist, then train the system.
      else:
         self.train(pos_doc_file, neg_doc_file, num_doc_pos_file, num_doc_neg_file, j)

   def train(self, pos_doc_file, neg_doc_file, num_doc_pos_file, num_doc_neg_file, j):   
      """Trains the Naive Bayes Sentiment Classifier."""

      #Gets the names of all the files in the "movies_reviews/" directory
      IFileList =[]
      for fFileObj in os.walk("movies_reviews/"):
         IFileList = fFileObj[2]
         break
      
      if j != -1:
         #shuffle the list
         random.seed(0)
         random.shuffle(IFileList)

         data_size=len(IFileList) #number of instances

         #bin size
         bin_size=int(round(data_size/10))

         #list of ten bins. each element is a list. [[bin1]...[bin10]]
         list_of_bins=[IFileList[i:i + bin_size] for i in range(0, data_size, bin_size)]

         training_set = []
         for bin in list_of_bins:
            if bin != list_of_bins[j]:
               training_set.extend(bin)

         IFileList = training_set


      #Parse each file
      for review in IFileList:
         review = "movies_reviews/" + review
         loaded_review = self.loadFile(review)
         words = self.tokenize(loaded_review)
         words = [ex.lower() for ex in words]
         visited_pos = {}
         visited_neg = {}

         #it is a negative review
         #print loaded_review


         #presence, frequency
         if review[22] == "1":
            self.num_doc_neg += 1
            for word in words:
               if word in self.negative_words:
                  if word not in visited_neg:
                     self.negative_words[word][0] += 1
                     visited_neg[word] = True
                  self.negative_words[word][1] += 1
               else:
                  visited_neg[word] = True
                  self.negative_words[word] = [1,1]

         #otherwise it is a positive review
         elif review[22] == "5":
            self.num_doc_pos += 1
            for word in words:
               if word in self.positive_words:
                  if word not in visited_pos:
                     self.positive_words[word][0] += 1
                     visited_pos[word] = True
                  self.positive_words[word][1] += 1
               else:
                  visited_pos[word] = True
                  self.positive_words[word] = [1,1]


      if j == -1:
         pos_file_name = "positive.p"
         neg_file_name = "negative.p"
         pos_doc_file = "num_doc_pos.txt"
         neg_doc_file = "num_doc_neg.txt"
      else:
         pos_file_name = "positive" + str(j) + ".p"
         neg_file_name = "negative" + str(j) + ".p"
         pos_doc_file = "num_doc_pos" + str(j) + ".txt"
         neg_doc_file = "num_doc_neg" + str(j) + ".txt"

      pickle.dump(self.positive_words, open(pos_file_name, "wb"))
      pickle.dump(self.negative_words, open(neg_file_name, "wb"))
      
      pickle.dump(self.num_doc_pos, open(pos_doc_file, "wb"))
      pickle.dump(self.num_doc_neg, open(neg_doc_file, "wb"))


   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """

      prior_dict_pos, prior_dict_neg = self.calc_cond_prior_prob(sText)
      pos_class_prior, neg_class_prior = self.class_prior_prob()

      prob_pos_given_text = self.do_bayes(prior_dict_pos, pos_class_prior)
      prob_neg_given_text = self.do_bayes(prior_dict_neg, neg_class_prior)

      if abs(prob_pos_given_text) < abs(prob_neg_given_text):
         return "positive"
      else:
         return "negative"
         
   def calc_cond_prior_prob(self, sText):
      """Calculates the prior probability of each element in the positive and negative word dicitonaries
         Returns two dictionaries, positive and negative, with each key being the word and its value being
         its prior probability
      """
      prior_dict_pos = {}
      prior_dict_neg = {}

      sText = self.tokenize(sText)

      #to account for the same words but different cases (upper vs lower)
      sText = [ex.lower() for ex in sText]

      for word in sText:
         if word in self.positive_words:
            presence_pos = self.positive_words[word][0]
            
            #add-1 smoothing
            prior_dict_pos[word] = (presence_pos+1)/float(self.num_doc_pos)
         
         else:
            prior_dict_pos[word] = 1/float(self.num_doc_pos)

         if word in self.negative_words:
            presence_neg = self.negative_words[word][0]
            prior_dict_neg[word] = (presence_neg+1)/float(self.num_doc_neg)
         
         else:
            prior_dict_neg[word] = 1/float(self.num_doc_neg)

      return prior_dict_pos, prior_dict_neg

   def class_prior_prob(self):
      """Calculates the class prior probabilities
      """
      total_doc = self.num_doc_pos + self.num_doc_neg
      return self.num_doc_pos/float(total_doc), self.num_doc_neg/float(total_doc)

   def do_bayes(self, prior_dict, class_prior):
      """Performs the Bayes equation
      """
      prob = 0
      for key in prior_dict:
         prob += math.log(prior_dict[key])
      return math.log(class_prior) + prob
      

   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""

      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens

      
   def tokens_to_string(self, tokens):
      new_string = ""
      for t in tokens:
         new_string = new_string + t + " "
      return new_string

def cross_validation():
   """Performs 10-fold cross validation on the naive bayes classifier
   """
   pos_fmeasure = []
   neg_fmeasure = []

   pos_precision = []
   neg_precision = []

   pos_recall = []
   neg_recall = []

   IFileList =[]
   for fFileObj in os.walk("movies_reviews/"):
      IFileList = fFileObj[2]
      break
   #shuffling the list in place
   random.seed(0)
   random.shuffle(IFileList)

   data_size=len(IFileList) #number of instances

   #bin size
   bin_size=int(round(data_size/10))

   #list of ten bins. each element is a list. [[bin1]...[bin10]]
   list_of_bins=[IFileList[i:i + bin_size] for i in range(0, data_size, bin_size)]

   for j in range(10):


      pos_file_name = "positive" + str(j) + ".p"
      neg_file_name = "negative" + str(j) + ".p"
      pos_doc_file = "num_doc_pos" + str(j) + ".txt"
      neg_doc_file = "num_doc_neg" + str(j) + ".txt"
      bc = Bayes_Classifier(pos_file_name, neg_file_name, pos_doc_file, neg_doc_file, j)
      
      
      IFileList = list_of_bins[j]


      false_pos = 0
      false_neg = 0
      true_pos = 0
      true_neg = 0

      counter = 0
      for review in IFileList:

         review = "movies_reviews/" + review
         loaded_review = bc.loadFile(review)
         tokens_review = bc.tokenize(loaded_review)
         review_text = bc.tokens_to_string(tokens_review)

         result = bc.classify(review_text)
         rating = review[22]

         #it is a negative review
         if rating == "1":
            if result == "negative":
               true_neg += 1
            elif result == "positive":
               false_pos += 1
         elif rating == "5":
            if result == "negative":
               false_neg += 1
            elif result == "positive":
               true_pos += 1
         counter += 1
         print str(j) + " in progress: " + str(int(round(100*counter/len(IFileList)))) + "%"

      pos_precision.append(true_pos/float(true_pos+false_pos))
      pos_recall.append(true_pos/float(true_pos+false_neg))
      pos_fmeasure.append((2*pos_precision[j]*pos_recall[j])/float(pos_precision[j]+pos_recall[j]))

      neg_precision.append(true_neg/float(true_neg+false_neg))
      neg_recall.append(true_neg/float(true_neg+false_pos))
      neg_fmeasure.append((2*neg_precision[j]*neg_recall[j])/float(neg_precision[j]+neg_recall[j]))


   avg_pos_fmeasure = sum(pos_fmeasure)/len(pos_fmeasure)
   avg_neg_fmeasure = sum(neg_fmeasure)/len(neg_fmeasure)

   avg_pos_precision = sum(pos_precision)/len(pos_precision)
   avg_neg_precision = sum(neg_precision)/len(neg_precision)

   avg_pos_recall = sum(pos_recall)/len(pos_recall)
   avg_neg_recall = sum(neg_recall)/len(neg_recall)
   
   print str(avg_pos_fmeasure) + " pos_f1-measure"
   print str(avg_neg_fmeasure) + " neg_f1-measure"
   print str(avg_pos_precision) + " avg pos precision"
   print str(avg_neg_precision) + " avg neg precision"
   print str(avg_pos_recall) + " avg pos recall"
   print str(avg_neg_recall) + " avg neg recall"

cross_validation()