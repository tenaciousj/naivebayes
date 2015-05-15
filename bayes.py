# Name: 
# Date:
# Description:
#
#

import math, os, pickle, re

class Bayes_Classifier:

   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""

      self.positive_words = {}
      self.negative_words = {}

      self.num_doc_pos = 0
      self.num_doc_neg = 0

      #If the pickled files exist, then load the dictionaries into memory.
      if os.path.exists("positive.p"):
         self.positive_words = pickle.load(open("positive.p", "rb"))
      if os.path.exists("negative.p"):
         self.negative_words = pickle.load(open("negative.p", "rb"))

      #If the pickled files do not exist, then train the system.
      else:
         self.train()
      

   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""

      #Gets the names of all the files in the "movies_reviews/" directory
      IFileList =[]
      for fFileObj in os.walk("movies_reviews\\"):
         IFileList = fFileObj[2]
         break
      
      #Parse each file
      for review in IFileList:
         loaded_review = loadFile(review)
         words = tokenize(loaded_review)

         #it is a negative review
         if loaded_review[7] == "1":
            self.self.num_doc_neg += 1
            for word in words:
               if self.negative_words[word]:
                  self.negative_words[word][1] += 1
               else:
                  self.negative_words[word][0] += 1
                  self.negative_words[word][1] = 1

         #otherwise it is a positive review
         elif loaded_review[7] == "5":
            self.self.num_doc_pos += 1
            for word in words:
               if self.positive_words[word]:
                  self.positive_words[word][1] += 1
               else:
                  self.positive_words[word][0] += 1
                  self.positive_words[word][1] = 1

      pickle.dump(self.positive_words, open("positive.p", "wb"))
      pickle.dump(self.negative_words, open("negative.p", "wb"))
    
   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """

      #we might have to load pickle dictionaries
      prior_dict_pos, prior_dict_neg = self.calc_cond_prior_prob(sText)
      pos_class_prior, neg_class_prior = class_prior_prob()

      prob_pos_given_text = do_bayes(prior_dict_pos, pos_class_prior)
      prob_neg_given_text = do_bayes(prior_dict_neg, neg_class_prior)

      threshold = 0.2
      if abs(prob_pos_given_text - prob_neg_given_text) < threshold:
         return "neutral"
      elif prob_pos_given_text>prob_neg_given_text:
         return "positive"
      else:
         return "negative"

   def calc_cond_prior_prob(self, sText):
      #Check to see if +1 smoothing is necessary because of the 0 occurence
      prior_dict_pos = {}
      prior_dict_neg = {}
      for word in sText:
         if self.positive_words[word]:
            pres_freq = self.positive_words[word]
            prior_dict_pos[word] = pres_freq[0]/self.num_doc_pos
         
         else:
            prior_dict_pos[word] = 0

         if self.negative_words[word]:
            pres_freq = self.negative_words[word]
            prior_dict_neg[word] = pres_freq[0]/self.num_doc_neg
         
         else:
            prior_dict_neg[word] = 0

      return prior_dict_pos, prior_dict_neg

   def class_prior_prob(self):
      total_doc = self.num_doc_pos + self.num_doc_neg
      return self.num_doc_pos/total_doc, self.num_doc_neg/total_doc

   def do_bayes(self, prior_dict, class_prior):
      prob = 1
      for key in prior_dict:
         prob *= prior_dict[key]
      return class_prior*prob

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

   #def cross_validation(self, self.num_doc_pos, self.num_doc_neg)   
b = Bayes_Classifier()