# Name: 
# Date:
# Description:
#
#

import math, os, pickle, re, random

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
         self.num_doc_pos = int(pickle.load(open("num_doc_pos.txt", "rb")))
      if os.path.exists("negative.p"):
         self.negative_words = pickle.load(open("negative.p", "rb"))
         self.num_doc_neg = int(pickle.load(open("num_doc_neg.txt", "rb")))

      #If the pickled files do not exist, then train the system.
      else:
         self.train()



   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""


      #Gets the names of all the files in the "movies_reviews/" directory
      IFileList =[]
      for fFileObj in os.walk("movies_reviews/"):
         IFileList = fFileObj[2]
      
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
                  self.positive_words[word] = [1,1]

      pickle.dump(self.positive_words, open("positive.p", "wb"))
      pickle.dump(self.negative_words, open("negative.p", "wb"))

      
      pickle.dump(self.num_doc_pos, open("num_doc_pos.txt", "wb"))
      pickle.dump(self.num_doc_neg, open("num_doc_neg.txt", "wb"))


   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """

      prior_dict_pos, prior_dict_neg = self.calc_cond_prior_prob(sText)
      pos_class_prior, neg_class_prior = self.class_prior_prob()

      print prior_dict_pos


      prob_pos_given_text = self.do_bayes(prior_dict_pos, pos_class_prior)
      print str(prob_pos_given_text) + " pos probability"

      prob_neg_given_text = self.do_bayes(prior_dict_neg, neg_class_prior)
      print str(prob_neg_given_text) + " neg probability"

      if prob_pos_given_text>prob_neg_given_text:
         return "positive"
      else:
         return "negative"

   def calc_cond_prior_prob(self, sText):
      #Check to see if +1 smoothing is necessary because of the 0 occurence
      prior_dict_pos = {}
      prior_dict_neg = {}

      sText = self.tokenize(sText)
      sText = [ex.lower() for ex in sText]
      for word in sText:
         if word in self.positive_words:
            presence_pos = self.positive_words[word][0]
            prior_dict_pos[word] = presence_pos/float(self.num_doc_pos) + 1
         
         else:
            prior_dict_pos[word] = 1

         if word in self.negative_words:
            presence_neg = self.negative_words[word][0]
            prior_dict_neg[word] = presence_neg/float(self.num_doc_neg) + 1
         
         else:
            prior_dict_neg[word] = 1

      #print prior_dict_pos
      return prior_dict_pos, prior_dict_neg

   def class_prior_prob(self):
      total_doc = self.num_doc_pos + self.num_doc_neg
      return self.num_doc_pos/float(total_doc), self.num_doc_neg/float(total_doc)

   def do_bayes(self, prior_dict, class_prior):
      """
      prob = 1
      for key in prior_dict:
         prob *= prior_dict[key]
      return class_prior*prob
      """
      prob = 0
      for key in prior_dict:
         prob += math.log(prior_dict[key])
      print str(prob) + "prob"
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

   def cross_validation(self):
      IFileList =[]
      for fFileObj in os.walk("movies_reviews/"):
         IFileList = fFileObj[2]
         break
      
      #shuffling the list in place
      random.shuffle(IFileList)

      data_size=len(IFileList) #number of instances

      #bin size
      bin_size=int(round(data_size/10))

      #list of ten bins. each element is a list. [[bin1]...[bin10]]
      list_of_bins=[IFileList[i:i + 10] for i in range(0, data_size, 10)]

      testing_set=[]
      training_set=[]

      for j in range(10):
         testing_set=list_of_bins[j]

         for bin in list_of_bins:
            if bin!=testing_set:
               training_set.extend(bin)


         for review in training_set:
            review = "movies_reviews\\" + review
            loaded_review = self.loadFile(review)
            words = self.tokenize(loaded_review)

            #it is a negative review
            #print loaded_review
            if review[23] == "1":
               self.num_doc_neg += 1
               for word in words:
                  if word in self.negative_words:
                     self.negative_words[word][1] += 1
                  else:
                     self.negative_words[word][0] += 1
                     self.negative_words[word][1] = 1

            #otherwise it is a positive review
            elif review[23] == "5":
               self.num_doc_pos += 1
               for word in words:
                  if word in self.positive_words:
                     self.positive_words[word][1] += 1
                  else:
                     self.positive_words[word][0] += 1
                     self.positive_words[word][1] = 1

         #now run classification on the testing set 
         #not sure how to save




