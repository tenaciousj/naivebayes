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

      # self.num_pos_words = 0
      # self.num_neg_words = 0

      #If the pickled files exist, then load the dictionaries into memory.
      if os.path.exists("positive_best.p"):
         self.positive_words = pickle.load(open("positive_best.p", "rb"))
         self.num_doc_pos = int(pickle.load(open("num_doc_pos_best.txt", "rb")))
         #self.num_pos_words = int(pickle.load(open("num_pos_words_best.txt", "rb")))

      if os.path.exists("negative_best.p"):
         self.negative_words = pickle.load(open("negative_best.p", "rb"))
         self.num_doc_neg = int(pickle.load(open("num_doc_neg_best.txt", "rb")))
         # self.num_neg_words = int(pickle.load(open("num_neg_words_best.txt", "rb")))

      #If the pickled files do not exist, then train the system.
      else:
         self.train()

   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""


      #Gets the names of all the files in the "movies_reviews/" directory
      IFileList =[]
      for fFileObj in os.walk("movies_reviews/"):
         IFileList = fFileObj[2]
         break
      
      #Parse each file
      for review in IFileList:
         review = "movies_reviews/" + review
         loaded_review = self.loadFile(review)
         words = self.bigram_tokenize(loaded_review)
         words = [ex.lower() for ex in words]
         visited_pos = {}
         visited_neg = {}

         #it is a negative review
         #print loaded_review


         #presence, frequency
         if review[22] == "1":
            self.num_doc_neg += 1
            for word in words:
               # self.num_neg_words += 1
               #word = words[i] + " " + words[i+1]
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
            for word in words:#for i in range(len(words)-1):
               # self.num_pos_words += 1
               #word = words[i] + " " + words[i+1]
               if word in self.positive_words:
                  if word not in visited_pos:
                     self.positive_words[word][0] += 1
                     visited_pos[word] = True
                  self.positive_words[word][1] += 1
               else:
                  visited_pos[word] = True
                  self.positive_words[word] = [1,1]

      pickle.dump(self.positive_words, open("positive_best.p", "wb"))
      pickle.dump(self.negative_words, open("negative_best.p", "wb"))
      
      pickle.dump(self.num_doc_pos, open("num_doc_pos_best.txt", "wb"))
      pickle.dump(self.num_doc_neg, open("num_doc_neg_best.txt", "wb"))

      # pickle.dump(self.num_pos_words, open("num_pos_words_best.txt", "wb"))
      # pickle.dump(self.num_neg_words, open("num_neg_words_best.txt", "wb"))


   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """

      # prior_dict_pos, prior_dict_neg, prior_dict_pos_freq, prior_dict_neg_freq = self.calc_cond_prior_prob(sText)
      prior_dict_pos, prior_dict_neg = self.calc_cond_prior_prob(sText)
      pos_class_prior, neg_class_prior = self.class_prior_prob()

      #print prior_dict_pos


      prob_pos_given_text = self.do_bayes(prior_dict_pos, pos_class_prior)
      #print str(prob_pos_given_text) + " pos probability"

      prob_neg_given_text = self.do_bayes(prior_dict_neg, neg_class_prior)
      #print str(prob_neg_given_text) + " neg probability"

      # prob_pos_given_freq = self.do_bayes(prior_dict_pos_freq, pos_class_prior)
      #print str(prob_pos_given_freq) + " pos probability with freq"

      # prob_neg_given_freq = self.do_bayes(prior_dict_neg_freq, neg_class_prior)
      #print str(prob_neg_given_freq) + " neg probability with freq"

      
      if abs(prob_pos_given_text)>abs(prob_neg_given_text):
         return "positive"
      else:
         return "negative"
      '''
      if abs(prob_pos_given_freq) > abs(prob_neg_given_freq):
         return "positive"
      else:
         return "negative"
      '''
         
   def calc_cond_prior_prob(self, sText):
      prior_dict_pos = {}
      prior_dict_neg = {}
      # prior_dict_pos_freq = {}
      # prior_dict_neg_freq = {}

      sText = self.bigram_tokenize(sText)
      sText = [ex.lower() for ex in sText]
      for word in sText:
         if word in self.positive_words:
            presence_pos = self.positive_words[word][0]
            # freq_pos = self.positive_words[word][1]
            prior_dict_pos[word] = (presence_pos+1)/float(self.num_doc_pos)
            # prior_dict_pos_freq[word] = (freq_pos+1)/float(self.num_pos_words)
         
         else:
            prior_dict_pos[word] = 1/float(self.num_doc_pos)
            # prior_dict_pos_freq[word] = 1/float(self.num_pos_words)

         if word in self.negative_words:
            presence_neg = self.negative_words[word][0]
            # freq_neg = self.negative_words[word][1]
            prior_dict_neg[word] = (presence_neg+1)/float(self.num_doc_neg)
            # prior_dict_neg_freq[word] = (freq_neg+1)/float(self.num_neg_words)
         
         else:
            prior_dict_neg[word] = 1/float(self.num_doc_neg)
            # prior_dict_neg_freq[word] = 1/float(self.num_neg_words) 

      #print prior_dict_pos
      # return prior_dict_pos, prior_dict_neg, prior_dict_pos_freq, prior_dict_neg_freq
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
      return math.exp(math.log(class_prior) + prob)
      

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

   def bigram_tokenize(self, sText):
      unigram_tokens = self.tokenize(sText)
      bigram_tokens = []

      for i in range(len(unigram_tokens)-1):
         bigram_tokens.append(unigram_tokens[i] + " " + unigram_tokens[i+1])

      return bigram_tokens
      
   def tokens_to_string(self, tokens):
      new_string = ""
      for t in tokens:
         new_string = new_string + t + " "
      return new_string

   def cross_validation(self):
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

      testing_set=[]
      training_set=[]

      false_pos = 0
      false_neg = 0
      true_pos = 0
      true_neg = 0

      for j in range(10):
         training_set = []
         b = Bayes_Classifier()
         testing_set=list_of_bins[j]

         for bin in list_of_bins:
            if bin!=testing_set:
               training_set.extend(bin)

         counter = 0
         for review in training_set:

            review = "movies_reviews/" + review
            loaded_review = b.loadFile(review)
            tokens_review = b.bigram_tokenize(loaded_review)
            review_text = b.tokens_to_string(tokens_review)

            result = b.classify(review_text)

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
            print str(j) + " in progress: " + str(int(round(100*counter/len(training_set)))) + "%"

      pos_precision = true_pos/float(true_pos + false_pos)
      neg_precision = true_neg/float(true_neg + false_neg) 

      pos_recall = true_pos/float(true_pos + false_neg)
      neg_recall = true_neg/float(true_neg + false_pos)

      precision=(pos_precision+neg_precision)/2.0
      recall=(pos_recall+neg_recall)/2.0
      f1measure=(2*recall*precision)/float(precision+recall)
      


      print str(precision) + " precision"
      print str(recall) + " recall"
      print str(f1measure) + " f1-measure"



b = Bayes_Classifier()
b.cross_validation()
'''
print b.classify("bad horrible awful terrible")
print b.classify("I love my AI class")
print b.classify("I hate my AI class")
#b.classify("horrible")
print b.classify("amazing great awesome phenomenal")
print b.classify("I think that this movie was really well done. It had a lot of interesting plot lines. The acting was very good as well. Despite the director's mistakes, the screenplay was amazing. I absolutely loved this film")
'''