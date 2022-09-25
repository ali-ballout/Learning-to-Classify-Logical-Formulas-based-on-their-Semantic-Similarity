# -*- coding: utf-8 -*-

from random import randrange, choice
import pandas as pd
import numpy as np
import random
import time
from multiprocessing import Pool


class Interpretation(set):
  """ An interpretation, consisting of a set of ground atoms that are true. """

  def __init__(self):
    """ Initialize an interpretation so that all atoms are false. """
    super()

  def assign(self, symbol, value = True):
    """ Assign a truth value to a symbol. """
    if value:
      self.add(symbol)
    else:
      self.discard(symbol)

  def truth(self, symbol):
    """ Return the truth value of the given symbol in the interpretation. """
    return symbol in self

class Sentence:
  """ A sentence in an arbitrary language.
      A sentence is a syntax tree, whose root contains a symbol,
      which may be a logical connective or an atom.
      The children are, in turn, sentences. """

  def __init__(self, symbol, *subsentences):
    self.parent = None
    self.symbol = symbol
    self.child = subsentences
    for s in self.child:
      s.parent = self

  def __str__(self):
    if len(self.child)==0:
      return self.symbol
    if len(self.child)==1:
      return self.symbol + ' ' + str(self.child[0])
    s = ''
    if self.parent:
      s = s + '('
    s = s + str(self.child[0]) + ' ' + self.symbol + ' ' + str(self.child[1])
    if self.parent:
      s = s + ')'
    return s


  def truth_enhanced(self, w):
    """ Evaluate the sentence under interpretation w. """
    match self.symbol:
       case '¬':
           return not self.child[0].truth_enhanced(w)
       case '∧':
           return self.child[0].truth_enhanced(w) and self.child[1].truth_enhanced(w)
       case '∨':
           return self.child[0].truth_enhanced(w) or self.child[1].truth_enhanced(w)
       # case '⊃':
       #     return self.child[0].truth_enhanced(w) <= self.child[1].truth_enhanced(w)
       # case '≡':
       #     return self.child[0].truth_enhanced(w) == self.child[1].truth_enhanced(w)
       # case '⊕':
       #     return self.child[0].truth_enhanced(w) != self.child[1].truth_enhanced(w)
       # If an exact match is not confirmed, this last case will be used if provided
       case _:
           return w.truth(self.symbol)
    

  def signature(self):
    """ Return the set of the atoms that occur in the formula. """
    if len(self.child)==0:
      return { self.symbol }
    if len(self.child)==1:
      return self.child[0].signature()
    return self.child[0].signature() | self.child[1].signature()

       
def interpretations(signature):
   """ Generate all the interpretations for the given signature.
       The signature can be a set or a list. """
   l = sorted(signature)
   interps = {}
   b = [0 for i in range(len(l))]
   carry = 0
   _id = 0
   while(carry==0):
     w = Interpretation()
     for i in range(len(l)):
       if b[i]:
         w.assign(l[i])
     interps.update({_id:w})
     carry = 1
     _id = _id + 1
     for i in range(len(l)):
       b[i] = b[i] + carry
       carry = b[i] // 2
       b[i] = b[i] % 2
   return list(interps.values())


def sampled_interpretations(signature,k):
   """ Generate all the interpretations for the given signature.
       The signature can be a set or a list. """
   l = sorted(signature)
   interps = []
   for i in range(k):
       w = Interpretation()
       for j in range(len(l)):
           if randrange(0,2) == 1:
               w.assign(l[j])
       interps.append(w)
   return interps
   
    





def random_sentence(signature, level = 0):
  """ Generate a random logical sentence from the given set of ground atoms.
      The generation process is slightly biased towards sentences that are
      neither too simple, nor too complex. """
  if randrange(level + 4)>2:# lower the level gain if you want more complex sentences.. to increase the interpretations and be able to sample them
     # N.B.: the higher the depth, the higher the probability of stopping!
    return Sentence(choice(signature))
  if randrange(3)==0:
    return Sentence('¬', random_sentence(signature, level + 1))
#  return Sentence(choice(['∧', '∨', '⊃', '≡', '⊕']),
  return Sentence(choice(['∧', '∨']),
    random_sentence(signature, level + 1), random_sentence(signature, level + 1))

#### Generate random formulas with their truth value

# This is the signature of the language or its Herbrand universe:
# signature = ['p', 'q', 'r', 's', 't']
# Inspired by the block world


def sim_enhanced(a, b, n):
  """ Compute the semantic similarity between two sentences """
  #tic = time.perf_counter()
  signature = a.signature() | b.signature()
  if n == 3:
        interp = interpretations(signature)
  else:
        samplesize = [30,100,1000]
        interp = sampled_interpretations(signature, k=samplesize[n] )
        #interp = random.choices(base_interpretations, k=samplesize[n])
  #toc = time.perf_counter()
  #print(f"interpritation and sampling took {toc- tic:0.5f} seconds")
  #tic = time.perf_counter()
  truths = np.array([row[0] == row[1] for row in np.array([[sent.truth_enhanced(w) for sent in [a, b]] for w in interp])])
  #toc = time.perf_counter()
  #print(f"truth comparison took {toc- tic:0.4f} seconds")
  return sum(truths)/len(interp)
  
    
# for z in range(4):
#     print("started " , z)
#     all ={}
#     # Print the dataset, in CSV format, on the standard output (updated to save to a file):
#     row = [s.truth_enhanced(i) for s in sentences]
#     all.update({"truth":row})
#     print("starting sim")
#     count = 0
#     for s in sentences:
#       print(count," ",str(s))
#       row = np.array([sim_enhanced(s, t, z) for t in sentences[count:]])
#       all.update({str(s):row})
#       count = count + 1
#       row = None
#     alldf = pd.DataFrame(all)
#     print("saving " , z)
#     alldf.to_csv("kousa"+ f'{z}'+ ".csv", index = False )


def construct_matrix(n, sentences,i):
    print("started " , n)
    truth=[s.truth_enhanced(i) for s in sentences]
    header = [str(s) for s in sentences]
    print("starting sim")
    count = 0
    all=[]
    for s in sentences:
      print(n," ",count," ",str(s))
      row = None
      row = np.array([sim_enhanced(s, t, n) for t in sentences])
      all.append(row)
      count = count + 1
    alldf = pd.DataFrame(all, columns = header)
    alldf.insert(0,'truth', truth)
    print("saving " , n)
    alldf.to_csv("kousa"+ f'{n}'+ ".csv", index = False )


#################################################################
# signature = [
#   'Acovered', 'AonB', 'AonC','AonD' ,'AonTbl',
#   'Bcovered', 'BonA', 'BonC','BonD' ,'BonTbl',
#   'Ccovered', 'ConA', 'ConB','ConD' ,'ConTbl',
#   'Dcovered', 'DonA', 'DonB','DonC' ,'DonTbl'
# ]

# # The following will be the reference interpretation:
# i = Interpretation()
# # { p, r, t }
# #i.assign("p")
# #i.assign("r")
# #i.assign("t")

# #  +---+      +---+
# #  | B |      | D |
# #  +---+      +---+       
# #  | A |      | C |       
# #--+---+------+---+----------------
# # { AonTbl, ConTbl, BonA, Acovered }
# i = Interpretation()
# i.assign("AonTbl")
# i.assign("ConTbl")
# i.assign("BonA")
# i.assign("DonC")
# i.assign("Acovered")
# i.assign("Ccovered")
###################################################################
signature = [
  'Acovered', 'AonB', 'AonC','AonD','AonE' ,'AonTbl',
  'Bcovered', 'BonA', 'BonC','BonD','BonE' ,'BonTbl',
  'Ccovered', 'ConA', 'ConB','ConD','ConE' ,'ConTbl',
  'Dcovered', 'DonA', 'DonB','DonC','DonE' ,'DonTbl',
  'Ecovered', 'EonA', 'EonB','EonC','EonD' ,'EonTbl'
]

# The following will be the reference interpretation:
i = Interpretation()
# { p, r, t }
#i.assign("p")
#i.assign("r")
#i.assign("t")

#  +---+      +---+
#  | B |      | D |
#  +---+      +---+       +---+
#  | A |      | C |       | E |
#--+---+------+---+-------+---+-----
# { AonTbl, ConTbl, BonA, Acovered }
i = Interpretation()
i.assign("AonTbl")
i.assign("ConTbl")
i.assign("EonTbl")
i.assign("BonA")
i.assign("DonC")
i.assign("Acovered")
i.assign("Ccovered")
###################################################################
# signature = [
#   'Acovered', 'AonB', 'AonC', 'AonTbl',
#   'Bcovered', 'BonA', 'BonC', 'BonTbl',
#   'Ccovered', 'ConA', 'ConB', 'ConTbl'
# ]

# # The following will be the reference interpretation:
# i = Interpretation()
# # { p, r, t }
# #i.assign("p")
# #i.assign("r")
# #i.assign("t")

# #  +---+
# #  | B |
# #  +---+      +---+
# #  | A |      | C |
# #--+---+------+---+-----------------
# # { AonTbl, ConTbl, BonA, Acovered }
# i = Interpretation()
# i.assign("AonTbl")
# i.assign("ConTbl")
# i.assign("BonA")
# i.assign("Acovered")
###################################################################



# We can generate as many sentences as we want,
# all evaluated in the reference interpretation:

# sentences = []
# for k in range(500): # parametrizzare questo che è il numero di casi generati
#   s = random_sentence(signature)
#   if s in sentences:
#     continue
#   sentences.append(s)
# sentences = np.array(sentences)

def check_duplicate(s, sentences):
    for sentence in sentences:
        if str(s)== str(sentence):
            return True
    return False




sentences = []
k = 0
lit = 0 
while k < 500:
  k = k+1# parametrizzare questo che è il numero di casi generati
  s = random_sentence(signature)
  if check_duplicate(s, sentences):
    print(".")
    k = k - 1
    continue
  if len(s.child)==0:
      if lit <= 6:
          lit = lit + 1
      else:  
          k = k - 1
          continue
  sentences.append(s)  

for n in range(1):
    print("started " , n)
    truth=[s.truth_enhanced(i) for s in sentences]
    header = [str(s) for s in sentences]
    print("starting sim")
    count = 0
    all=[]
    for s in sentences:
      print(n," ",count," ",str(s))
      row = None
      row = np.array([sim_enhanced(s, t, n) for t in sentences])
      all.append(row)
      count = count + 1
    alldf = pd.DataFrame(all, columns = header)
    alldf.insert(0,'truth', truth)
    print("saving " , n)
    alldf.to_csv("30_500_1000"+ f'{n}'+ ".csv", index = False, encoding = "UTF-16" )






