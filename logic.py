# -*- coding: utf-8 -*-

from random import randrange, choice
import pandas as pd
import numpy as np

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

  def truth(self, w):
    """ Evaluate the sentence under interpretation w. """
    if not isinstance(w, Interpretation):
      raise ValueError()
    if self.symbol=='¬' or self.symbol=='~' or self.symbol=='not':
      return not self.child[0].truth(w)
    if self.symbol=='∧' or self.symbol=='&' or self.symbol=='and':
      return self.child[0].truth(w) and self.child[1].truth(w)
    if self.symbol=='∨' or self.symbol=='|' or self.symbol=='or':
      return self.child[0].truth(w) or self.child[1].truth(w)
    if self.symbol=='⊃' or self.symbol=='⇒' or self.symbol=='=>':
      return self.child[0].truth(w) <= self.child[1].truth(w)
    if self.symbol=='≡' or self.symbol=='=' or self.symbol=='iff':
      return self.child[0].truth(w) == self.child[1].truth(w)
    if self.symbol=='⊕' or self.symbol=='≢' or self.symbol=='xor':
      return self.child[0].truth(w) != self.child[1].truth(w)
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
   b = []
   for i in range(len(l)):
     b.append(0)
   carry = 0
   while(carry==0):
     w = Interpretation()
     for i in range(len(l)):
       if b[i]:
         w.assign(l[i])
     yield w
     carry = 1
     for i in range(len(l)):
       b[i] = b[i] + carry
       carry = b[i] // 2
       b[i] = b[i] % 2

def sim(a, b):
  """ Compute the semantic similarity between two sentences """
  signature = a.signature() | b.signature()
  count = 0
  total = 0
  for w in interpretations(signature):
    total = total + 1
    if a.truth(w)==b.truth(w):
      count = count + 1
  return count/total

def random_sentence(signature, level = 0):
  """ Generate a random logical sentence from the given set of ground atoms.
      The generation process is slightly biased towards sentences that are
      neither too simple, nor too complex. """
  if randrange(level + 4)>2:
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
signature = [
  'Acovered', 'AonB', 'AonC', 'AonTbl',
  'Bcovered', 'BonA', 'BonC', 'BonTbl',
  'Ccovered', 'ConA', 'ConB', 'ConTbl'
]

# The following will be the reference interpretation:
i = Interpretation()
# { p, r, t }
#i.assign("p")
#i.assign("r")
#i.assign("t")

#  +---+
#  | B |
#  +---+      +---+
#  | A |      | C |
#--+---+------+---+-----------------
# { AonTbl, ConTbl, BonA, Acovered }
i = Interpretation()
i.assign("AonTbl")
i.assign("ConTbl")
i.assign("BonA")
i.assign("Acovered")

# We can generate as many sentences as we want,
# all evaluated in the reference interpretation:
sentences = []
for k in range(500): # parametrizzare questo che è il numero di casi generati
  s = random_sentence(signature)
  if s in sentences:
    continue
  sentences.append(s)

all =[]
# Print the dataset, in CSV format, on the standard output (updated to save to a file):
header = ["Sentence", "Truth"]
#print("Sentence, Truth", end = ", ")
for s in sentences:
  header.append(str(s))
  #print(s, end = ", ")
#print()
for s in sentences:
  row = [str(s), s.truth(i)]
  #print(s, ", ", s.truth(i), end = ", ")
  for t in sentences:
    row.append(sim(s, t))
    #print(sim(s, t), end = ", ")
  all.append(row)
  #print()
alldf = pd.DataFrame(all, columns = header)
alldf.drop(columns = "Sentence", inplace = True)
alldf.to_csv("kousa.csv", index = False )

