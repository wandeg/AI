from sqlite3 import dbapi2 as sqlite
import re
import math
import csv
from nltk import FreqDist
from nltk.corpus import stopwords
import nltk
# nltk.download()
import xlrd
from brands import *
from featureprobs import *

def getwords(doc):
  # splitter=re.compile('\\W*')
  # print doc

  # Split the words by non-alpha characters
  engstop = stopwords.words('english')
  # print engstop
  words =[]
  if not isinstance(doc,int):
    words=[s.lower() for s in doc.split(" ") if s.lower() not in engstop]
  
  # Return the unique set of words only
  # return dict([(w,1) for w in words])
  return words

class classifier:
  def __init__(self,getfeatures,filename=None):
    # Counts of feature/category combinations
    self.fc={}
    # Counts of documents in each category
    self.cc={}
    self.getfeatures=getfeatures
    
  def setdb(self,dbfile):
    self.con=sqlite.connect(dbfile)    
    self.con.execute('create table if not exists fc(feature,category,count)')
    self.con.execute('create table if not exists cc(category,count)')


  def incf(self,f,cat):
    # Increase number of times feature appears in category
    count=self.fcount(f,cat)
    # print count
    if count==0:
      self.con.execute("insert into fc values ('%s','%s',1)" 
                       % (f,cat))
    else:
      self.con.execute(
        "update fc set count=%d where feature='%s' and category='%s'" 
        % (count+1,f,cat)) 
  
  def fcount(self,f,cat):
    # return how many times feature appears in category
    res=self.con.execute(
      'select count from fc where feature="%s" and category="%s"'
      %(f,cat)).fetchone()
    if res==None: return 0
    else: return float(res[0])

  def incc(self,cat):
    #increase number of items in category
    count=self.catcount(cat)
    # print count
    if count==0:
      self.con.execute("insert into cc values ('%s',1)" % (cat))
    else:
      self.con.execute("update cc set count=%d where category='%s'" 
                       % (count+1,cat))    

  def catcount(self,cat):
    #Returns how many items are in this category
    res=self.con.execute('select count from cc where category="%s"'
                         %(cat)).fetchone()
    if res==None: return 0
    else: return float(res[0])

  def categories(self):
    cur=self.con.execute('select category from cc');
    return [d[0] for d in cur]

  def totalcount(self):
    res=self.con.execute('select sum(count) from cc').fetchone();
    if res==None: return 0
    return res[0]


  def train(self,item,cat):
    features=self.getfeatures(item)
    print features
    # Increment the count for every feature with this category
    for f in features:
      self.incf(f,cat)

    # Increment the count for this category
    self.incc(cat)
    self.con.commit()

  def fprob(self,f,cat):
    if self.catcount(cat)==0: return 0

    # The total number of times this feature appeared in this 
    # category divided by the total number of items in this category
    return self.fcount(f,cat)/self.catcount(cat)

  def weightedprob(self,f,cat,prf,weight=1.0,ap=0.5):
    # Calculate current probability
    basicprob=prf(f,cat)

    # Count the number of times this feature has appeared in
    # all categories
    totals=sum([self.fcount(f,c) for c in self.categories()])

    # Calculate the weighted average
    bp=((weight*ap)+(totals*basicprob))/(weight+totals)
    return bp




class naivebayes(classifier):
  
  def __init__(self,getfeatures):
    classifier.__init__(self,getfeatures)
    self.thresholds={}
  
  def docprob(self,item,cat):
    features=self.getfeatures(item)   

    # Multiply the probabilities of all the features together
    p=1
    for f in features: p*=self.weightedprob(f,cat,self.fprob)
    return p

  def prob(self,item,cat):
    catprob=self.catcount(cat)/self.totalcount() #P(category)
    docprob=self.docprob(item,cat) # P(document|category)
    return docprob*catprob
  
  def setthreshold(self,cat,t):
    self.thresholds[cat]=t
    
  def getthreshold(self,cat):
    if cat not in self.thresholds: return 1.0
    return self.thresholds[cat]
  
  def classify(self,item,default=None):
    probs={}
    # Find the category with the highest probability
    best = None
    highest=0.0
    # print self.categories()
    for cat in self.categories():
      # print cat
      probs[cat]=self.prob(item,cat)
      if probs[cat]>highest: 
        highest=probs[cat]
        print highest
        best=cat

    # Make sure the probability exceeds threshold*next best
    # for cat in probs:
    #   if cat==best: continue
    #   if probs[cat]*self.getthreshold(best)>probs[best]: return default
    return best




# def sampletrain(cl):
#     cl.train('I love this car', 'positive')
#     cl.train('This view is amazing', 'positive')
#     cl.train('I feel great this morning', 'positive')
#     cl.train('I am so excited about the concert', 'positive')
#     cl.train('He is my best friend', 'positive')
#     cl.train('I do not like this car', 'negative')
#     cl.train('This view is horrible', 'negative')
#     cl.train('I feel tired this morning', 'negative')
#     cl.train('I am not looking forward to the concert', 'negative')
#     cl.train('He is my enemy', 'negative')

# def sampletrain(cl):
#   with open('sent_train1.txt','r') as s1:
#     for line in s1.readlines():
#       cl.train(process_line(line))

  

#       # print type(line), len(line), type(line[0])
#       print process_line(line)

def process_line(line):
  # print line, len(line)
  sent=None
  category=None
  if line and len(line) >=5:
    # print line[4].value
    category = 'postitive' if line[4].value== 1 else 'negative'
    sent = line[0].value.strip()
    print sent,category
  return sent,category

# sampletrain

# class Semanticizer
# def semanticize(sent):
#   words=sent.split(" ")
#   



def sampletrain(cl):
  worksheet = open_worksheet('nusents.xlsx','Sheet1')
  num_rows = worksheet.nrows - 1
  words=[]
  curr_row = -1
  while curr_row < num_rows:
    curr_row += 1
    row = worksheet.row(curr_row)
    try:
      a,b = process_line(row)
      # print a,b
      if a and b:
        cl.train(a,b)
    except Exception, e:
      pass
    


def open_worksheet(workbook,worksheet):
  workbook = xlrd.open_workbook(workbook)
  worksheet = workbook.sheet_by_name(worksheet)
  return worksheet

def semanticize(brandname):
  worksheet = open_worksheet('nusents.xlsx','nusents')
  num_rows = worksheet.nrows - 1
  words=[]
  curr_row = -1
  while curr_row < num_rows:
    curr_row += 1
    row = worksheet.row(curr_row)
    # print row[1].value
    if row[1].value == brandname:
      words.extend(getwords(row[0].value))
      # print process_line(row)
  
  fdist = FreqDist(words)
  vocab = fdist.keys()
  print vocab[0:200]
  # print words

# semanticize('nokia')
# semanticize('lg')
# semanticize('sony')
# semanticize('samsung')
# semanticize('tecno')

# NOKIA = ['lumia', '928', '920', 'windows', '900','microsoft', '925','nokia']
# LG = ['optimus', 'android', 'lg']
# SONY = ['xperia', 'android', 'z', 'x10', 'z2', 'z1']
# SAMSUNG = ['galaxy', 'android', 's4', 's3', 'tablet']
# TECNO = ['phantom', 'p3', '1000', 'n3', 'm5']

# sampletrain()

# def summarize(dataset):
#   print zip(*dataset)
#   summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
#   del summaries[-1]
#   return summaries
# dataset = [[1,20,0], [2,21,1], [3,22,0]]
# summary = summarize(dataset)
# print('Attribute summaries: {0}').format(summary)
# # summarize()

def get_similar_words(s1,s2):
  a=set(s1)
  b=set(s2)
  inter = a.intersection(b)
  return list(inter)


def predict_brand(statement):
  highest=0
  brand=None
  for k,v in BRANDS.items():
    a=getwords(statement)
    b=[s.lower() for s in v['values']]
    inter=get_similar_words(a,b)
    if len(inter) > highest:
      highest=len(inter)
      brand=k

  return brand

def predict_sentiment(cl, statement, brand):
  prior_pos = 1
  prior_mod = 1
  prior_neg = 1
  print brand
  mapped = MAPPER[brand]
  print mapped
  brand = FEATPROBS[mapped]
  class_priors = brand["classpreference"]
  print class_priors
  print brand.keys(), statement
  a=brand.keys()
  b=statement.split(" ")
  sim = get_similar_words(a,b)
  print sim
  if len(sim):
    for i in sim:
      if i != 'os':
        print brand[i]

def test(value, expected):
  mapper = {'positive':1, 'negative': -1}
  actual = mapper[value]
  return actual == int(expected)


# from brands import *
# for k,v in BRANDS.items():
#   print k, v['values']
# stm = 'nokia lumia xperia compact ultra 3250 evolve z os ram'
# br = predict_brand(stm)
# predict_sentiment(None,stm, br)
cl = naivebayes(getwords)
cl.setdb('test1.db')
# sampletrain(cl)
# print cl.classify("nokia lumia ")

worksheet = open_worksheet('nusents.xlsx','Sheet1')
num_rows = worksheet.nrows - 1
words=[]
curr_row = 0
num_tru =0
num_false =0
while curr_row < num_rows:
  curr_row += 1
  row = worksheet.row(curr_row)
  try:
    text = row[0].value
    exp = row[4].value
    val = cl.classify(text)
    # print val, text, exp
    pasd = test(val, exp)
    if pasd:
      num_tru +=1
    else:
      num_false +=1
    # break
    # print a,b
    # if a and b:
    #   cl.train(a,b)
    print curr_row
    if curr_row ==20:
      break
  except Exception, e:
    pass

print num_tru, num_false