from sqlite3 import dbapi2 as sqlite
import re
import math
import csv
from nltk import FreqDist
from nltk.corpus import stopwords
import nltk
# nltk.download()
import xlrd

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
    count=self.fcount(f,cat)
    if count==0:
      self.con.execute("insert into fc values ('%s','%s',1)" 
                       % (f,cat))
    else:
      self.con.execute(
        "update fc set count=%d where feature='%s' and category='%s'" 
        % (count+1,f,cat)) 
  
  def fcount(self,f,cat):
    res=self.con.execute(
      'select count from fc where feature="%s" and category="%s"'
      %(f,cat)).fetchone()
    if res==None: return 0
    else: return float(res[0])

  def incc(self,cat):
    count=self.catcount(cat)
    if count==0:
      self.con.execute("insert into cc values ('%s',1)" % (cat))
    else:
      self.con.execute("update cc set count=%d where category='%s'" 
                       % (count+1,cat))    

  def catcount(self,cat):
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
    catprob=self.catcount(cat)/self.totalcount()
    docprob=self.docprob(item,cat)
    return docprob*catprob
  
  def setthreshold(self,cat,t):
    self.thresholds[cat]=t
    
  def getthreshold(self,cat):
    if cat not in self.thresholds: return 1.0
    return self.thresholds[cat]
  
  def classify(self,item,default=None):
    probs={}
    # Find the category with the highest probability
    max=0.0
    for cat in self.categories():
      probs[cat]=self.prob(item,cat)
      if probs[cat]>max: 
        max=probs[cat]
        best=cat

    # Make sure the probability exceeds threshold*next best
    for cat in probs:
      if cat==best: continue
      if probs[cat]*self.getthreshold(best)>probs[best]: return default
    return best

class fisherclassifier(classifier):
  def cprob(self,f,cat):
    # The frequency of this feature in this category    
    clf=self.fprob(f,cat)
    if clf==0: return 0

    # The frequency of this feature in all the categories
    freqsum=sum([self.fprob(f,c) for c in self.categories()])

    # The probability is the frequency in this category divided by
    # the overall frequency
    p=clf/(freqsum)
    
    return p
  def fisherprob(self,item,cat):
    # Multiply all the probabilities together
    p=1
    features=self.getfeatures(item)
    for f in features:
      p*=(self.weightedprob(f,cat,self.cprob))

    # Take the natural log and multiply by -2
    fscore=-2*math.log(p)

    # Use the inverse chi2 function to get a probability
    return self.invchi2(fscore,len(features)*2)
  def invchi2(self,chi, df):
    m = chi / 2.0
    sum = term = math.exp(-m)
    for i in range(1, df//2):
        term *= m / i
        sum += term
    return min(sum, 1.0)
  def __init__(self,getfeatures):
    classifier.__init__(self,getfeatures)
    self.minimums={}

  def setminimum(self,cat,min):
    self.minimums[cat]=min
  
  def getminimum(self,cat):
    if cat not in self.minimums: return 0
    return self.minimums[cat]
  def classify(self,item,default=None):
    # Loop through looking for the best result
    best=default
    max=0.0
    for c in self.categories():
      p=self.fisherprob(item,c)
      # Make sure it exceeds its minimum
      if p>self.getminimum(c) and p>max:
        best=c
        max=p
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
  sent=None
  category=None
  if line:
    category = 'postitive' if line[4].value== 1 else 'negative'
    sent = line[0].value.strip()
  return sent,category

# sampletrain

# class Semanticizer
# def semanticize(sent):
#   words=sent.split(" ")
#   
def calculate_probs(dct):
  total = 0.0
  for item in dct.keys():
    total+=dct[item]

  for item in dct.keys():
    dct[item] = dct[item]/total

  return dct
def categorize():
  with open('newdata.csv') as cs:
    reader = csv.DictReader(cs)
    dct={}
    for row in reader:
      dct[row['brandname']]=dct.get(row['brandname'],{})
      for key in row.keys():
        dct[row['brandname']][key] = {}
  with open('newdata.csv') as cs:
    reader = csv.DictReader(cs)
    for row in reader:
      dct[row['brandname']]['features'][row['features']]=dct[row['brandname']]['features'].get(row['features'],0)+1
      dct[row['brandname']]['speakers'][row['speakers']]=dct[row['brandname']]['speakers'].get(row['speakers'],0)+1
      dct[row['brandname']]['processor'][row['processor']]=dct[row['brandname']]['processor'].get(row['processor'],0)+1
      dct[row['brandname']]['application'][row['application']]=dct[row['brandname']]['application'].get(row['application'],0)+1
      dct[row['brandname']]['screen'][row['screen']]=dct[row['brandname']]['screen'].get(row['screen'],0)+1
      dct[row['brandname']]['ram'][row['ram']]=dct[row['brandname']]['ram'].get(row['ram'],0)+1
      # dct[row['brandname']]['sno'][row['sno']]=dct[row['brandname']]['sno'].get(row['sno'],0)+1
      dct[row['brandname']]['classpreference'][row['classpreference']]=dct[row['brandname']]['classpreference'].get(row['classpreference'],0)+1
      dct[row['brandname']]['resolution'][row['resolution']]=dct[row['brandname']]['resolution'].get(row['resolution'],0)+1
      dct[row['brandname']]['hardware'][row['hardware']]=dct[row['brandname']]['hardware'].get(row['hardware'],0)+1
      dct[row['brandname']]['battery'][row['battery']]=dct[row['brandname']]['battery'].get(row['battery'],0)+1
      dct[row['brandname']]['memory'][row['memory']]=dct[row['brandname']]['memory'].get(row['memory'],0)+1
      dct[row['brandname']]['os'][row['os']]=dct[row['brandname']]['os'].get(row['os'],0)+1
    # print dct
    nudct=dct.copy()
    # print nudct
    for key in nudct.keys():
      item = nudct[key]
      # print key
      for i in item.keys():
        # print i,key,nudct[key][i]
        item[i] = calculate_probs(item[i])

    # print nudct
    # return nudct

      # dct[row['brandname']][row['resolution']]=dct[row['brandname']].get(row['resolution'],0)+1
      # dct[row['brandname']][row['camera']]=dct[row['brandname']].get(row['camera'],0)+1
    # print dct
      # print row['os']
      # while row['brandname'] == reader.next()['brandname']:
      #   print row['brandname']
    # print dir(reader)
    # print reader.next()
    # print reader.next()
    # print reader.next()


# categorize()

def sampletrain():
  worksheet = open_worksheet('nusents.xlsx','Sheet1')
  num_rows = worksheet.nrows - 1
  words=[]
  curr_row = -1
  while curr_row < num_rows:
    curr_row += 1
    row = worksheet.row(curr_row)
    print process_line(row)



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

NOKIA = ['lumia', '928', '920', 'windows', '900','microsoft', '925','nokia']
LG = ['optimus', 'android', 'lg']
SONY = ['xperia', 'android', 'z', 'x10', 'z2', 'z1']
SAMSUNG = ['galaxy', 'android', 's4', 's3', 'tablet']
TECNO = ['phantom', 'p3', '1000', 'n3', 'm5']

sampletrain()