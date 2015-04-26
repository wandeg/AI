import itertools
from sqlite3 import dbapi2 as sqlite
import re
import os
import math
import csv
from nltk import FreqDist
from nltk.corpus import stopwords
import nltk
import operator
# nltk.download()
import xlrd
from brands import *
from featureprobs import *

from flask import Flask, render_template, request, redirect, jsonify, url_for

app = Flask(__name__)
from werkzeug import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt','csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    total = 0.0
    # print self.categories()
    for cat in self.categories():
      # print cat
      probs[cat]=self.prob(item,cat)
      total+=probs[cat]
      if probs[cat]>highest: 
        highest=probs[cat]
        print highest
        best=cat

    ideal = ['1','2','3']

    for item in ideal:
        if total > 0:
            probs[item] = (probs.get(item,0)+1)/(total+len(ideal))
    return best, probs


def process_line(line):
  # print line, len(line)
  sent=None
  category=None
  if line and len(line) >=5:
    category = '3' if line[4].value== 1 else '1'
    sent = line[0].value.strip()
  return sent,category


def open_worksheet(workbook,worksheet):
  workbook = xlrd.open_workbook(workbook)
  worksheet = workbook.sheet_by_name(worksheet)
  return worksheet

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

def most_common(lst):
    return max(set(lst), key=lst.count)

def predict_sentiment(cl, statement, brand):
  mapped = MAPPER[brand]
  total = 1
  brand = FEATPROBS[mapped]
  class_priors = brand["classpreference"]
  classed = cl.classify(statement)
  a=brand.keys()
  b=statement.split(" ")
  sim = get_similar_words(a,b)
  total = len(sim) 
  mc = None
  high_prior = max(class_priors.iteritems(), key=operator.itemgetter(1))[0]
  high_classed = classed[0]
  if total >0:
    var = [item for item in itertools.combinations_with_replacement('123', total)]
    probs=[]
    for v in var:
      pr=[]
      val = 0
      ct=1
      for i in range(len(sim)):
        if sim[i]!= 'os':
          ct+=1
          val += brand[sim[i]][v[i]]
          val+=class_priors[v[i]]
          ct+=1
      val /= ct
      # pr.append(val)
      probs.append(val)

    idx = probs.index(max(probs))
    mc = most_common(list(var[idx]))
    if probs[idx] > classed[1][high_classed]:
      return mc
    else:
      return high_classed
  else:
    return high_classed


def test(value, expected):
  mapper = {'positive':1, 'negative': -1}
  actual = mapper[value]
  return actual == int(expected)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def get_sent(sent):
  cl = naivebayes(getwords)
  cl.setdb('test1.db')
  brand = predict_brand(sent)
  if brand:
    val = predict_sentiment(cl,sent,brand)
  else:
    val = cl.classify(sent)[0]
  return "The classified value for  %s is %s" %(sent,CLASS_MAPPER[val])


@app.route('/postsent', methods=['POST'])
def postSentiment():
  
  ans = []
  if request.method == 'POST':
    fil = request.files['file']
    print dir(fil), type(fil)
    if fil and allowed_file(fil.filename):
        filename = secure_filename(fil.filename)
        fil.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print 
        # print dir(fil.stream)
        # print fil.stream.read()
        # print [f for f in fil.stream.readlines()]
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'r') as f:
          for l in f.readlines():
            ans.append(get_sent(l))

    else:
      sent = request.form.get('sent')
      if sent:
        ans.append(get_sent(sent))
    print ans
  return render_template('output.html', output = ans)

@app.route('/')
def main():
  return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)