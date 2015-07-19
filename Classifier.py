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
from pylab import figure, pie, savefig, axes, title

from flask import Flask, render_template, request, redirect, jsonify, url_for

app = Flask(__name__)
from werkzeug import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt','csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASS_MAPPER = {
'1':'bad',
'2': 'moderate',
'3': 'good'}

to_db = {
'1':'negative',
'2': 'moderate',
'3': 'postitive'}

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
    self.con.execute('create table if not exists brands(name,postitive,negative,moderate)')


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

  def totalvocab(self):
    res=self.con.execute('select sum(count) from fc').fetchone();
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
    # print self.fcount(f,cat),f,cat
    return (self.fcount(f,cat)+1)/(self.catcount(cat)+self.totalvocab())

  def weightedprob(self,f,cat,prf,weight=1.0,ap=0.5):
    # Calculate current probability
    basicprob=prf(f,cat)
    # Count the number of times this feature has appeared in
    # all categories
    totals=sum([self.fcount(f,c) for c in self.categories()])
    return basicprob

  def initialize_brands(self):
    self.con=sqlite.connect(dbfile)
    self.con.execute('create table if not exists brands(name,postitive,negative,moderate)')
    for k,v in MAPPER.items():
      data.append({'brand':k, 'classprefs':to_percent(FEATPROBS[v]['classpreference'])})

    for item in data:
      self.con.execute("insert into brands values (%s,%s,%s,%s)" 
        %(item['brand'],item['classprefs']['3'],item['classprefs']['1'],item['classprefs']['2']))


class naivebayes(classifier):
  
  def __init__(self,getfeatures):
    classifier.__init__(self,getfeatures)
  
  def docprob(self,item,cat):
    features=self.getfeatures(item)   
    # Multiply the probabilities of all the features together
    p=1
    for f in features: p*=self.weightedprob(f,cat,self.fprob)
    return p

  def prob(self,item,cat):
    catprob=(self.catcount(cat)+1)/(self.totalcount()+self.totalvocab()) #P(category)
    docprob=self.docprob(item,cat) # P(document|category)
    # print docprob,catprob
    return docprob*catprob
  
  def classify(self,item,default=None):
    class_probs={}
    # Find the category with the highest probability
    best_class = None
    highest=0.0
    total = 0.0
    temp = {}
    for cat in self.categories():
      class_probs[cat]=self.prob(item,cat)
      total+=class_probs[cat]
      if class_probs[cat]>highest: 
        highest=class_probs[cat]
        # print highest
        best_class=cat
    # print class_probs, best_class
    if '3' in class_probs and 'postitive' in class_probs:
      temp['postitive'] = class_probs['postitive'] + class_probs['3']
    if '1' in class_probs and 'negative' in class_probs:
      temp['negative'] = class_probs['negative'] +class_probs['1']
    if '2' in class_probs and 'neutral' in class_probs:
      temp['neutral'] = class_probs['neutral'] +class_probs['2']
    ideal = ['negative','neutral','postitive']
    for item in ideal:
        if total > 0:
            temp[item] = (temp.get(item,0)+1)/(total+self.totalvocab())
    
    return best_class, temp


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

def sampletrain(cl):
  worksheet = open_worksheet('nusents.xlsx','Sheet1')
  num_rows = worksheet.nrows - 1
  words=[]
  curr_row = -1
  while curr_row < num_rows:
    curr_row += 1
    row = worksheet.row(curr_row)
    # print process_line(row)

    try:
      a,b = process_line(row)
      # print a,b
      if a and b:
        cl.train(a,b)
    except Exception, e:
      pass
    
def sampletrain2(cl):
  with open('full_training_dataset.csv') as f:
    reader = csv.reader(f)
    for row in reader:
      try:
        a = row[1].strip()
        if row[0] == 'positive':
          b = '3'
        elif row[0] == 'negative':
          b = '1'
        elif row [0] == 'neutral':
          b = '2'
        cl.train(a,b)
      except Exception, e:
        pass

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

def rename_keys(dct):
  """
  Returns a dict where all category integers have been renamed 
  to their human readable values
  """
  dc = {}
  for k,v in dct.items():
    if k == 'postitive':
      k='3'
    elif k == 'negative':
      k='1'
    elif k == 'neutral':
      k='2'
    dc[CLASS_MAPPER[k]] = v
  return dc


def predict_sentiment(cl, statement, brand_id):
  brand_name = MAPPER[brand_id]
  total = 1
  brand_data = FEATPROBS[brand_name]
  classed, probs = cl.classify(statement)
  a=brand_data.keys()
  b=statement.split(" ")
  sim = get_similar_words(a,b)
  total = len(sim) 
  mc = None
  used_probs={}
  pr = {}
  for cat in cl.categories():
    pr[cat] = (cl.catcount(cat)+1)/(cl.totalcount()+cl.totalvocab())
  used_probs['priors']=normalize_values(rename_keys(pr.copy()))
  print "using brand"
  print used_probs
  print probs

  # if total >0:
  #   var = [item for item in itertools.combinations_with_replacement('123', total)]
  #   probs=[]
  #   for v in var:
  #     pr=[]
  #     val = 0
  #     ct=1
  #     for i in range(len(sim)):
  #       if sim[i]!= 'os':
  #         ct+=1
  #         val *= brand[sim[i]][v[i]]
  #         ct+=1
  #     val /= ct
  #     # pr.append(val)
  #     probs.append(val)

  #   idx = probs.index(max(probs))
  #   mc = most_common(list(var[idx]))
  #   if probs[idx] > classed[1][high_classed]:
  #     return mc
  #   else:
  #     return high_classed
  # print probs
  if total >0:
    for k,v in probs.items():
      for i in range(len(sim)):
        if sim[i] != 'os':
          if k == 'postitive':
            print probs
            if '3' in probs:
              probs['3']*=probs['postitive']
            else:
              probs['3']=probs['postitive']
            del probs['postitive']
            print probs
            k='3'
            probs[k] *=brand_data[sim[i]][k]
          elif k == 'negative':
            if '1' in probs:
              probs['1']*=probs['negative']
            else:
              probs['1']=probs['negative']
            del probs['negative']
            k='1'
            probs[k] *=brand_data[sim[i]][k]
          elif k == 'neutral':
            if '2' in probs:
              probs['2']*=probs['neutral']
            else:
              probs['2']=probs['neutral']
            del probs['neutral']
            k='2'
            # print probs
          # print brand_data[sim[i]]
            probs[k] *=brand_data[sim[i]][k]
    print probs
    for i in range(len(sim)):
      used_probs[sim[i]] = rename_keys(brand_data[sim[i]])
    tot = sum(probs.values())
    # for k,v in probs.items():
    #   probs[k] = v/tot
    # print probs, 'last'
    used_probs['posteriors'] = normalize_values(rename_keys(probs))
    return max(probs.iteritems(), key=operator.itemgetter(1))[0], used_probs

  else:
    used_probs['posteriors'] = normalize_values(rename_keys(probs))
    return classed, used_probs


def test(value, expected):
  mapper = {'positive':1, 'negative': -1}
  actual = mapper[value]
  return actual == int(expected)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def normalize_values(probs):
  temp = {}
  total = sum(probs.values())
  for k in probs.keys():
    temp[k] = probs[k]/float(total)
  return temp

def get_sent(sent):
  cl = naivebayes(getwords)
  cl.setdb('test1.db')
  # sampletrain(cl)
  # sampletrain2(cl)
  brand = predict_brand(sent)
  if brand:
    cat, probs = predict_sentiment(cl,sent,brand)
    probs['brand'] = brand
  else:
    probs = {}
    cat, p = cl.classify(sent)
    probs['brand'] = 'None'
    pr = {}
    for cat in cl.categories():
      pr[cat] = cl.catcount(cat)/cl.totalcount()
      probs['priors' ]= normalize_values(rename_keys(pr.copy()))
    probs['posteriors'] = normalize_values(rename_keys(p))
  cl.train(sent,cat)
  probs['statement'] = sent
  best = max(probs['posteriors'].iteritems(), key=operator.itemgetter(1))[0]
  probs['class'] = best
  probs['best'] = probs['posteriors'][best]
  return probs

def to_percent(dct):
  for k,v in dct.items():
    dct[k] = v*100
  return dct




def plot(item,i=0, key='classprefs',brand=None):
  figure(i, figsize=(8,8))
  its = sorted(item[key].items(),key=lambda x:x[0])
  print its,'its'
  fracs=[i[1]for i in its]
  print fracs
  mylabels=['good', 'bad', 'moderate']

  mycolors=['green','red','blue']
  pie(fracs,labels=mylabels,colors=mycolors,autopct='%.4f')
  # print [item['brand'],fracs,mylabels,mycolors]
  title(item['brand'])
  savefig(item['brand'])
  # title(brand)
  # savefig(brand)

@app.route('/postsent', methods=['POST'])
def postSentiment():
  
  ans = []
  if request.method == 'POST':
    fil = request.files['file']
    if fil and allowed_file(fil.filename):
        filename = secure_filename(fil.filename)
        fil.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print 
        # print dir(fil.stream)
        # print fil.stream.read()
        # print [f for f in fil.stream.readlines()]
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'r') as f:
          for l in f.readlines():
            a = get_sent(l)
            ans.append(a)

    else:
      sent = request.form.get('sent')
      if sent:
        a = get_sent(sent)
        plot(a,key='posteriors')
        ans.append(a)
    # print ans
    data=[]
    # for k,v in MAPPER.items():
    #   data.append({'brand':k, 'classprefs':to_percent(FEATPROBS[v]['classpreference'])})
    # # print data
    # i=1
    # for item in data:
    #   plot(item,i)
    #   i+=1
      
  return render_template('output.html', output = ans, datas=data)

@app.route('/')
def main():
  return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=50000)