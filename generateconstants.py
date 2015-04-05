import urllib2
from bs4 import BeautifulSoup
import json
import string
import csv


url_dct={'lg':{'url':"http://www.esato.com/phones/by-LG?p=1",'values':[]},
		'sony':{'url':"http://www.esato.com/phones/by-Sony?p=1",'values':[]},
		'samsung':{'url':"http://www.esato.com/phones/by-Samsung?p=1",'values':[]},
		'nokia':{'url':"http://www.esato.com/phones/by-Nokia?p=1",'values':[]},
		'tecno':{'url':"http://bestmobs.com/phones/tecno/page/1/",'values':[]}}

def remove_punctuation(s):
	return s.translate(string.maketrans("",""),string.punctuation)

def calculate_probs(dct):
  total = 0.0
  for item in dct.keys():
    total+=dct[item]

  for item in dct.keys():
    dct[item] = dct[item]/total

  return dct

def get_brands():
# url=URLS[0]
# print url
	for key in url_dct.keys():
		url=url_dct[key]['url']
		current=1
		if key != 'tecno':
			# pass
			url=url[:-1]+str(current)
			# print url
			content = urllib2.urlopen(url).read()
			soup = BeautifulSoup(content)
			ul=soup.find('ul', class_="indexlist")
			# print ul.children
			# print soup.find('ul', class="indexlist")
			if ul:
				while len(ul.contents) >1 and current<10:
					# soup = BeautifulSoup(content)
					# ul=soup.find('ul', class_="indexlist")
					for li in ul.find_all('li'):
					    if li.a is not None:
					        # print li.a.text
					        url_dct[key]['values'].append(li.a.text)
					current+=1
					url=url[:-1]+str(current)
					# print url
					content = urllib2.urlopen(url).read()
					soup = BeautifulSoup(content)
					ul=soup.find('ul', class_="indexlist")
		else:
			url=url[:-2]+str(current)+'/'
			content = urllib2.urlopen(url).read()
			soup = BeautifulSoup(content)
			div = soup.find('div', class_="phone")
			while div is not None:
				if div.ul:
					for li in div.ul.find_all('li'):
						for a in li.find_all('a'):
							if a.text:
								# print a.text
								url_dct[key]['values'].append(a.text)
				current+=1
				url=url[:-2]+str(current)+'/'
				# print url
				content = urllib2.urlopen(url).read()
				soup = BeautifulSoup(content)
				div = soup.find('div', class_="phone")

	return url_dct
#print dir(ul)
# print help(soup.find)

# print get_brands()
def cleanup(url_dct):
	for key in url_dct:
		vals=url_dct[key]['values']
		unpacked=[v for item in vals for v in item.split(" ")]
		# print unpacked
		uniq=list(set(unpacked))
		url_dct[key]['values']=uniq
	with open('brands.py','wb') as b:
		b.write("BRANDS=")
		b.write(json.dumps(url_dct, indent=4))
		# b.write("}")
	return url_dct

# udct = get_brands()
# print cleanup(udct)
# # print remove_punctuation("kdj??*(#")
# from brands import *
# print data['Nokia']

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

    with open('featureprobs.py','wb') as b:
		b.write("FEATPROBS=")
		b.write(json.dumps(nudct, indent=4))

    print nudct
    return nudct

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


categorize()