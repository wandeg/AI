import webapp2
import logging
import time
#import pyinclude
from pydelicious import get_popular,get_userposts,get_urlposts

def FetchDeliciousStats(tag):
	def initializeUserDict(tag,count):
		user_dict={}
		# get the top count' popular posts
		for p1 in get_popular(tag=tag)[0:count]:
		# find all users who posted this
			#logging.info(get_popular(tag=tag)[0:count])
			for p2 in get_urlposts(p1['url']):
				user=p2['user']
				user_dict[user]={}
		return user_dict


	def fillItems(user_dict):
		all_items={}
		# Find links posted by all users
		for user in user_dict:
			for i in range(3):
				try:
					posts=get_userposts(user)
					break
				except:
					print "Failed user "+user+", retrying"
					time.sleep(4)
		for post in posts:
			url=post['url']
			#print user_dict[user]
			user_dict[user]=1.0
			all_items[url]=1
		# Fill in missing items with 0
		#print (type(set(user_dict.values())),type(set(all_items.values())))
		#print json.dumps(user_dict,indent=1)
		#print json.dumps(all_items,indent=1)
		not_in=set.difference(set(user_dict.keys()),set(all_items.keys()))
		#logging.info(not_in)
		# for ratings in user_dict.values( ):
		# 	for item in all_items:
		# 		if item not in ratings:
		# 			ratings[item]=0.0
		#print not_in
		for item in not_in:
			user_dict[item]=0.0
		return user_dict



	return fillItems(initializeUserDict(tag,5))