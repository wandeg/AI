import webapp2
import logging
from Workers import *
from pydelicious import get_popular,get_userposts,get_urlposts

class FetchStats(webapp2.RequestHandler):
	def get(self,tag):
		# stats=FetchDeliciousStats(tag)
		# logging.info(stats)
		pass


