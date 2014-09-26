import webapp2
import views
import logging
from google.appengine.api import users
from google.appengine.ext import ndb

import jinja2
import os
import urllib


JINJA_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
    extensions=['jinja2.ext.autoescape'],
    autoescape=True)
logging.info(JINJA_ENVIRONMENT)

application = webapp2.WSGIApplication(
	[
    webapp2.Route('/', handler=views.MainPage),
], debug=True)