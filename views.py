import webapp2
import logging
import ai
import jinja2
import os
# from ai import JINJA_ENVIRONMENT
JINJA_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
    extensions=['jinja2.ext.autoescape'],
    autoescape=True)
logging.info(dir(ai))
class MainPage(webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/plain'
        # self.response.write('Hello, World! Please bear with me for a little while I get this up and running')
        template = JINJA_ENVIRONMENT.get_template('index.html')
        logging.info(dir(template))
        self.response.write(template)