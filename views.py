import webapp2
from ai import *

class MainPage(webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/plain'
        # self.response.write('Hello, World! Please bear with me for a little while I get this up and running')
        template = JINJA_ENVIRONMENT.get_template('index.html')
        self.response.write(template.render(template_values))