import webapp2
import views


application = webapp2.WSGIApplication(
	[
    webapp2.Route('/', handler=views.MainPage),
], debug=True)