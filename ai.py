import webapp2
import views
import cronActions


application = webapp2.WSGIApplication(
	[
    webapp2.Route('/', handler=views.MainPage),

    #cron
    webapp2.Route('/cron/fetch_stats/<tag>', handler=cronActions.FetchStats)
], debug=True)