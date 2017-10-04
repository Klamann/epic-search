import logging
import os
import uuid

from flask import Flask, render_template, request, send_from_directory
from flask_assets import Environment
from pyhocon import ConfigTree

from search_ui import ClientError, SearchEngine
from search_ui.util import PreferredMime

logger = logging.getLogger('search-ui')

# flask webserver
app = Flask(__name__)
app.secret_key = str(uuid.uuid4())
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True
assets = Environment(app)
assets.debug = False

# search engine
engine = None   # type: SearchEngine


@app.route("/")
def index():
    """the start page, where the user may enter the first query"""
    return render_template('search-startpage.html')


@app.route("/search")
def search():
    """the search interface. Executes the query and returns the search results as HTML or JSON"""
    search_request = engine.parse_search_request(request)
    if not search_request.follow_up:
        q = search_request.query
        author = ', author: "{}"'.format(q.author) if q.author else ""
        date = ', date: {} to {}'.format(q.date_after or "any", q.date_before or "any") \
            if (q.date_after or q.date_before) else ""
        logger.info('search query from {} (session: {}, step: {}, query: "{}"{}{})'
                    .format(request.remote_addr, search_request.sid, q.step, q.query, author, date))
    mime = PreferredMime(request)
    if mime.pref_json:
        return engine.search_json(search_request)
    else:
        return engine.search_html(search_request)


@app.errorhandler(ClientError)
def handle_es_request_error(error):
    """
    handle all sorts of client errors (ClientError is a custom exception that is raised whenever
    the search fails due to client errors)
    """
    return str(error), 400


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


def main(conf: ConfigTree = None):
    global engine
    engine = SearchEngine.from_config(conf)
    config = engine.config
    logger.info("search engine initialized, launching webserver...")
    app.run(debug=config.get_bool('webserver.debug'), host=config.get('webserver.host'),
            port=config.get_int('webserver.port'), threaded=True)
