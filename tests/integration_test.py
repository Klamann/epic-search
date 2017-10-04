import os
import unittest
from threading import Thread
from urllib.parse import urlparse, parse_qs

import requests
from lxml.etree import _Element
from lxml.html import html5parser
from pyhocon import ConfigFactory, ConfigTree

from search_ui import app


class SearchUiTest(unittest.TestCase):

    conf = ConfigFactory.parse_file("integration_test.conf")    # type: ConfigTree
    host, port = conf.get("webserver.host"), conf.get_int("webserver.port")
    es_host, es_port = conf.get("elastic.host"), conf.get_int("elastic.port")
    file_topics = conf.get("data.topics.file")
    app_url = "http://{}:{}/".format(host, port)
    es_url = "http://{}:{}/".format(es_host, es_port)

    def setUp(self):
        # check that files are accessible and elasticsearch is online
        if not os.path.isfile(self.file_topics):
            self.fail("cannot read topics from " + self.file_topics)
        try:
            r = requests.get(self.es_url)
            assert r.ok
        except requests.ConnectionError:
            self.fail("elasticsearch is not reachable")
            # start the service
        # start the service
        t = Thread(target=app.main, args=[self.conf], daemon=True)
        t.start()

    def tearDown(self):
        pass

    def test_hello(self):
        r = requests.get(self.app_url)
        assert r.ok
        assert 'id="search-heading"' in r.text

    def test_query(self):
        query = "test"
        r = requests.get(self.app_url + "search", params={'q': query})
        # check response & redirects
        assert r.ok, "problematic status code: " + r.status_code
        assert r.history[-1].status_code == 302, "you were not redirected"
        # check params
        params = parse_qs(urlparse(r.url).query)
        assert 'sid' in params, "no session-id was given!"
        assert 'step' in params, "the 'step' parameter is missing!"
        assert 'q' in params, "the query parameter is missing!"
        assert params['q'][0] == query
        # check contents
        html = html5parser.fromstring(r.content)    # type: _Element
        assert html.cssselect("#result-list")
        assert len(html.cssselect("#result-list .result-entry")) == 10
        assert html.cssselect("#search-input")[0].attrib['value'] == query
        assert len(html.cssselect("#topic-centroid-list .topic-item")) > 5
        assert len(html.cssselect("#suggestions-list .result-entry")) == 10
        js = list(html.iter('{*}script'))[0].text
        assert "const query" in js
        assert "const topicGraph" in js

    def test_empty_result(self):
        query = "togetanansweryoufirsthavetoknowthequestion"
        r = requests.get(self.app_url + "search", params={'q': query})
        # check response & redirects
        assert r.ok, "problematic status code: " + r.status_code
        # check contents
        html = html5parser.fromstring(r.content)    # type: _Element
        assert html.cssselect("#result-failure")
        assert html.cssselect("#result-sidebar")
