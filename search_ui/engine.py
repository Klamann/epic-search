import logging
import os
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Set, DefaultDict, Optional

import elasticsearch
import urllib3
from flask import jsonify, request, url_for, render_template, flash, Request
from markupsafe import Markup
from pyhocon import ConfigTree, ConfigFactory
from werkzeug.datastructures import MultiDict
from werkzeug.utils import redirect

from search_ui import SessionManager, ClientError, SearchRequest, EsBridge, Query, \
    TopicModel, Topic, SearchResult, SearchHit, PersistentSession, SpellChecker, \
    Spelling, util

logger = logging.getLogger('search-ui')


class SearchEngine:

    def __init__(self, config: ConfigTree, es: EsBridge = None, manager: SessionManager = None,
                 topic_model: TopicModel = None):
        super().__init__()
        self.config = config
        self.es = es
        self.manager = manager
        self.topic_model = topic_model
        self.spell_checker = None   # type: SpellChecker
        # parsed config
        self.result_limit = self.config.get_int('search.result_limit')
        self.spellcheck_enabled = self.config.get_bool('search.spellchecker.enabled', False)
        # initialize stuff based on config
        if self.spellcheck_enabled:
            self.spell_checker = SpellChecker()

    @classmethod
    def from_config(cls, config: ConfigTree = None) -> 'SearchEngine':
        # merge config
        base_conf = cls.load_default_config()
        config = config.with_fallback(base_conf) if config else base_conf
        # logging
        cls.logging_setup('search-ui', config, set_default=True)
        # initialize
        es = EsBridge(config)
        topic_model = cls.load_topic_model(config)
        manager = SessionManager(topic_model=topic_model)
        return SearchEngine(config, es=es, manager=manager, topic_model=topic_model)

    @staticmethod
    def load_default_config() -> ConfigTree:
        config_string = util.read_resource('base.conf')
        return ConfigFactory.parse_string(config_string)

    @classmethod
    def load_topic_model(cls, config: ConfigTree) -> TopicModel:
        data_dir = config.get('data.dir')
        topic_conf = config.get_config('data.topics')
        file_topics = topic_conf.get('file')
        file_cache = os.path.join(data_dir, topic_conf.get('cache'))
        file_checksum = os.path.join(data_dir, topic_conf.get('checksum'))
        return TopicModel.restore(file_topics, file_cache, file_checksum)

    def search_json(self, search_request: SearchRequest):
        """
        run the search and return the result as json
        :param search_request: the parsed search request
        :return: the search result as json string
        """
        query = search_request.query
        if query.is_empty():
            return 404, "empty query"
        if query.start > self.result_limit:
            raise ClientError("You are not allowed to scroll past {} search results".format(self.result_limit))
        if query.start + query.size > self.result_limit:
            query.size = self.result_limit - query.start
        session = None
        sid = search_request.sid
        if sid and self.manager.has_session(sid):
            session = self.manager.get_session(sid)
        result = self.es.search_main(query, session)
        return jsonify(result)

    def search_html(self, search_request: SearchRequest):
        """
        run the search and return the result as html
        :param search_request: the search request submitted by the user
        :return: the search result as html (call to render_template)
        """
        # interrupt if data structures are missing
        if not self.topic_model:
            flash(message="topic model is not available. please contact the system administrator!", category='alert')
            return render_template('search-startpage.html')

        # break early if the query is empty or redirect if there is no session id
        message = None
        query = search_request.query
        sid = search_request.sid
        if query.is_empty():
            return render_template('search-startpage.html')
        if not (sid and self.manager.has_session(sid)):
            return self.search_redirect_with_id(request.args)

        # submit this request to the current session
        session = self.manager.submit_request(search_request)

        # enforce request limits and fix mistakes
        if query.start >= self.result_limit:
            message = "I'm sorry, I can't show you more than the top {} most relevant results.".format(self.result_limit)
            return self.search_html_render(session, message=message)
        if query.start + query.size > self.result_limit:
            query.size = self.result_limit - query.start

        # execute the search, return results or notify about exceptions
        try:
            es_response = self.es.search_main(query, session)
            result = SearchResult.from_es(es_response, self.topic_model)
            if result.hits_total < 1:
                message = "Your search did not match any documents."
            elif result.hits_total <= query.start:
                message = "There are no more results to display."
            session.submit_response(result)
            # if this is an explicit follow-up query, we're done here
            if search_request.follow_up:
                return self.search_html_render(session, result=result, message=message, follow_up=True)
            # suggested search results
            if session.is_follow_up():
                suggestions = session.last_suggestions
            else:
                suggestions_raw = self.es.search_suggestions(query, session)
                suggestions = SearchResult.from_es(suggestions_raw, self.topic_model)
                result_ids = set(hit.id for hit in result.hits)
                suggestions.hits = [s for s in suggestions.hits if s.id not in result_ids]
                session.last_suggestions = suggestions
            return self.search_html_render(session, result=result, suggestions=suggestions,
                                           message=message, follow_up=False, spelling=True)
        except ClientError as e:
            flash(message=str(e), category='alert')
        except (elasticsearch.exceptions.ConnectionError, urllib3.exceptions.ConnectTimeoutError, ConnectionError):
            flash(message="sorry, elasticsearch seems to be down :'(", category='alert')
        except elasticsearch.exceptions.ElasticsearchException:
            logger.error("there was an issue with the submitted elasticsearch query", exc_info=1)
            flash(message="sorry, I was unable to process your query due to an internal error :'( "
                          "This is likely a bug. Please report it to the site's administrator.",
                  category='alert')
        return render_template('search-startpage.html')

    def search_redirect_with_id(self, args: MultiDict):
        """
        create a new session with a random id and redirect the user
        so the session ID appears in the URL parameters.
        :param args: the request args
        :return: a redirect to the same request with a new random session parameter
        """
        # redirects a user that does not yet have a session id
        params = args.copy()
        sid = util.random_id()
        params['sid'] = sid
        params['step'] = 0
        self.manager.add_session(sid)
        return redirect(url_for('search', **params))

    def search_html_render(self, session: PersistentSession, result: SearchResult = None,
                           suggestions: SearchResult = None, message: str = None,
                           follow_up=False, spelling=False):
        kwargs = self.build_template_kwargs(session, result, suggestions, message, follow_up=follow_up)
        if follow_up:
            return render_template('search-followup.html', **kwargs)
        else:
            return render_template('search-results.html', **kwargs)

    def build_template_kwargs(self, session: PersistentSession, result: SearchResult = None,
                              suggestions: SearchResult = None, message: str = None,
                              follow_up=False) -> Dict[str, Any]:
        """
        create a dictionary that contains all data that is expected by the search result page template.
        :param session: the current search session
        :param result: the search result
        :param message: a notification message (optional)
        :param follow_up: indicates that this is a follow-up request
        :return: a dictionary containing the expected data
        """
        query = session.current_query()
        response = {
            'sid': session.sid,
            'step': session.step,
            'query': query.to_dict(),
            'query_history': session.queries,
            'follow_up': session.follow_up,
        }
        if result:
            response['total'] = result.hits_total
            response['took'] = result.search_time
            response['hits'] = [self._convert_hit(hit) for hit in result.hits]
        if message:
            response['message'] = message
        if not follow_up:
            if session:
                topics = session.topic_centroid.sorted_topics()
                response['topic_centroid'] = self._convert_topics(topics)
                response['topic_graph'] = self._convert_topic_graph(topics)
            if suggestions:
                sugg = [self._convert_hit(hit, snippet_max_chars=200, preview_max_chars=1300)
                        for hit in suggestions.hits]
                response['suggestions'] = sugg
            if self.spellcheck_enabled:
                response['spellcheck'] = self._correct_spelling(query.query, suggest=result.suggest)
        return response

    @classmethod
    def _convert_hit(cls, search_hit: SearchHit, snippet_max_chars=300, preview_max_chars=2500) -> Dict[str, str]:
        """
        converts an elasticsearch hit into a more simplified view that can be
        easily interpreted by the template engine.
        :param hit: a dictionary containing a single elasticsearch hit
        :return: a dictionary containing relevant information about the hit
        """
        source = search_hit.es_hit['_source']
        highlight = search_hit.es_hit.get('highlight', {})
        title = next(iter(highlight.get('title.english', [])), search_hit.title)
        abstract = next(iter(highlight.get('abstract.english', [])), source.get('abstract'))
        snippets = [abstract]
        snippets.extend(highlight.get('pages', [])[:5])
        preview = highlight.get('pages', [])
        preview_missing = "sorry, there is no preview text available"
        return {
            'id': search_hit.id,
            'score': search_hit.score,
            'title': cls._escape(title),
            'authors': source['authors'],
            'abstract': cls._escape(abstract),
            'snippets': cls._build_preview_text(snippets, preview_missing, max_snippets=5, max_char_len=snippet_max_chars),
            'preview': cls._build_preview_text(preview, preview_missing, max_snippets=50, max_char_len=preview_max_chars),
            'url': source['arxiv-url'],
            'date': source['date-created'],
            'topics': cls._convert_topics(search_hit.topics) if search_hit.topics else []
        }

    @staticmethod
    def _convert_topics(topics: List[Tuple[Topic, float]]):
        return [{
            'id': topic.topic_id,
            'score': score,
            'tokens': topic.tokens or [],
        } for topic, score in topics if topic]     # TODO filter by layer (has children?)

    @classmethod
    def _convert_topic_graph(cls, topics: List[Tuple[Topic, float]], n_topics_soft=4, n_topics_hard=6, base_size=1000):
        # get top 4 topics
        # add up to two other topics, if they share a parent node
        # get ancestors
        # skip ancestors that do not branch out (i.e. have only one child)
        # join at a virtual root node, if we have a forest

        # break early, if there are no topics
        if not topics:
            return { 'nodes': [], 'links': [] }

        # select topics (add more if they have common parents)
        leaf_topics = topics[:n_topics_soft]
        leaf_parents = set(topic.parent for topic, score in leaf_topics)
        for topic, _ in topics[n_topics_soft:n_topics_hard]:
            accept = (len(leaf_topics) < n_topics_hard and
                      (len(leaf_parents) < n_topics_soft or
                       (len(leaf_parents) == n_topics_soft and topic.parent in leaf_parents)))
            if accept:
                leaf_topics.append(topic)
                leaf_parents.add(topic.parent)
            else:
                break

        # generate links to ancestors
        nodes = set()              # type: Set[Topic]
        links = defaultdict(set)    # type: Dict[Topic, Set[Topic]]
        for topic, score in topics[:4]:
            cls._topic_get_ancestor_nodes_and_links(topic, nodes, links)

        # reduce the amount of ancestor nodes
        for node in sorted(nodes, key=lambda t: -t.layer):
            # skip already deleted nodes (we're iterating over a copy of this set)
            if node not in nodes:
                continue

            # remove nodes that don't branch
            children = links.get(node)
            if children and len(children) == 1:
                if node.parent:
                    # remove non-branching non-leaf nodes
                    parent = node.parent
                    child = next(iter(children))
                    nodes.remove(node)
                    del links[node]
                    links[parent].remove(node)
                    links[parent].add(child)
                else:
                    # remove top nodes with only one child
                    nodes.remove(node)
                    del links[node]
                continue

        # build the data structure for d3.js
        node_list = sorted(nodes, key=lambda t: t.topic_id)
        node_indices = {n: i for i, n in enumerate(node_list)}
        link_tuples = util.flatten([(source, target) for target in sorted(targets, key=lambda t: t.topic_id)]
                                   for source, targets in sorted(links.items(), key=lambda t: t[0].topic_id))
        topic_scores = dict(topics)
        node_to_layer, layers = cls.__node_to_layer(node_list, links)
        nodes_per_layer = Counter(node_to_layer.values())
        nodes_per_layer_done = defaultdict(int)
        margin = int(base_size * 0.1)
        y_step = int((base_size - 2*margin) / layers-1) if layers > 1 else int(base_size/2)
        node_data = []
        for node in node_list:
            layer_idx_y = node_to_layer[node]
            layer_idx_x = nodes_per_layer_done[layer_idx_y]
            layer_x_total = nodes_per_layer[layer_idx_y]
            x = cls.__x_pos_by_len(layer_idx_x, layer_x_total, width=base_size, min_margin=margin)
            if layer_idx_y < 2 and len(nodes_per_layer) > 2:
                y = margin + margin * layer_idx_y
            else:
                y = margin + layer_idx_y * y_step
            nodes_per_layer_done[layer_idx_y] += 1
            data = {
                'name': node.topic_id,
                'tokens': [token for token, score in node.tokens] if node.tokens else None,
                'score': round(topic_scores[node], 3) if node in topic_scores else 0,
                'x': x,
                'y': y,
            }
            if layer_idx_y < 2:
                data['fixed'] = True
            node_data.append(data)
        return {
            'nodes': node_data,
            'links': [{
                'source': node_indices[source],
                'target': node_indices[target]
            } for source, target in link_tuples]
        }

    @staticmethod
    def __x_pos_by_len(idx, total, width=1000, min_margin=100):
        margin = int(round((width/2 - min_margin) / (total**1.5))) + min_margin
        step = int(round((width - 2*margin) / (total-1))) if total > 1 else 0
        return margin + idx*step

    @staticmethod
    def __node_to_layer(nodes: List[Topic], links: Dict[Topic, Set[Topic]]) -> Tuple[Dict[Topic, int], int]:
        min_layer = min(n.layer for n in nodes)
        node_to_layer = {}
        candidates = [n for n in nodes if n.layer == min_layer]
        layers = 0
        for i in range(100):
            layers = i
            children = []
            for node in candidates:
                node_to_layer[node] = i
                if node in links:
                    children.extend(links[node])
            if children:
                candidates = children
            else:
                break
        return node_to_layer, layers+1

    @classmethod
    def _topic_get_ancestor_nodes_and_links(cls, topic: Topic, nodes: Set[Topic],
                                            links: DefaultDict[Topic, Set[Topic]], depth: int = None):
        nodes.add(topic)
        if topic.parent:
            nodes.add(topic.parent)
            links[topic.parent].add(topic)
            if depth is None or depth > 0:
                cls._topic_get_ancestor_nodes_and_links(topic.parent, nodes, links, depth=(depth-1 if depth else None))
        return nodes, links


    @classmethod
    def _build_preview_text(cls, snippet_list: List[str], default: str = "",
                            max_snippets: int = 5, max_char_len: int = 300) -> Markup:
        """
        generates a result preview from a list of highlighted texts generated by elasticsearch.
        shortens the snippets provided by elasticsearch and concatenates them with '...'
        :param snippet_list: the list of elasticsearch snippets
        :param default: an alternative text to display if there are no snippets
        :param max_char_len: the maximum length of the entire preview text
        :return: a preview text generated from the specified elasticsearch text highlight snippets
        """
        # display the abstract, if there are no snippets (e.g. author filter)
        if snippet_list:
            snippets = [cls._clean_snippet(snippet) for snippet in snippet_list]
            snippets = [s for s in snippets if len(s) > 5]
            if max_snippets:
                snippets = snippets[:max_snippets]
            text = ' … '.join(snippets)
        else:
            text = default
        preview_text = util.smart_truncate(text, length=max_char_len, suffix=' …', strip_punctuation=True)
        return cls._escape(preview_text)

    @classmethod
    def _clean_snippet(cls, snippet: str, max_chars_before=40, mach_chars_after=50) -> str:
        """
        reduces the length of a snippet returned by elasticsearch highlight.
        text gets highlighted with <em>.
        This function sets a limit to the number of allowed characters before and after the snippet.
        Text is not simply cut off, but gets shortened at word boundaries
        :param snippet: the snippet as string. highlighted text is marked with <em> tags
        :param max_chars_before: no more than this many chars before the first highlighted element
        :param mach_chars_after: no more than this many chars after the last highlighted element
        :return: the truncated snipped (if truncation was necessary)
        """
        snippet = snippet.replace('\n', ' ')
        low = snippet.find('<em>')
        if low - max_chars_before > 0:
            snippet = util.smart_truncate_l(snippet, begin_index=low-max_chars_before)
        high = snippet.find('</em>')
        if high + mach_chars_after < len(snippet):
            snippet = util.smart_truncate(snippet, length=high+mach_chars_after)
        return snippet

    @staticmethod
    def _escape(html_string):
        """escape the html string, preserving highlighting (em tags)"""
        return util.escape_html(html_string, whitelist=['em'])

    @classmethod
    def parse_search_request(cls, request: Request):
        args = request.args     # type: MultiDict
        return SearchRequest(
            query=cls.parse_query(args),
            sid=args.get('sid'),
            follow_up=args.get('follow-up', False, type=bool),
            spelling=args.get('spelling', False, type=bool),
        )

    def _correct_spelling(self, text: str, suggest: Dict[str, List[Dict[str, Any]]] = None,
                          max_distance: int = 2) -> Optional[str]:
        """
        looks for spelling errors and suggests corrections
        :param text: the text to correct
        :param max_distance: the maximum allowed Levenshtein distance for corrections
        """
        # find misspelled words and try to correct them based on the Hunspell dictionary
        corrections = self.spell_checker.correct(text, max_distance=max_distance)

        # try to find corrections for terms that were not found in the dictionary
        if suggest:
            for i, (term, state) in enumerate(corrections):
                if state is Spelling.failed:
                    suggestions = suggest.get(term)
                    if suggestions:
                        corrections[i] = (suggestions[0]['text'], Spelling.fixed)

        # return a string with highlighted suggestions, if there were any
        if any(term for term, state in corrections if state is Spelling.fixed):
            # build a highlighted text
            highlighted_correction = " ".join("<em>{}</em>".format(term) if state is Spelling.fixed
                                              else term for term, state in corrections)
            return self._escape(highlighted_correction)

    @classmethod
    def parse_query(cls, args: MultiDict) -> Query:
        """
        parses the query based on the current request args
        :return: a Query object
        """
        return Query(
            query=util.str_blank_to_none(args.get('q')),
            start=args.get('start', 0, type=int),
            size=args.get('size', 10, type=int),
            author=util.str_blank_to_none(args.get('author')),
            date_before=util.parse_date(args.get('date-before')) if args.get('date-before') else None,
            date_after=util.parse_date(args.get('date-after'))  if args.get('date-after') else None,
            step=args.get('step', type=int),
        )

    @staticmethod
    def logging_setup(name: str, config: ConfigTree, set_default: bool = False):
        """
        initialize a logger based on the specified configuration
        :param name: the name of the logger (use the empty string to configure the root logger)
        :param config: the configuration object
        :param set_default: set this configuration for the default (root) logger aswell
        """
        # read config
        log_conf = config.get_config('logging')
        format = log_conf.get('format', '{asctime} [{levelname}]: {message}')
        stdout = log_conf.get_bool('console.enabled', True)
        stdout_level = logging._nameToLevel[log_conf.get('console.level', 'info').upper()]
        logfile = log_conf.get_bool('logfile.enabled', False)
        logfile_level = logging._nameToLevel[log_conf.get('logfile.level', 'info').upper()]
        logfile_path = log_conf.get('logfile.path')
        # setup logging
        if set_default:
            util.logging_setup('', format=format, stdout=stdout, stdout_level=stdout_level)
        util.logging_setup(name, format=format, stdout=stdout, stdout_level=stdout_level,
                           logfile=logfile, logfile_level=logfile_level, logfile_path=logfile_path)

