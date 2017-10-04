import json
import logging
from collections import OrderedDict
from typing import List, Any, Union
from typing import Tuple, Dict

from elasticsearch import Elasticsearch
from pyhocon import ConfigTree

from search_ui import Query, TopicCentroid, Session, util

logger = logging.getLogger('search-ui')


class EsQuery(OrderedDict):
    """
    Representation of an elasticsearch query as python dict.
    Provides some convenience functions to build the skeleton that is used to execute
    search queries and to extend it using custom query functions.
    """

    # spelling suggestions (elasticsearch's "suggest" feature) are generated using these parameters
    suggestion_params = {
        "term": {
            "field": "abstract",
            "max_edits": 2,
            "prefix_length": 2,
            "min_doc_freq": 2
        }
    }

    def __init__(self, *args, **kwargs):
        """
        initialize a new elasticsearch query.
        if no arguments are provided, a new query skeleton is built.
        otherwise, the contents of this dictionary will be updated so they follow the
        expected format
        """
        super().__init__(*args, **kwargs)
        # build the query skeleton, if necessary
        if len(self) == 0:
            self.update(self.__base_query())
        # shortcuts to relevant nodes
        self._bool_outer = self['query']['bool']
        self._bool_inner = self._bool_outer['must']['bool']
        self._should = self._bool_inner['should'] if 'should' in self._bool_inner else []
        self._must = self._bool_inner['must'] if 'must' in self._bool_inner else []
        self._filter = self._bool_outer['filter'] if 'filter' in self._bool_outer else []

    def add_should_query(self, query: Dict[str, Any]):
        """
        add a new query to the "should" clause of this query.
        documents that match this query will be added to the result list, but it is not required
        that ALL results match this query.
        :param query: the query to add
        """
        if not self._should:
            self._bool_inner['should'] = self._should
        self._should.append(query)

    def extend_should_query(self, queries: List[Dict[str, Any]]):
        """
        extends the "should" clause with a list of queries.
        :param queries: the queries to add
        """
        if queries:
            if not self._should:
                self._bool_inner['should'] = self._should
            self._should.extend(queries)

    def add_must_query(self, query: Dict[str, Any]):
        """
        adds a new query to the "must" clause of this query.
        all documents in the result list must match this query.
        :param query: the query to add
        """
        if not self._must:
            self._bool_inner['must'] = self._must
        self._must.append(query)

    def add_filter(self, filter: Dict[str, Any]):
        """
        adds a selector to the "filter" clause of this query.
        documents that do not match this filter will be removed from the result list.
        :param filter: the filter to add
        """
        if not self._filter:
            self._bool_outer['filter'] = self._filter
        self._filter.append(filter)

    def update_offset(self, query: Query = None, start: int = None, size: int = None):
        """
        updates the start and size parameters of this query.
        start = changes the index from which results should be returned (default: 0).
                useful when more results should be loaded, e.g. when the user switches
                to the second result page
        size = the number of results that should be returned after the start index (default: 10)
        :param query: reads the start and size parameter from the query object, if specified
        :param start: the index offset for the result list
        :param size: the number of results to retrieve
        """
        if start:
            self['from'] = start
        elif query:
            self['from'] = query.start
        if size:
            self['size'] = size
        elif query:
            self['size'] = query.size
        return self

    def set_suggestion_text(self, text: str):
        """
        retrieve spelling corrections for the specified text, based on terms in the index.
        :param text: the text to retrieve suggestions for
        """
        self["suggest"] = {
            "text": text,
            # we can store different parameters under arbitrary keys.
            # by convention, let's store the preferred suggestions under the key 'main'.
            "main": self.suggestion_params
        }

    @classmethod
    def from_query(cls, q: Query = None) -> 'EsQuery':
        """
        creates a new EsQuery from the specified Query object.
        :param q: the query object to read properties from
        """
        es_query = EsQuery()
        es_query.update_offset(query=q)
        return es_query

    @staticmethod
    def restore(query_dict: Dict[str, Any]) -> 'EsQuery':
        """
        creates an EsQuery object from a raw elasticsearch query (dict).
        If the query_dict is already an EsQuery instance, it will be returned unchanged.
        :param query_dict: the elasticsearch query
        :return: an EsQuery instance build from the specified elasticsearch query.
        """
        if isinstance(query_dict, EsQuery):
            return query_dict
        else:
            return EsQuery(query_dict)

    @staticmethod
    def __base_query(start: int = 0, size: int = 10) -> Dict[str, Any]:
        """
        generate the basic template for the query.
        Includes offset, result length, source filter, text highlighting.
        :param start: the start offset of the query (default: 0)
        :param size: the number of results to return (default: 10)
        :return: the basic elasticsearch query template
        """
        return {
            "from": start,
            "size": size,
            "_source": {
                "exclude": ["pages"]
            },
            "query": {
                "bool": {
                    # we can use filters in the outer bool query. however, these "filters" actually
                    # match all elements that match the filter condition, not just the returned
                    # results. therefore we use the must clause. This calculates the intersection
                    # of the results of the filter and the must clause, which is our desired behaviour.
                    'must': {
                        'bool': {}
                    }
                }
            },
            "highlight": {
                "fields": [
                    # highlight the entire text for title, authors and abstract
                    { "title.english": {"number_of_fragments" : 0} },
                    { "authors": {"number_of_fragments" : 0} },
                    { "abstract.english": {"number_of_fragments" : 0} },
                    # highlight only the relevant text snippets for the page contents
                    { "pages": {"fragment_size" : 500, "number_of_fragments" : 50, "order" : "score"} },
                ]
            }
        }


class EsBridge:
    """
    the bridge to elasticsearch. Build, analyze, expand & execute queries.
    """

    multi_match_fields = [
        "title.english^3",
        "authors^2",
        "abstract.english^2",
        "pages"
    ]
    more_like_this_fields = [
        "title.english",
        "abstract.english",
        "pages"
    ]
    script_text_query = "Math.log1p(_score) * 0.434"
    script_topic_query = "Math.log1p(_score) * 1.7"

    def __init__(self, config: ConfigTree):
        """
        initialize the bridge by connecting to elasticsearch using these parameters
        :param host: the elasticsearch host
        :param port: the elasticsearch tcp port (json-based protocol)
        :param http_auth: http authentication (tuple of username and password, optional)
        """
        super().__init__()
        # elasticsearch config
        es_conf = config.get_config('elastic')  # type: ConfigTree
        self.host = es_conf.get('host')
        self.port = es_conf.get_int('port')
        self.http_auth = (es_conf.get('user'), es_conf.get('pass')) if es_conf.get('user') else None
        self.index = es_conf.get('index')
        self.doc_type = es_conf.get('doctype')
        # elasticsearch instance
        self.es = Elasticsearch(hosts=[{'host': self.host, 'port': self.port}], http_auth=self.http_auth)
        # search config
        search_conf = config.get_config('search')   # type: ConfigTree
        self.result_limit = search_conf.get_int('result_limit')
        self.max_queries_per_session = search_conf.get_int('session.max_queries')
        self.first_query_must = search_conf.get_bool('session.first_query_must')
        self.exp_base = search_conf.get_float('session.exp_base')
        self.weight_search_text = search_conf.get_float('weights.search.text')
        self.weight_search_topics = search_conf.get_float('weights.search.topics')
        self.weight_suggestions_text = search_conf.get_float('weights.suggestions.text')
        self.weight_suggestions_topics = search_conf.get_float('weights.suggestions.topics')

    def search_main(self, query: Query, session: Session) -> Dict[str, Any]:
        """
        execute the query and return the results as given by elasticsearch (json parsed as dict)
        :param query: the latest search query
        :param session: the current search session
        :return: the elasticsearch result as dict
        """
        # determine if we can make an exact match on the author name
        exact_match_author = False
        if query.author:
            logger.info("author query detected. checking for exact matches...")
            query_author_exact = self.exact_match_query("authors.raw", query.author)
            result_count = self.es.count(self.index, self.doc_type, query_author_exact)
            exact_match_author = result_count.get('count', 0) > 0

        # compare to last query & skip, if the query is the same
        if session.last_es_query and session.is_follow_up():
            # same query as before (this is not unusual, e.g. when more results are retrieved)
            logger.info("same user query, reusing previously expanded query with adjusted parameters")
            es_query = self.adjust_query(session.last_es_query, query)
        else:
            # build a new es query
            es_query = self.build_search_query(query, session, exact_match_author)
        # store the last es query
        session.last_es_query = es_query

        # execute
        logger.debug('es query (for search): %s', json.dumps(es_query))
        result = self.es.search(self.index, self.doc_type, es_query)
        return result

    def search_suggestions(self, query: Query, session: Session) -> Dict[str, Any]:
        """
        find suggested documents based on the current topic centroid.
        Also applies full-text search based on the current session context,
        but with significantly lower weight.
        :param query: the latest search query
        :param session: the current search session (contains the topic centroid)
        :return: the elasticsearch result as dict
        """
        es_query = self.build_suggestion_query(query, session)
        logger.debug('es query (for suggestions): %s', json.dumps(es_query))
        result = self.es.search(self.index, self.doc_type, es_query)
        return result

    def build_search_query(self, query: Query, session: Session, exact_match_author=False) -> EsQuery:
        """
        generates a new elasticsearch query for the specified search terms and the
        current session context.
        this query has a strong focus on full-text search, especially the latest query
        submitted by the user. all filter options are available. the ranking of results
        can be influenced by the topics in the topic centroid.
        the results of this query are intended for the main search result list.
        :param query: the latest user query
        :param session: the current search session
        :param exact_match_author: will only match the exact author name specified in the
               user query, if set to True
        :return: the elasticsearch query for the specified search terms
        """
        return self.build_query(query,
                                session,
                                add_query_conditions=True,
                                exact_match_author=exact_match_author,
                                first_query_must=self.first_query_must,
                                topics_must=False,
                                spell_check=True,
                                exp_base=self.exp_base,
                                weight_terms=self.weight_search_text,
                                weight_topics=self.weight_search_topics)

    def build_suggestion_query(self, query: Query, session: Session) -> EsQuery:
        """
        generates a new elasticsearch query for the specified session context and
        the current search terms.
        this query has a strong focus on the topics in the topic centroid.
        it will only match document that are associated with at least one topic in the
        top 10 topics of the current topic centroid. documents are primarily ranked
        by matching topics, but the ranking is also influenced to a limited degree
        by the search terms in the query history. highlighting is supported,
        filtering based on time and author is disabled.
        :param query: the latest user query
        :param session: the current search session
        :return: the elasticsearch query for the current session context
        """
        return self.build_query(query,
                                session,
                                add_query_conditions=False,
                                exact_match_author=False,
                                first_query_must=False,
                                exp_base=self.exp_base,
                                spell_check=False,
                                weight_terms=self.weight_suggestions_text,
                                weight_topics=self.weight_suggestions_topics)\
            .update_offset(start=0, size=20)

    @classmethod
    def build_query(cls, query: Query, session: Session, add_query_conditions=True,
                    exact_match_author=False, first_query_must=False, topics_must=False,
                    spell_check=False, weight_terms: float = None, weight_topics: float = None,
                    exp_base=0.5) -> EsQuery:
        """
        generates a new elasticsearch query for the specified session context and
        the current search terms.
        This is a quite generic template that can be used to make a fulltext search or to match
        topics in the topic centroid or just to filter documents based on some criteria.
        Also, the weights for each of these operations can be adjusted freely.
        Have a look at the documentation of the function arguments for more information.
        :param query: the latest user query
        :param session: the current search session
        :param add_query_conditions: use the conditions that might be specified in the current
               user query (default: True)
        :param exact_match_author: match the exact author name that might be specified in the
               current user query (default: False, which initializes a fulltext search for the
               author name instead)
        :param first_query_must: at least one term of the latest user query must match (default: False).
               this can be used to prevent situations where the top results are just based on the
               query history and have nothing to do with the latest query
        :param topics_must: at least one topic from the top 10 topics of the topic centroid must match
               (default: False). This can be used to prefer documents based on the topic centroid
        :param spell_check: get suggestions for misspelled terms using the es "term suggester"
        :param weight_terms: multiplier for the results of the fulltext search (scores are already
               scaled to about the same range as the topic search)
        :param weight_topics: multiplier for the results of the topic search
        :param exp_base: the base for the query history weight function. lower values give less
               weight to older queries. must be between 0 and 1.
        :return: the elasticsearch query for this specification
        """
        es_query = EsQuery.from_query(query)
        cls._query_add_fulltext(session, es_query, exp_base=exp_base, first_must=first_query_must, score_weight=weight_terms)
        if add_query_conditions:
            cls._query_add_conditions(query, es_query, exact_match_author)
        if session.topic_centroid.centroid:
            cls._query_add_topics(es_query, session.topic_centroid, force=topics_must,  score_weight=weight_topics)
        if spell_check:
            es_query.set_suggestion_text(query.query)
        return es_query

    @classmethod
    def adjust_query(cls, es_query: Dict, query: Query):
        """
        Adjusts an existing elasticsearch query based on the specified user request.
        (actually just changes offset and size)
        :param es_query: the existing elasticsearch query
        :param query: the updated user query
        :return: the same (mutated) es query
        """
        es_query = EsQuery.restore(es_query)
        es_query.update_offset(query)
        return es_query

    @classmethod
    def _query_add_fulltext(cls, session: Session, es_query: EsQuery, score_weight: float = None,
                            first_must=False, exp_base=0.5, more_like_this: bool = False) -> EsQuery:
        """
        creates a fulltext search query based on the specified session and adds it to the
        provided elasticsearch query.
        :param session: a search session. contains the user's query history, including the latest query
        :param es_query: an existing elasticsearch query. the fulltext search will be added to this request
        :param score_weight: the overall weight of the fulltext search (optional. can be used to make
               this more or less relevant compared to other functions). Defaults to 1.0 (i.e. no change)
        :param first_must: at least one term of the latest user query must match (default: False).
               this can be used to prevent situations where the top results are just based on the
               query history and have nothing to do with the latest query
        :param exp_base: the base of the function that is used to calculate the weight of each query
               based on it's position in the query history. must be between 0 and 1. reasonable
               values are between 0.5 and 0.9
        :param more_like_this: use a more_like_this query, if set to True
        :return: the mutated elasticsearch query
        """
        query_last = session.current_query()
        if not query_last.query:
            es_query.add_should_query({"match_all": {}})
            return es_query

        # iterate over queries in reverse order (newest query first), ignoring author-only queries
        fulltext_queries = []   # type: List[List[Dict[str,Any]]]
        # TODO limit query history length, but keep first query!
        query_history = [q for q in session.queries[::-1] if q.query]
        for index, query in enumerate(query_history):
            query_string = query.query
            # weight := base ** index, except for first query
            weight = exp_base if (index+1 == len(query_history)) else exp_base ** index
            fulltext_query = [
                cls.dis_max([
                    cls.match_phrase("title", query_string, slop=2, boost=2.0 * weight),
                    cls.match_phrase("title.english", query_string, slop=2, boost=2.0 * weight),
                ]),
                cls.match_phrase("abstract.english", query_string, slop=2, boost=2.0 * weight),
                cls.match_phrase("pages", query_string, slop=3, boost=1.5 * weight),
                cls.multi_match(query_string, cls.multi_match_fields, boost=0.3 * weight, minimum_should_match="67%"),
            ]
            if more_like_this:
                fulltext_query.append(cls.more_like_this(query_string, cls.more_like_this_fields, boost=2.0 * weight))
            fulltext_queries.append(fulltext_query)

        # calculate weights and add to the es query
        if first_must:
            # add the latest query to the must block
            first_query = {'bool': {'should': fulltext_queries[0]}}
            fulltext_queries = fulltext_queries[1:]
            query_scored = cls.function_score_script(first_query, script=cls.script_text_query, factor=score_weight)
            es_query.add_must_query(query_scored)
        if fulltext_queries:
            # add all remaining queries in the query history to the should block
            query_should = {'bool': {'should': util.flatten(fulltext_queries)}}
            query_scored = cls.function_score_script(query_should, script=cls.script_text_query, factor=score_weight)
            es_query.add_should_query(query_scored)
        return es_query

    @staticmethod
    def _query_add_conditions(q: Query, es_query: EsQuery, exact_match_author=False) -> Dict:
        if q.is_regular_query():
            return es_query

        # filter by date (before and/or after)
        if q.date_before or q.date_after:
            date_created = {}
            range_filter = {
                "range" : {
                    "date-created": date_created
                }
            }
            if q.date_after:
                date_created['gte'] = q.date_after.isoformat()
            if q.date_before:
                date_created['lte'] = q.date_before.isoformat()
            es_query.add_filter(range_filter)

        # filter by author (if exact match) or search for authors names (must query)
        if q.author:
            if exact_match_author:
                term_filter = {
                    "term": {
                        'authors.raw': q.author
                    }
                }
                es_query.add_filter(term_filter)
            else:
                author_search = {'match': {"authors": q.author}}
                es_query.add_must_query(author_search)
        return es_query

    @classmethod
    def _query_add_topics(cls, es_query: EsQuery, topic_centroid: TopicCentroid, force=False,
                          score_weight=1.0, n_topics=10) -> Dict:
        """
        adds a topic query to the existing es query. Searches for documents based on the selected topic
        :param es_query: the existing elasticsearch query
        :param topic_centroid: the topics to search for
        :param force: all returned documents must have at least one of the specified topics,
                      if force=True (adds the topic query to the must clasue instead of the
                      default should clause)
        :param score_weight: multiply the results of this query by this factor
        :param n_topics: use the top n topics from the topic centroid
        :return: the adjusted elasticsearch query (same reference, query gets mutated)
        """
        # select topics
        top_topics = topic_centroid.sorted_topics()[:n_topics]
        # create query from topics
        should = []
        query = {'bool': {'should': should}}
        for topic, score in top_topics:
            # term query for topic matching, has to be wrapped in a nested query
            topic_query = cls.term_score_query("topics.topic", topic.topic_id, "topics.score", score)
            nested = cls.nested_query("topics", topic_query)['query']
            should.append(nested)

        # add the topic query to the must or should clause
        query_scored = cls.function_score_script(query, script=cls.script_topic_query, factor=score_weight)
        if force:
            es_query.add_must_query(query_scored)
        else:
            es_query.add_should_query(query_scored)
        return es_query

    @staticmethod
    def match_phrase(field: str, query: str, slop: int = None, boost: float = None) -> Dict[str, Any]:
        """
        builds a phrase query (prefers appearance of terms in order).
        see https://www.elastic.co/guide/en/elasticsearch/guide/current/phrase-matching.html
        :param field: the field to match on
        :param query: the phrase to match
        :param slop: how many times are you allowed to move a term in order to make query
               and document match? (increases recall, decreases precision and performance)
        :param boost: how much to boost this query
        :return: a query fragment with "match_phrase" at the root
        """
        query_params = {"query": query}
        if slop:
            query_params['slop'] = slop
        if boost:
            query_params['boost'] = boost
        return {
            "match_phrase": {
                field: query_params
            }
        }

    @staticmethod
    def multi_match(query: str, fields: List[str], minimum_should_match: Union[int, str] = "50%",
                    boost: float = None) -> Dict[str, Any]:
        """
        builds a multi match query (matches any search term, ignores ordering).
        see https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-match-query.html
        :param query: the phrase to match
        :param fields: the fields to match on
        :param minimum_should_match: at least this many query terms must match for a document to be selected
        :param boost: how much to boost this query
        :return: a query fragment with "multi_match" at the root
        """
        query_params = {
            "query": query,
            "type": "most_fields",
            "minimum_should_match": minimum_should_match,
            "fields": fields
        }
        if boost:
            query_params['boost'] = boost
        return {"multi_match": query_params}

    @staticmethod
    def more_like_this(query: str, fields: List[str], boost: float = None, max_query_terms: int = None):
        query_params = {
            "like": query,
            "fields": fields,
            "min_term_freq": 1,
        }
        if boost:
            query_params['boost'] = boost
        if max_query_terms:
            query_params['max_query_terms'] = max_query_terms
        return {"more_like_this": query_params}

    @staticmethod
    def exact_match_query(field: str, value: str):
        """
        Executes an exact match query on the specified field.
        If the field is an array, at least one value must match.
        The field must be not analyzed or you won't get exact matches!
        :param field: the field to search in
        :param value: the value to match
        :return: the query as python dict
        """
        return {
            "query": {
                "constant_score": {
                    "filter": {
                        "term": {
                            field: value
                        }
                    }
                }
            }
        }

    @staticmethod
    def term_query(field: str, terms: List[Tuple[str,float]]):
        q_term_list = []
        q_bool_should = {"query": {"bool": {"should": q_term_list}}}
        for term, boost in terms:
            q_term = {
                "term": {
                    field: {
                        "value": term,
                        "boost": boost
                    }
                }
            }
            q_term_list.append(q_term)
        return q_bool_should

    @staticmethod
    def term_score_query(field: str, term: str, score_field: str, score_multi: float = 1.0,
                         default: float = 0.0) -> Dict[str, Any]:
        """
        a term query whose match score is defined by another field
        :param field: the field to match on
        :param term: the term to match
        :param score_field: the field to get the score from
        :param score_multi: an optional multiplier for the field's score (default: 1.0)
        :param default: the value to return when no score field is found (default: 0)
        :return: a function_score query with field_value_factor that wraps a term query
        """
        return {
            "function_score": {
                "query": {
                    "constant_score": {
                        "filter": {
                            "term": {
                                field: term
                            }
                        },
                        # this is just the return value for the constant_score, not an actual "boost"
                        "boost": 1.0
                    }
                },
                "field_value_factor": {
                    "field": score_field,
                    "factor": score_multi,
                    "missing": default
                }
            }
        }

    @staticmethod
    def dis_max(queries: List[Dict[str, Any]], tie_breaker: float = None, boost: float = None):
        dis_max = {"dis_max": {"queries": queries}}
        if tie_breaker:
            dis_max['tie_breaker'] = tie_breaker
        if boost:
            dis_max['boost'] = boost
        return dis_max

    @staticmethod
    def function_score_field_value(query: Dict[str,Any], field="_score", factor=1.0, modifier="none", missing=1) -> Dict[str, Any]:
        """
        wraps a query in a function_score, which calculates a custom score for the inner query.
        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html#function-field-value-factor
        :param query: the query to apply the function score to
        :param field: the field to get the score from (use `_score` to access the function score)
        :param factor: multiply the score by this value
        :param modifier: modify the score using a function, e.g. square, sqrt or  log1p
        :param missing: default value for missing results
        :return: a function_score query
        """
        return {
            "function_score": {
                "field_value_factor": {
                    "field": field,
                    "factor": factor,
                    "modifier": modifier,
                    "missing": missing
                },
                "query": query
            }
        }


    @staticmethod
    def function_score_script(query: Dict[str,Any], script: str, factor: float = None) -> Dict[str, Any]:
        """
        wraps a query in a function_score [1], which calculates a custom score for the inner query.
        In the function, use `_score` to access the function score.
        Read the API docs for math functions [2] and more that can be used inside scripts.
        [1]: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html#function-script-score
        [2]: https://www.elastic.co/guide/en/elasticsearch/painless/5.5/painless-api-reference.html#painless-api-reference-Math
        :param query: the query to apply the function score to
        :param script: the script to execute (painless scripting language)
        :param factor: multiply the result of the script by this value
        :return: a function_score query
        """
        if factor:
            script = "({}) * {}".format(script, factor)
        return {
            "function_score": {
                "script_score": {
                    "script": {
                        "inline": script
                    }
                },
                "query": query
            }
        }

    @staticmethod
    def nested_query(path: str, query: Dict[str,Any] = None, score_mode: str = None):
        outer_query = {
            "query": {
                "nested": {
                    "path": path
                }
            }
        }
        nested = outer_query['query']['nested']
        if score_mode:
            nested['score_mode'] = score_mode
        if query:
            nested['query'] = query['query'] if 'query' in query else query
        return outer_query
