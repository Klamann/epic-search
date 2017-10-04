import logging
import math
import os
import pickle
import time
from collections import OrderedDict, Counter, defaultdict
from datetime import date
from typing import Dict, List, Union, Any, Tuple, Optional

from search_ui import util

logger = logging.getLogger('search-ui')


class Query:
    """
    A single search query, as submitted by the user.
    Contains the query string and some additional metadata.
    """

    def __init__(self, query: str, start: int = 0, size: int = 10, author: str = None,
                 date_before: date = None, date_after: date = None, step: int = None):
        """
        Initializes a new query
        :param query: the query string for the fulltext search
        :param start: display results beginning with this index
        :param size: retrieve this many results
        :param author: the author name to look for (exact match, if possible, else ranked fuzzy matching)
        :param date_before: only include results before this date
        :param date_after: only include results after this date
        :param step: an identifier for this query within a session
        """
        super().__init__()
        self.query = query              # type: str
        self.start = start              # type: int
        self.size = size                # type: int
        self.author = author            # type: str
        self.date_before = date_before  # type: date
        self.date_after = date_after    # type: date
        self.step = step                # type: int

    def is_regular_query(self) -> bool:
        """
        determines, whether this is a "regular" query,
        i.e. it has a query string and no other filters (author or date) are applied
        :return: True, iff this is a regular query
        """
        return (self.query is not None) and (not self.author) and (not self.date_after) and (not self.date_before)

    def is_empty(self) -> bool:
        """
        determines, whether this query is empty.
        a data-only filter is also considered to be empty. This is a search engine, not a library...
        :return: True, iff this query is empty (no query string, no author filter)
        """
        return (not self.query) and (not self.author)

    def same_as(self, other: 'Query', ignore_date=False) -> bool:
        """
        determines whether this query is the same as the other, considering query terms and filtering,
        but not start, offset or step.
        This can be used to identify identical queries that are already cached by elasticsearch.
        :param other: the other query to compare this one to
        :param ignore_date: if True, ignores differences in the fields date_before and date_after
        :return: True, iff this query is equal to the other, according to the rules of this function
        """
        return (other and isinstance(other, Query)
                and self.query == other.query
                and self.author == other.author
                and (ignore_date or (self.date_before == other.date_before
                                     and self.date_after == other.date_after)))

    def to_dict(self) -> Dict[str, Union[str, int]]:
        """
        Transforms this query to a dictionary.
        objects are converted to strings or numbers (e.g. date -> string in iso format)
        :return: this object as dictionary, ready for json serialization
        """
        d = {}
        if self.query: d['query'] = self.query
        if self.start: d['start'] = self.start
        if self.size: d['size'] = self.size
        if self.author: d['author'] = self.author
        if self.date_before: d['date_before'] = self.date_before.isoformat()
        if self.date_after: d['date_after'] = self.date_after.isoformat()
        if self.step: d['step'] = self.step
        return d

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class SearchRequest:
    """
    defines all data provided by a user that submits a new search request
    """

    def __init__(self, query: Query, sid: str = None, follow_up: bool = False, spelling: bool = False):
        """
        initializes a new search request
        :param query: the current search query
        :param sid: the search id, an identifier of the current search session
        :param follow_up: marks this request as follow-up. Retrieve just the search results.
        :param spelling: mark this request as an accepted spelling correction of the previous one.
        """
        super().__init__()
        self.query = query
        self.sid = sid
        self.follow_up = follow_up
        self.spelling = spelling


# the significant terms of a topic. List of (term, score) tuples.
TopicTokens = List[Tuple[str, float]]


class Topic:
    """
    a single topic in the topic model.
    each topic has a unique identifier and a few tokens that are significant for this topic.
    in case of a hierarchical topic model, links to parent and child topics may be stored.
    """

    def __init__(self, topic_id: str, tokens: TopicTokens = None, layer: int = 1,
                 parent: 'Topic' = None, children: List['Topic'] = None, **kwargs):
        """
        initializes a new topic
        :param topic_id: the ID of this topic
        :param tokens: a list of (token, score) tuples, in descending score order
        :param layer: this topic's layer (depth in the tree, 0 for root node, 1 for
               root's children, and so on)
        """
        super().__init__()
        self.topic_id = topic_id
        self.tokens = tokens
        self.layer = layer
        self.parent = parent
        self.children = children

    @staticmethod
    def restore_topics(topic_dicts: List[Dict[str, Any]]) -> Dict[str, 'Topic']:
        """
        restore a bunch of topics that have been transformed by `store_topics`.
        Also restores object references between topic objects.
        :param topic_dicts: a list of topic dictionaries
        :return: a dictionary mapping from topic id to topic object
        """
        topic_map = {td['topic_id']: Topic._from_dict(td) for td in topic_dicts}
        for topic in topic_map.values():
            topic._restore_links(topic_map)
        return topic_map

    def _restore_links(self, topic_map: Dict[str, 'Topic']):
        """
        restore the links between parent and child topics that have been restore from
        a dictionary representation. Mutates the `parent` and `child` field in all Topics
        :param topic_map: a mapping from topic id to topic object
        """
        if self.parent:
            self.parent = topic_map[self.parent]
        if self.children:
            self.children = [topic_map[child_id] for child_id in self.children]

    def to_dict(self):
        """
        transforms this topic into an easily serializable dictionary structure without object references
        :return: this topic's fields as dictionary, without object references
        """
        return OrderedDict([
            ('topic_id', self.topic_id),
            ('layer', self.layer),
            ('tokens', self.tokens),
            ('parent', self.parent.topic_id if self.parent else None),
            ('children', [child.topic_id for child in self.children] if self.children else None),
        ])

    @staticmethod
    def _from_dict(topic_dict: Dict[str, Any]) -> 'Topic':
        """
        restores a Topic object from it's dictionary representation.
        Does NOT restore links to other Topic objects. To restore a
        serialized topic model, please refer to `restore_topics()`
        :param topic_dict: a topic in dictionary representation (see `to_dict()`)
        :return: the restored topic object (without parent-child-links)
        """
        return Topic(**topic_dict)

    def __str__(self) -> str:
        return util.class_str_from_dict("Topic", self.to_dict())

    def __repr__(self) -> str:
        return self.__str__()


class TopicModel:
    """
    a topic model consists of a set of topics that has been created by a topic
    modeling algorithm based on a document collection.
    """

    def __init__(self, topics: Dict[str, Topic], frequency: Dict[str, int] = None, docs_total: int = None):
        """
        initializes a new topic model from a bunch of topics and some stats about the document
        collection that it was generated from.
        :param topics:  a dictionary mapping from topic id to topic object
        :param frequency: a dictionary mapping from topic id to the number of occurrences of
               this topic in the document collection.
        :param docs_total: the total number of documents this model was built from
        """
        super().__init__()
        self.topics = topics
        self.frequency = frequency
        self.docs_total = docs_total

    @classmethod
    def restore(cls, file_topics: str, file_cache: str, file_checksum: str) -> 'TopicModel':
        """
        restore a previously stored topic from cache (preferred) or parse a topic model
        that has been serialized to json.
        Only accepts the cached model, if the stored checksum matches the json topic model.
        :param file_topics: path to the serialized topic model (json. can be compressed, e.g. json.bz2)
        :param file_cache: path to the cached version of the topic model (will be created, if nonexistent)
        :param file_checksum: path to the file that stores the checksum of file_topics
        :return: the topic model
        """
        if os.path.isfile(file_topics):
            # source exists -> compare to cache or create from source
            if os.path.isfile(file_checksum):
                stored_hash = open(file_checksum, 'rt').read()
                file_hash = util.file_checksum(file_topics)
                if stored_hash == file_hash:
                    logger.debug("restoring topics from cache (checksum validated)")
                    return cls._load_cached_topics(file_cache)
                else:
                    logger.info("reading topics from file '%s' (cache checksum mismatch)", file_topics)
                    return cls._read_and_cache_topics(file_topics, file_cache, file_checksum)
            else:
                logger.info("reading topics from file '%s' (cache empty)", file_topics)
                return cls._read_and_cache_topics(file_topics, file_cache, file_checksum)
        else:
            # no source -> load from cache or fail
            if os.path.isfile(file_cache):
                logger.warning("file '%s' was not found, restoring last cached topic model "
                               "from '%s'", file_topics, file_cache)
                return cls._load_cached_topics(file_cache)
            else:
                logger.warning("topics could not be loaded: file '%s' does not exist "
                               "and no topic model is cached", file_topics)

    @staticmethod
    def _load_cached_topics(fname) -> 'TopicModel':
        """
        restores the topic model from cache (very fast)
        :param fname: the pickle dump to restore from
        :return: the restored topic model
        """
        with util.open_by_ext(fname, 'rb') as fp:
            stored_dict = pickle.load(fp)
            return TopicModel.from_dict(stored_dict)

    @classmethod
    def _read_and_cache_topics(cls, file_topics: str, file_cache: str, file_checksum: str) -> 'TopicModel':
        """
        restores the topic model from the json serialization (can be rather slow) and
        writes the restored topic model as pickle dump to the specified cache file,
        so it can be later restored mich faster.
        also stores a checksum of file_topics, so when the topic model changes,
        the cache will be rebuilt instead of using the now deprecated cache.
        :param file_topics: path to the serialized topic model (json. can be compressed, e.g. json.bz2)
        :param file_cache: will write the cached version of the topic model to this file
        :param file_checksum: the checksum of file_topics will be stored in this file
        :return: the restored topic model
        """
        t0 = time.time()
        doc_collection = util.json_read(file_topics)
        topic_model = cls._from_doc_collection(doc_collection)
        cache_dir = os.path.dirname(file_cache)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        with util.open_by_ext(file_cache, 'wb') as fp:
            pickle.dump(topic_model.to_dict(), fp, protocol=4)
        checksum = util.file_checksum(file_topics)
        open(file_checksum, 'wt').write(checksum)
        logger.info("topics loaded and cached, took {:.2f} s".format(time.time() - t0))
        return topic_model

    @staticmethod
    def _from_doc_collection(doc_collection: Dict[str, Any]) -> 'TopicModel':
        """
        restores the topic model from the json serialization.
        generates Topic objects and restores references between the objects.
        also gathers some stats about the document collection.
        :param doc_collection: the document collection, with topics, as generated by
               `index_builder.topic_model`
        :return: the restored topic model
        """
        topics = Topic.restore_topics(doc_collection['topics'])
        frequency = Counter(util.flatten(([topic_id for topic_id, _ in doc['topics'] or []]
                                          for doc in doc_collection['documents']), generator=True))
        docs_total = len(doc_collection.get('documents'))
        return TopicModel(topics, dict(frequency.items()), docs_total)

    def to_dict(self):
        """
        stores all relevant fields of this topic model in a dictionary.
        these fields can be used as kwargs for the constructor of this class.
        """
        return OrderedDict([
            ('topics', self.topics),
            ('frequency', self.frequency),
            ('docs_total', self.docs_total)
        ])

    @staticmethod
    def from_dict(stored_dict: Dict[str, Any]) -> 'TopicModel':
        """
        restores a TopicModel from it's dictionary representation.
        """
        return TopicModel(**stored_dict)


class SearchHit:
    """
    a search hit is a single document of a search result list
    """

    def __init__(self, es_hit: Dict[str, Any], id: str, score: float, title: str = None,
                 topics: List[Tuple[Topic, float]] = None):
        """
        initializes a new search hit
        :param es_hit: the elasticsearch dictionary of this search hit
        :param id: the document id
        :param score: the document's score for this search query
        :param title: the document's title
        :param topics: the topics of this document (list of (topic, score) tuples)
        """
        super().__init__()
        self.es_hit = es_hit
        self.id = id
        self.score = score
        self.title = title
        self.topics = topics

    @classmethod
    def from_es(cls, es_hit: Dict[str, Any], topic_model: TopicModel = None) -> 'SearchHit':
        """
        creates a new search hit from an elasticsearch result
        :param es_hit: a single search hit, found in es_response['hits']['hits']
        :param topic_model: the topic model for this document collection
               (used to match topic IDs with their respective objects)
        :return: a SearchHit object representation of this hit
        """
        source = es_hit['_source']
        topics = None
        if topic_model:
            topics = cls.convert_topics(source.get('topics', []), topic_model)
        return SearchHit(es_hit, id=source['arxiv-id'], score=es_hit['_score'],
                         title=source['title'], topics=topics)

    @staticmethod
    def convert_topics(source_topics: List[Dict[str, Any]], topic_model: TopicModel)\
            -> List[Tuple[Topic, float]]:
        """
        converts a list of (topic id, score) tuples to a list of (topic object, score) tuples.
        also, sorts the topics by descending score
        :param source_topics: a list of (topic id, score) tuples
        :param topic_model: the topic model to get the topic id -> object mapping from
        :return: a list of (topic object, score) tuples
        """
        topics = ((topic_model.topics.get(t['topic']), t['score']) for t in source_topics)
        return sorted(topics, key=lambda x: x[1], reverse=True)

    def __str__(self):
        return util.class_str_from_dict("SearchHit", OrderedDict([
            ('id', self.id),
            ('score', self.score),
            ('title', self.title),
            ('topics', [(t.topic_id if t else t, score) for t, score in (self.topics or [])]),
        ]))

    def __repr__(self):
        return self.__str__()


class SearchResult:
    """
    a search result is the response to a search query and mainly consists of a list of SearchHits
    """

    def __init__(self, hits: List[SearchHit], hits_total: int = None,
                 suggest: Dict[str, List[Dict[str, Any]]] = None, search_time: float = None):
        """
        initializes a new SearchResult
        :param hits: the list of search results (SearchHit objects)
        :param hits_total: the total amount of hits for the query
        :param suggest: suggested spelling corrections for the search terms (optional)
        :param search_time: the time when the query was submitted
        """
        super().__init__()
        self.hits = hits
        self.hits_total = hits_total
        self.suggest = suggest
        self.search_time = search_time

    @classmethod
    def from_es(cls, es_response: Dict[str, Any], topic_model: TopicModel = None) -> 'SearchResult':
        """
        creates a SearchResult object from an elasticsearch response
        :param es_response: the elasticsearch response (as dict)
        :param topic_model: the topic model for this document collection
        :return: a SearchResult object containing all relevant data from the es response
        """
        hits = [SearchHit.from_es(hit, topic_model) for hit in es_response['hits']['hits']]
        suggest = cls._convert_suggest(es_response)
        return SearchResult(hits, suggest=suggest, hits_total=es_response['hits']['total'],
                            search_time=es_response['took'])

    def hits_per_topic(self) -> Dict[Topic, List[SearchHit]]:
        """
        :return: all hits grouped by topic ID
                  (i.e. for each topic ID, return a list of all hits that contain this topic)
        """
        topic_hits = defaultdict(list)
        for hit in self.hits:
            for topic, _ in (hit.topics or []):
                topic_hits[topic].append(hit)
        return topic_hits

    @staticmethod
    def _convert_suggest(es_response: Dict[str, Any], identifier: str = 'main')\
            -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        convert spell checking suggestions, if available.
        returns a dictionary from term to a list of suggestions (each suggestion is a dict that
        contains the actual term as well as some other stats) or None, if there are no suggestions.
        :param es_response: the elasticsearch response to read suggestions from (if any)
        :param identifier: the identifier under "suggest" to read from
        :return: the converted suggestions or None
        """
        if 'suggest' in es_response and identifier in es_response['suggest']:
            return {sug['text']: sug['options'] for sug in es_response['suggest']['main']}

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class TopicScore:
    """
    a helper class that is used to calculate the score of a topic in the topic centroid.
    first we generate a score for each topic from various properties of the search results.
    when all scores are known, the components are normalized to the [0..1] range and the final score
    can be calculated.
    please refer to the thesis for a detailed description of the algorithm.
    """

    def __init__(self, topic: Topic, agg_max: float, agg_sum: float, agg_count: float, tfidf: float, raw_scores: bool = True):
        """
        creates a new TopicScore instance
        :param agg_max: the value of the agg_max score (the highest document score for this topic)
        :param agg_sum: the value of the agg_sum score (the sum of all document scores for this topic)
        :param agg_count: the value of the agg_count score (the number of times this topic occurs
                          in the result list)
        :param tfidf: the value of the tfidf score (tf-idf is based on the frequency of this topic
                      in the result list relative to the frequency of this topic in the entire
                      document collection)
        :param raw_scores: indicates, that the specified scores are raw, i.e. have not been
               scaled to a range between 0 and 1. Use `scale_scores(...)` to change this.
        """
        super().__init__()
        self.topic = topic
        self.agg_max = agg_max
        self.agg_sum = agg_sum
        self.agg_count = agg_count
        self.tfidf = tfidf
        self.raw_scores = raw_scores
        self.score = None   # type: float

    def weighted_score(self, w_tfidf=0.5, w_agg=0.5, w_agg_max=0.45, w_agg_sum=0.35,
                       w_agg_count=0.2, memoize=True, **kwargs) -> float:
        """
        calculates the weighted score for this topic & memoizes the value.
        the topic scores need to be scaled to a common range first, or the weighted sum will not
        be meaningful. Call `scale_scores()` before calculating the weighted score.
        :param w_tfidf: the weight of the tf-idf part of the topic score
        :param w_agg: the weight of all aggregated scores (agg_max, agg_sum, agg_count)
                      you can adjust the relative weights of each of these scores with the
                      following arguments, but the sum of these weights will never exceed this value
        :param w_agg_max: the weight of agg_max score (relative to the other agg* scores)
        :param w_agg_sum: the weight of agg_sum score (relative to the other agg* scores)
        :param w_agg_count: the weight of agg_count score (relative to the other agg* scores)
        :param memoize: use the memoized score value, if possible (warning: won't update the
               score after the first call, even when the specified weights change!)
        :return: the weighted score for this topic
        """
        if not self.score or not memoize:
            if self.raw_scores:
                raise ValueError("cannot calculate the weighted score for raw score values "
                                 "(call `scale_scores()` for the search result list first!)")
            agg_avg = w_agg_max * self.agg_max +  w_agg_sum * self.agg_sum + w_agg_count * self.agg_count
            self.score = w_tfidf * self.tfidf + w_agg * agg_avg
        return self.score

    @classmethod
    def from_hits(cls, topic: Topic, hits: List[SearchHit], topic_model: TopicModel) -> 'TopicScore':
        """
        calculates the topic scores from a topic and the list of search hits that contain
        this topic. The topic model is required to calculate tf-idf (need global topic frequency)
        :param topic: the topic to find the score for
        :param hits: all search hits that are about this topic
        :param topic_model: the topic model of this document collection
        :return: the TopicScore for the specified topic and search hits.
        """
        # the search result score of every hit, multiplied with the score of the specified topic
        # for the document
        hit_scores_adjusted = [next((score for t, score in hit.topics if t.topic_id == topic.topic_id), 0)
                               * hit.score for hit in hits]

        # aggregations: max, sum and count
        agg_max = max(hit_scores_adjusted)
        agg_sum = sum(hit_scores_adjusted)
        agg_count = len(hit_scores_adjusted)

        # tf-idf
        tf = len(hits)
        idf = math.log(topic_model.docs_total / topic_model.frequency[topic.topic_id], 10)
        tfidf = tf * idf

        return TopicScore(topic, agg_max, agg_sum, agg_count, tfidf, raw_scores=True)

    @classmethod
    def scale_scores(cls, topic_scores: List['TopicScore']) -> None:
        """
        normalizes all scores in the specified list of TopicScore objects to the [0..1] range.
        For each score type, the maximum in the list is calculated, then all scores are divided
        by this maximum
        :param topic_scores: a list of topic scores (usually from a single search result)
        :return mutates the topic scores and sets raw_scores to False
        """
        max_agg_max = max(topic_score.agg_max for topic_score in topic_scores)
        max_agg_sum = max(topic_score.agg_sum for topic_score in topic_scores)
        max_agg_count = max(topic_score.agg_count for topic_score in topic_scores)
        max_tfidf = max(topic_score.tfidf for topic_score in topic_scores)
        for topic_score in topic_scores:
            topic_score.agg_max = topic_score.agg_max / max_agg_max
            topic_score.agg_sum = topic_score.agg_sum / max_agg_sum
            topic_score.agg_count = topic_score.agg_count / max_agg_count
            topic_score.tfidf = topic_score.tfidf / max_tfidf
            topic_score.raw_scores = False

    def __str__(self):
        return util.class_str_from_dict("SearchHit", OrderedDict([
            ('topic', self.topic.topic_id),
            ('raw_scores', self.raw_scores),
            ('score', self.score),
            ('tfidf', self.tfidf),
            ('agg_max', self.agg_max),
            ('agg_sum', self.agg_sum),
            ('agg_count', self.agg_count),
        ]))

    def __repr__(self):
        return self.__str__()


class TopicCentroid:
    """
    the topic centroid represents the currently most important topics in a search session.
    It's main data structure is a Topic -> score mapping, where the relevant topics are stored.
    With each new search query of the user, the topics of the search results are analyzed and
    merged with the topic centroid.
    """

    def __init__(self, topic_model: TopicModel, f_cooldown: float = 0.7, w_shift: float = 0.4):
        """
        Initializes a new, empty topic centroid
        :param topic_model: the topic model for the target document collection
        :param f_cooldown: the cooldown fraction. During each step, the scores of the existing
               topic model will be multiplied by this value. Recommended range: between 0.5 and 1.0
        :param w_shift: the topic shift weight. If an existing topic is added in a new step, their
               scores will be combined. For w_float=1.0, the scores will be summed up, while for
               0.0, only the higher score will remain.
        """
        super().__init__()
        self.topic_model = topic_model
        self.f_cooldown = f_cooldown
        self.w_shift = w_shift
        self.centroid = {}      # type: Dict[Topic, float]

    def sorted_topics(self) -> List[Tuple[Topic, float]]:
        """
        :return: a list of (topic, score) tuples generated from the current topic centroid,
                  ordered by descending score.
        """
        return sorted(self.centroid.items(), key=lambda x: x[1], reverse=True)

    def update(self, search_result: SearchResult) -> None:
        """
        update the topic centroid by analyzing the topics in the provided search results,
        then merging the topic scores with the current topic centroid.
        :param search_result: a new search result
        """
        current_topics = self.identify_topics(search_result)
        self.merge(current_topics)

    def identify_topics(self, search_result: SearchResult, **kwargs) -> Dict[Topic, float]:
        """
        takes a search result list and finds the most significant topics among them.
        The exact algorithm is described in the `TopicScore` class
        :param search_result: the search result to find the topics for
        :param kwargs: further arguments will be passed along to `TopicScore.weighted_score(...)`
        :return: a dictionary containing the topics of the result list and their scores
        """

        # TODO influence of ancestor topic scores!
        # (low scoring ancestor topics should reduce the score of their descendants...)

        try:
            topic_hits = search_result.hits_per_topic()
            topic_hits_filtered = [(topic, hits) for topic, hits in topic_hits.items() if not topic.children]
            topic_scores = [TopicScore.from_hits(topic, hits, self.topic_model) for topic, hits in topic_hits_filtered]
            TopicScore.scale_scores(topic_scores)
            weighted_scores = {topic_score.topic: topic_score.weighted_score(**kwargs) for topic_score in topic_scores}
            return weighted_scores
        except (AttributeError, ValueError):
            logger.error("Failed to find topics for certain documents. Are you sure that "
                         "the local topic model and the document annotations in elasticsearch "
                         "come from the same source?", exc_info=1)
            return {}

    def merge(self, new_topics: Dict[Topic, float], min_score=0.1) -> None:
        """
        merge a set of new topics with the existing topic centroid
        :param new_topics: the new topics, with scores
        :param min_score: after the merge, remove all topics from the centroid whose score
               is below this value
        :return: mutates the topic centroid
        """
        # merge a new centroid with the existing one...
        if self.centroid:
            f_cooldown = self.f_cooldown
            w_shift = self.w_shift
            # cooldown existing scores
            centroid = {topic: score * f_cooldown for topic, score in self.centroid.items()}
            # merge new topics with existing ones
            for topic, new_score in new_topics.items():
                if topic in centroid:
                    # merge scores
                    old_score = centroid[topic]
                    merged_score = max(old_score, new_score) + w_shift * min(old_score, new_score)
                    centroid[topic] = merged_score
                else:
                    # submit new topic
                    centroid[topic] = new_score
            # remove items with low score & replace the old topic centroid
            self.centroid = {topic: score for topic, score in centroid.items() if score >= min_score}
        else:
            # no topic centroid yet? put the new one in place
            self.centroid = new_topics
