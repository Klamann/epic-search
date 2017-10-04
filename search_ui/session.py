import logging
from collections import OrderedDict
from datetime import datetime
from typing import Optional, List, Dict, Any

from search_ui import SearchRequest, Query, TopicCentroid, SearchResult, TopicModel

logger = logging.getLogger('search-ui')


class Session:
    """
    a search session that contains the current state (es query, search results, topic centroid)
    as well as some historical data (query history, start time).
    Navigation within the query history is NOT supported (see class PersistentSession for that).
    """

    def __init__(self, sid: str, topic_model: TopicModel = None):
        """
        create a new session with the specified sid
        :param sid: the search id, a unique identifier for this session
        """
        super().__init__()
        self.sid = sid
        time_now = datetime.now()
        self.time_started = time_now
        self.last_active = time_now
        # the entire query history
        self.queries = []           # type: List[Query]
        # the last elasticsearch query as python dict
        self.last_es_query = None   # type: Dict[str, Any]
        # indicates, whether the last query was a follow-up to the previous query
        # (i.e. a query parameter has changed, but not the query terms)
        self.follow_up = False
        # the current topic centoid
        self.topic_centroid = TopicCentroid(topic_model)
        # suggestions that have been calculated for the most recent state of the topic centroid
        self.last_suggestions = None    # type: SearchResult

    def submit_request(self, search_request: SearchRequest) -> bool:
        """
        changes the current session according to the contents of the last request.
        usually, this means a new query will be added.
        if the current query is the same as the previous one (e.g. only the offset has changed),
        this query will replace the last, to keep the query history clean.
        :param search_request: the search request to evaluate
        :return: a boolean that indicates whether a new query has been added to the query history
                  (this is not the case for follow-up queries)
        """
        self.last_active = datetime.now()
        query = search_request.query
        # detect whether this is a new query
        if self.is_request_follow_up(search_request):
            # this is a follow-up to the previous query
            self.follow_up = True     # set marker
            self.queries[-1] = query  # replace the last one
        else:
            # this is a new query
            self.follow_up = False
            self.queries.append(query)
            logger.debug("query terms in session: " + str([q.query for q in self.queries[::-1]]))
        return not self.follow_up

    def submit_response(self, search_result: SearchResult):
        if not self.follow_up:
            if search_result.hits:
                self.topic_centroid.update(search_result)
            else:
                logger.warning("unable to update topic centroid in session {}: "
                               "there were no search results".format(self.sid))

    def current_query(self) -> Optional[Query]:
        return self.queries[-1] if self.queries else None

    def previous_query(self) -> Optional[Query]:
        return self.queries[-2] if len(self.queries) >= 2 else None

    def is_follow_up(self) -> bool:
        """
        :return: True, iff the current query is a follow-up of the previous one
                 (i.e. only the offset has changed, no new query terms)
        """
        return self.follow_up

    def is_request_follow_up(self, search_request: SearchRequest):
        """
        :param search_request: the search request to inspect
        :return: true, iff the specified request is a follow-up of the previous one
                  (determined by explicit setting of the follow-up argument or implicitly
                   if the query is the same as the previous one)
        """
        return self.queries and (search_request.follow_up or search_request.query.same_as(self.current_query()))

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class PersistentSession(Session):

    def __init__(self, sid: str, topic_model: TopicModel = None, step_storage_limit: int = 20):
        super().__init__(sid, topic_model)
        # the current step (unique identifier of the current state in this session)
        self.step = 0
        # the step counter, used to retrieve unique step IDs
        self.step_counter = 0
        # the step dictionary, contains the stored state of the last few steps
        self.step_dict = OrderedDict()
        # determines, how many of the previous steps will be preserved (older steps will be discarded)
        self.step_storage_limit = step_storage_limit

    def submit_request(self, search_request: SearchRequest) -> bool:
        # update timestamp and query
        self.last_active = datetime.now()
        query = search_request.query
        if query.step is None:
            query.step = self.step

        # decide if we need to restore a previous step
        if query.step != self.step and self.step_counter > 0:
            # the step parameter differs from the current step
            self.switch_step(query.step)

        # handle spell checking
        if search_request.spelling:
            # the user accepted a spelling suggestion
            # -> restore the previous step (removes the misspelled term from the query history)
            prev_step = self.queries[-1].step if self.queries else 0
            self.switch_step(prev_step)
            # replace the step id of the corrected query with the id of the misspelled query,
            # so we can return to the previous state when the user navigates in the query history
            query.step = prev_step

        # detect whether this is a new query
        if self.is_request_follow_up(search_request):
            # this is a follow-up to the previous query
            self.follow_up = True
            self.queries[-1].start = query.start
            self.queries[-1].size = query.size
        elif search_request.query.same_as(self.current_query(), ignore_date=True):
            # just the date filter has changed -> do not add a new element, but force re-evaluation
            self.follow_up = False
            self.queries[-1].date_before = query.date_before
            self.queries[-1].date_after = query.date_after
        else:
            # this is a new query
            self.persist_step()
            # mutate this state under a new id
            self.step = self.step_counter = self.step_counter + 1
            self.follow_up = False
            self.queries.append(query)
        return not self.follow_up

    def switch_step(self, step: int):
        """
        store the current state and switch to the specified step
        :param step: the step to switch to
        """
        if step in self.step_dict:
            # persist the current state
            self.persist_step()
            # switch to the previous state
            self.restore_step(step)
        else:
            # TODO show some kind of warning, if the step is not cached anymore
            # need to set up a noficiation scheme (can't just call alert from here)
            logger.warning("the requestet step {} was not found in this session. "
                           "maybe it has timed out".format(step))

    def persist_step(self):
        # persists the current state under the current step id.
        # create shallow copies of objects that are expected to be frequently mutated
        # so they won't change when the session state is updated
        state = {
            'step': self.step,
            'queries': self.queries.copy(),
            'last_es_query': self.last_es_query,
            'follow_up': self.follow_up,
            'topic_centroid': self.topic_centroid.centroid.copy(),
            'last_suggestions': self.last_suggestions
        }
        self.step_dict[self.step] = state
        # keep the cache tidy
        self.cleanup_storage()

    def restore_step(self, step: int, reset_query_state=True):
        # restore all objects exactly as they've been stored in persist_step()
        # create shallow copies of mutable objects so they don't temper with the storage state
        state = self.step_dict[step]
        self.step = state['step']
        self.queries = state['queries'].copy()
        self.last_es_query = state['last_es_query']
        self.follow_up = state['follow_up']
        self.topic_centroid.centroid = state['topic_centroid'].copy()
        self.last_suggestions = state['last_suggestions']
        if reset_query_state and self.queries:
            last_query = self.queries[-1]
            last_query.start = 0

    def cleanup_storage(self):
        step_dict = self.step_dict
        limit = self.step_storage_limit
        while len(step_dict) > limit:
            oldest_key = next(iter(step_dict.keys()))
            del step_dict[oldest_key]


class SessionManager:

    # TODO implement session timeout

    def __init__(self, session_timeout=1800, topic_model: TopicModel = None):
        super().__init__()
        self.sessions = {}  # type: Dict[str, PersistentSession]
        self.session_timeout = session_timeout
        self.topic_model = topic_model

    def add_session(self, sid: str) -> PersistentSession:
        session = PersistentSession(sid, topic_model=self.topic_model)
        self.sessions[sid] = session
        return session

    def get_session(self, sid: str) -> Optional[PersistentSession]:
        return self.sessions.get(sid)

    def has_session(self, sid: str) -> bool:
        return sid in self.sessions

    def get_or_create_session(self, sid: str) -> Optional[PersistentSession]:
        return self.sessions.get(sid) if (sid in self.sessions) else self.add_session(sid)

    def submit_request(self, search_request: SearchRequest) -> PersistentSession:
        if not search_request.sid:
            # note: users are always redirected before this can happen
            raise RuntimeError("no session id has been provided")
        session = self.get_or_create_session(search_request.sid)
        session.submit_request(search_request)
        return session

    def get_queries(self, sid: str) -> Optional[List[Query]]:
        session = self.get_session(sid)
        if session:
            return session.queries

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()
