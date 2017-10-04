"""

Copyright 2016-2017 Sebastian Straub

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""


class ClientError(Exception):
    """
    the client input was somehow invalid.
    This should result in an HTTP 400 or some kind of graceful error handling
    """


# module contents
from . import util
from .spelling import Spelling, SpellChecker
from .data import Query, SearchRequest, SearchHit, SearchResult, Topic, TopicModel, TopicCentroid
from .session import Session, PersistentSession, SessionManager
from .elastic import EsBridge
from .engine import SearchEngine
from . import app
