import bz2
import gzip
import hashlib
import itertools
import json
import logging
import random
import string
import sys
from datetime import date
from typing import Optional, IO, Dict, Any, Set, Iterable, T, Union, List

import dateutil.parser
import pkg_resources
from flask import Request, escape
from markupsafe import Markup

whitespace_and_punctuation = (string.whitespace + string.punctuation.replace('<', '').replace('>', '') + '…')


def json_read(file):
    with open_by_ext(file, 'rt', encoding='utf-8') as fp:
        return json.load(fp)


def open_by_ext(filename, mode='r', **kwargs) -> IO:
    if filename.endswith('.bz2'):
        return bz2.open(filename, mode=mode, **kwargs)
    elif filename.endswith('.gz'):
        return gzip.open(filename, mode=mode, **kwargs)
    else:
        return open(filename, mode=mode, **kwargs)


def read_resource(file: str) -> str:
    return pkg_resources.resource_string(__name__, 'res/' + file).decode(errors='ignore')


def smart_truncate(content: str, length: int = 100, suffix: str = '',
                   strip_punctuation: bool = False) -> str:
    """
    truncate a text to a specified maximum width, splitting at word boundaries.
    this means that the string have the specified max length and not end in the middle of a word.
    :param content: the content to truncate
    :param length: the max. length to truncate to
    :param suffix: an optional text to append to the truncated text (will only be appended, if the text is > length)
    :param strip_punctuation: remove leading and trailing punctuation from the result
    :return: the truncated text
    """
    add_suffix = False
    if strip_punctuation:
        content = content.strip(whitespace_and_punctuation)
    if length and len(content) > length:
        add_suffix = True
        content = content[:length].rsplit(' ', 1)[0] + suffix
    if strip_punctuation:
        content = content.strip(whitespace_and_punctuation)
    if add_suffix:
        content += suffix
    return content


def smart_truncate_l(content, begin_index=0, prefix='') -> str:
    """
    truncate a text from the left, splitting at word boundaries.
    this means that the start of the text may be removed.
    :param content: the content to truncate
    :param begin_index: the string index to begin at. If the character at the index is not a whitespace,
                        this function will seek the next whitespace and split there
    :param prefix: an optional text to prepend to the truncated text (will only be added, if begin_index > 0)
    :return: the truncated text
    """
    if begin_index <= 0:
        return content
    else:
        splt = content[begin_index:].split(' ', 1)
        return prefix + (splt[1] if len(splt) > 1 else content[begin_index:])


def escape_html(html_string: str, whitelist: List[str] = None) -> Markup:
    """
    escapes the html string and the tries to restore the elements in the specified whitelist.
    works only for tags without properties, e.g. <p> is fine, but not <p class="foo">.
    :param html_string: the html string to escape
    :param whitelist: a list of tags that should not be escaped
    :return: the escaped html string
    """
    escaped = str(escape(html_string))
    if whitelist:
        for tag in whitelist:
            escaped = escaped \
                .replace('&lt;'+tag+'&gt;', '<'+tag+'>') \
                .replace('&lt;/'+tag+'&gt;', '</'+tag+'>')
    return Markup(escaped)


def random_id(length=12) -> str:
    """
    generates a random string using the url-safe base64 characters
    (ascii letters, digits, hyphen, underscore).
    therefore, each letter adds an entropy of 6 bits (if SystemRandom can be trusted).
    :param length: the length of the random string, defaults to 12
    :return: a random string of the specified length
    """
    rand = random.SystemRandom()
    base64_chars = string.ascii_letters + string.digits + '-_'
    return ''.join(rand.choice(base64_chars) for _ in range(length))


def parse_date(date_string: str, default: date = date(2017, 1, 1)) -> Optional[date]:
    """
    parses the specified date using the dateutil library.
    a variety of formats is supported, but it is recommended to stick with ISO date format...
    :param date_string: the string to parse
    :param default: replace missing parts of the date by values from this date
    :return: the parsed date or None, if it can't be parsed
    """
    if date_string:
        try:
            return dateutil.parser.parse(date_string, default=default)
        except ValueError:
            return None


def str_blank_to_none(s: str) -> Optional[str]:
    """
    :return: the specified string, or None, if it is whitespace only
    """
    if s: return s.strip() or None


def flatten(iterable: Iterable[Iterable[T]], generator=False) -> Union[List[T], Iterable[T]]:
    """flattens a sequence of nested elements"""
    flat = itertools.chain.from_iterable(iterable)
    return flat if generator else list(flat)


def class_str_from_dict(class_name: str, d: Dict[str, Any]) -> str:
    return "{}({})".format(class_name, ", ".join(k+"="+str(v) for k,v in d.items()))


def file_checksum(fname):
    """
    calculates the md5 checksum of a file and returns it as hex string (0-f).
    warning: reads the entire file into memory. see https://stackoverflow.com/a/3431835
    in case you need a memory-efficient variant...
    """
    return hashlib.md5(open(fname, 'rb').read()).hexdigest()


def logging_setup(logger_name: str, format='{asctime} [{levelname}]: {message}', style='{',
                  logfile=False, logfile_path=None, logfile_level='DEBUG',
                  stdout=True, stdout_level='INFO') -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        print("warning: logger {} already has {} handlers. They will be removed now to prevent duplicate log output."
              .format(logger_name, len(logger.handlers)))
        logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False   # handle duplicate output
    formatter = logging.Formatter(fmt=format, style=style)
    if stdout:
        lvl = stdout_level if isinstance(stdout_level, int) else logging._nameToLevel.get(stdout_level, logging.INFO)
        ch = ConsoleHandler()
        ch.setLevel(lvl)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if logfile and logfile_path:
        fh = logging.FileHandler(logfile_path)
        fh.setLevel(logfile_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


class ConsoleHandler(logging.StreamHandler):
    """
    A handler optimized for console output:
    - logs to stdout by default, only errors go to stderr
    - flushes output after every log action to avoid mixed logs...
    """

    def __init__(self):
        super().__init__(self)
        self.duplicates = set()     # type: Set[int]

    def emit(self, record: logging.LogRecord):
        try:
            if record.__dict__.get('deduplicate', False):
                msg_hash = record.getMessage().__hash__()
                if msg_hash in self.duplicates:
                    return
                else:
                    self.duplicates.add(msg_hash)
        except RuntimeError as e:
            print('logging error: ' + str(e), file=sys.stderr)
        self.stream = sys.stderr if record.levelno >= logging.ERROR else sys.stdout
        logging.StreamHandler.emit(self, record)

    def flush(self):
        # Workaround a bug in logging module, see http://bugs.python.org/issue6333
        if self.stream and hasattr(self.stream, 'flush') and (not self.stream.closed if hasattr(self.stream, 'closed') else False):
            logging.StreamHandler.flush(self)


class PreferredMime:
    """
    discover the client's preferred mime type.
    Currently, html and json are available, html is preferred.
    """

    supported_types = ['text/html', 'application/json']

    def __init__(self, request: Request):
        super().__init__()
        self.preferred_type = request.accept_mimetypes.best_match(self.supported_types)
        self.pref_json = self.preferred_type == 'application/json'
        self.pref_html = self.preferred_type == 'text/html'
