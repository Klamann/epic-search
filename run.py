import argparse

from pyhocon import ConfigFactory
from pyhocon import ConfigTree

from search_ui import app


if __name__ == "__main__":
    # parse arguments and launch the application
    parser = argparse.ArgumentParser(description='arXiv search ui - a search engine '
                                                 'built with elasticsearch and flask.')
    parser.add_argument('-c', '--config', help='path to config file (HOCON format)')
    parser.add_argument('-d', '--debug', action='store_const', const=True, default=False,
                        help='enable debug mode (overrides config)')
    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.config) if args.config else ConfigTree()
    if args.debug:
        conf = ConfigFactory.from_dict({'webserver': {'debug': True}}).with_fallback(conf)
    app.main(conf)
