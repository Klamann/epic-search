# epic-search

*epic search* is the prototype of a session-based semantic search engine.

Session-based search means that we include previous queries from the user's query history into an expanded search query, so that we can interpret the user's requests in context. *Semantic* means that we don't just use keyword matching to get search results, but also rely on probabilistic topic models to find relevant search results. Our goal is to support the user in complex exploratory search tasks.

The concepts are described in detail in the thesis *Towards Collaborative Session-based Semantic Search* (soon to be published).

The prototype of the search engine consists of a web application written in Python 3 that uses the micro web framework [Flask](http://flask.pocoo.org/) to process queries and serve search results.

## Setup

To install the application with all required dependencies:

    python3 setup.py install --user

Now it can be started with

    python3 run.py

Point your browser to <http://localhost:1337> to see the homepage of the search engine. You won't get any search results until a search index is up and running (see next section).

You may specify a configuration file using the `--config` parameter that overrides the default options defined in [`search_ui/res/base.conf`](./search_ui/res/base.conf)

### Search Index

Please refer to the documentation of the [search index builder](https://github.com/Klamann/search-index-builder) to build your own search index.

If you have a working search index, make sure Elasticsearch is running on the same machine on port 9200 or specify `elastic.host` and `elastic.port` in the configuration.

There is one more thing you need for the search engine to work: a topic model. You can get it from the [search index builder](https://github.com/Klamann/search-index-builder), it is stored in a file named `*-topics.json`, where `*` is the prefix you pass to the topic model script. By default, it is expected to be placed in `data/topics.json`, but you can adjust the path in the config by specifying another path under the key `data.topics.file`.

This topic model will be parsed on first launch, which can take about a minute, but then it is cached in a binary format, so the time to launch the search engine with a cached topic model should never exceed a few hundret milliseconds.

### Troubleshooting

On some distributions, additional packages might be required that are not installed by default. E.g. on Debian:

    apt install python3-dev python3-pip hunspell

## License

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

Different licensing conditions may apply to bundled libraries and web fonts.
