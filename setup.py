#!/usr/bin/env python
from setuptools import setup

setup(
    name='epic-search',
    version='1.0.0',
    description='a session-based semantic search engine',
    author='Sebastian Straub',
    author_email='sstraub (at) posteo (dot) de',
    url='https://github.com/Klamann/epic-search',
    license='Apache 2.0',
    packages=['search_ui'],
    package_dir={'search_ui': 'search_ui'},
    package_data={'search_ui': ['search_ui/res/*.conf']},
    zip_safe=True,
    classifiers=(
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ),
    install_requires=[
        'elasticsearch>=5.3.0,<6.0.0',
        'flask>=0.12.0,<0.13.0',
        'Flask-Assets>=0.12.0,<0.13.0',
        'jsmin>=2.0',
        'cssmin>=0.2',
        'pyScss>=1.0',
        'python-dateutil>=2.0',
        'pyhocon>=0.3',
        'CyHunspell>=1.2.0',
        'editdistance>=0.3',
    ],
    tests_require=[
        'requests>=2.0'
        'lxml>=3.0'
    ]
)
