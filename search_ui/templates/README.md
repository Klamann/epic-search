# Template Documentation

The `templates` directory contains all templates that can be called directly by `flask.render_template()`.

Under `templates/partials` are all snippets that can be embedded into a template using the `include` statement, e.g. `{% include 'partials/topic-list.html' %}`.

## Templates

The templates are hierarchically organized, where templates that are lower in the hierarchy inherit from their parents and extend these templates.

* `layout.html`: the basic template for all public html pages
    - `search-startpage.html`: the start page of the search UI
    - `search-results.html`: the result page for a query
* `search-followup.html`: a stripped-down version of the result page, intended to be used by the ajax result loader.

## Partials

* `serp-list.html`: generates a formatted result list from a list of search result objects
* `serp-sidebar.html`: defines the entire sidebar of the search result page
* `topic-list.html`: generates a pretty html list from a list of topic objects
