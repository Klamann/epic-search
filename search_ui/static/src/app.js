// initialize zurb foundation
$(document).foundation();

// initialize localforage (for storage of data beyond browser sessions)
localforage.config({ name: 'arxiv-search-ui' });

// register String.format() - see https://stackoverflow.com/a/4673436
if (!String.prototype.format) {
  String.prototype.format = function() {
    let args = arguments;
    return this.replace(/{(\d+)}/g, function(match, number) {
      return typeof args[number] != 'undefined' ? args[number] : match;
    });
  };
}

// jquery addons
jQuery.fn.fadeOutAndRemove = function(speed) {
  $(this).fadeOut(speed,function() {
    $(this).remove();
  });
};
jQuery.fn.appendAndFadeIn = function(content, speed) {
  $(content).hide().appendTo($(this)).fadeIn(speed);
};


$(function () {

  // defaults
  let readmoreDefaultProps = {
    speed: 300,
    collapsedHeight: 300,
    moreLink: '<a class="readmore r-expand" href="#"><i class="fa fa-caret-down" aria-hidden="true"></i></a>',
    lessLink: '<a class="readmore r-collapse" href="#"><i class="fa fa-caret-up" aria-hidden="true"></i></a>'
  };

  // show the footer, if javascript is enabled
  $('#footer').show();

  // hide long sidebar entries
  let suggestionsList = $('#suggestions-list');
  suggestionsList.readmore(readmoreDefaultProps);
  let topicListBtn = $('#topic-centroid-list');
  let topicList = topicListBtn.find("ul.topic-list");
  topicList.readmore(readmoreDefaultProps);
  $('#topic-centroid-list-label').click(function(e) {
    setTimeout(function() {
      topicList.readmore(readmoreDefaultProps);
    }, 10);
  });

  // reset filters on button click
  $('button.filter-reset').click(function(e) {
    setTimeout(function() {
      $('input.filter-input').val("");
    }, 10);
  });


  // infinite scroll

  // list of document ids in sidebar
  let docsInSidebar = suggestionsList.find('.result-entry').map(function() {
    return $(this).data('docId');
  }).get();
  // make button invisible, if javascript is enabled, and add loading indicator
  $('#results-more').hide();
  let loading = $('#results-loading');
  // Each time the user scrolls
  let working = false;
  let enabled = true;
  let win = $(window);
  win.scroll(function () {
    // End of the document reached?
    if ((win.scrollTop() + 5) >= ($(document).height() - win.height()) && enabled && !working) {
      working = true;
      loading.css('visibility', 'visible');
      params = [];
      if ('query' in query)
        params.push('q=' + query.query);
      if ('author' in query)
        params.push('author=' + query.author);
      if ('date_before' in query)
        params.push('date-before=' + query.date_before);
      if ('date_after' in query)
        params.push('date-after=' + query.date_after);
      if (sid)
        params.push('sid=' + sid);
      if (step)
        params.push('step=' + step);
      params.push('start=' + ($("#result-list > li.result-entry").length + ('start' in query ? query.start : 0)));
      params.push('size=10');
      params.push('follow-up=true')
      $.ajax({
        url: "/search?" + params.join("&"),
        dataType: 'html',
        success: function (html_string) {
          let html = $.parseHTML(html_string);
          let results = $('#result-list .result-entry', html);
          if (results.length > 0) {
            // append results
            for (let result of results) {
              let elem = $(result);
              $('#result-list').append(elem);
              // hide the result if it's already in the sidebar
              if (docsInSidebar.indexOf(elem.data('docId')) >= 0) {
                elem.hide();
              }
              // check star state
              let docId = elem.data("docId");
              isBookmarkedPromise(docId).then(function(isBookmarked) {
                if (isBookmarked) {
                  elem.find(".result-star").addClass("starred");
                }
              });
            }
          } else {
            // display error message & disable ajax
            $('#result-list').after($('#result-failure', html));
            enabled = false;
            $('#footer > p').first().text("You've reached the end of the result list.")
          }
        },
        error: function (response) {
          console.log(response);
        },
        complete: function() {
          loading.css('visibility', 'hidden');
          working = false;
        }
      });
    }
  });



  // reduce the amount of displayed breadcrumbs to avoid overflow

  let breadcrumbs = $("#query-history");
  let allowed_height = parseInt(breadcrumbs.find("a").css('line-height')) * 1.1;
  let reduced = false;

  function fitToWidth() {
    // reset
    if (reduced) {
      breadcrumbs.find("a").first().remove();
      breadcrumbs.find("a").show();
      reduced = false;
    }
    // reduce, if necessary
    if (breadcrumbs.height() > allowed_height) {
      //console.log("hiding breadcrumbs to avoid overflow...");
      breadcrumbs.prepend('<a href="#">...</a>');
      reduced = true;
      let breadcrumbSize = breadcrumbs.children().length;
      // indizes: skip first (...) and last (must always remain, even if line breaks)
      for (let i = 1; i < breadcrumbSize-1; i++) {
        breadcrumb = breadcrumbs.find("a").eq(i);
        breadcrumb.hide();
        //console.log("hiding " + i + ": " + breadcrumb.text());
        if (breadcrumbs.height() <= allowed_height)
          break;
      }
    }
  }

  // adjust on resize
  $(window).resize(function() {
    fitToWidth()
  });

  fitToWidth();



  // save search results in localStorage

  let bmPersonal = $('#bookmarks-personal');
  let bmpList = bmPersonal.find('.bookmark-list').first();
  let bmpHintEmpty = bmPersonal.find(".hint-empty");
  let bmpButtons = bmPersonal.find(".bookmark-buttons");

  // handle click
  $(document).click(function(e) {
    let target = $(e.target);
    let targetParent = target.parent();
    if (targetParent.hasClass("result-star")) {
      // star icon clicked
      e.preventDefault();
      let result = targetParent.closest('.result-entry');
      let docId = result.data('docId');
      if (targetParent.hasClass("starred")) {
        // remove bookmark
        //console.log("removing {0} from the bookmark list".format(docId));
        resultRemove(docId);
      } else {
        // add bookmark
        //console.log("adding {0} to the bookmark list".format(docId));
        let docTitle = result.data('title');
        let docYear = result.data('year');
        let docUrl = result.data('url');
        let docAuthors = result.data('authors').split("|");
        resultAdd(docId, docTitle, docYear, docUrl, docAuthors)
      }
      targetParent.toggleClass("starred");
    } else if (target.hasClass("bm-remove")) {
      // remove button in bookmark list clicked
      e.preventDefault();
      let bookmark = target.closest('.bookmark');
      let docId = bookmark.data('docId');
      //console.log("removing " + docId);
      resultRemove(docId);
      setStarState(docId, false);
    }
  });

  // make the bookmarks sortable & handle change order
  if (bmpList.length) {
    sortable(bmpList)[0].addEventListener('sortupdate', function(e) {
      saveBookmarkList();
    });
  }

  // button clear all
  $('#clear-bookmarks').click(function(e) {
    // clear local storage
    localforage.getItem('bookmarks.personal').then(function(bookmarkList) {
      bookmarkList.forEach(function (docId, i) {
        localforage.removeItem('doc.' + docId).catch(function(err) {
          console.log(err);
        });
      });
      localforage.setItem('bookmarks.personal', []).catch(function(err) {
        console.log(err);
      });
    }).catch(function(err) {
      console.log(err);
    });

    // clear html list
    bmpList.empty();
    bmpButtons.hide();
    bmpHintEmpty.show();
    $(".result-star").removeClass("starred");
    updateListState();
  });

  // add the result to the list
  function resultAdd(id, title, year, url, authors) {
    // create new element for the bookmark list
    let authorShort = authors.length > 1 ? (authors[0] + " et al.") : authors[0];
    let elem = buildBookmarkListItem(id, title, year, url, authorShort);

    // show/hide other elements as needed, then add bookmark to list
    if (bmpList.find("li").length === 0) {
      // list was empty -> hide notification and display delete button
      bmpHintEmpty.hide();
      bmpButtons.show();
    }
    bmpList.appendAndFadeIn(elem, 200);
    updateListState();

    // store document metadata and the bookmark list locally
    let storage = {'id': id, 'title': title, 'year': year, 'url': url, 'author': authorShort};
    let p = localforage.setItem('doc.' + id, storage).then(function () {
      saveBookmarkList();
    }).catch(function(err) {
      console.log(err);
    });
    Promise.all([p]);
  }

  function buildBookmarkListItem(id, title, year, url, author) {
    return '<li id="b-{0}" class="bookmark personal" data-doc-id="{0}"><button class="bm-remove close-button" aria-label="remove bookmark" type="button">×</button><span class="author">{4}</span>: <a href="{3}" target="_blank"><span class="title">{1}</span></a> (<span class="year">{2}</span>)</li>'
      .format(id, title, year, url, author);
  }

  // remove the result from the list
  function resultRemove(docId) {
    // note that the short id selector fails when there are dots in the id, as it is usual for axiv doc ids...
    $('[id="b-' + docId + '"]').fadeOutAndRemove(200);
    if (bmpList.find("li").length == 0) {
      // list became empty -> show notification and hide delete button
      bmpHintEmpty.show();
      bmpButtons.hide();
    }
    updateReadmore(200);

    let p = localforage.removeItem('doc.' + docId).then(function() {
      saveBookmarkList();
    }).catch(function(err) {
      console.log(err);
    });
    Promise.all([p]);
  }

  function getBookmarkPromise(docId) {
    return localforage.getItem('doc.' + docId);
  }

  function isBookmarkedPromise(docId) {
    return getBookmarkPromise(docId).then(function(v) { return v !== null });
  }

  function saveBookmarkList() {
    //console.log("updating bookmark list in local storage");
    let bookmarkList = [];
    bmpList.find("li.bookmark").each(function() {
      let docId = $(this).data('docId');
      bookmarkList.push(docId);
    });
    let p = localforage.setItem('bookmarks.personal', bookmarkList).catch(function(err) {
      console.log(err);
    });
    Promise.all([p]);
  }

  function restoreBookmarkList() {
    let p = localforage.getItem('bookmarks.personal').then(function(bookmarkList) {
      // clear existing elements
      bmpList.empty();
      bmpHintEmpty.show();
      bmpButtons.hide();
      // fill with new elements
      if (bookmarkList === null || bookmarkList.length === 0) {
        //console.log("no bookmarks to restore");
      } else {
        //console.log("restoring " + bookmarkList.length + " bookmark(s)");
        bookmarkList.forEach(function (docId, i) {
          // lookup the actual metadata for each document and add it to the list
          let p = localforage.getItem('doc.' + docId).then(function(data) {
            let elem = buildBookmarkListItem(data.id, data.title, data.year, data.url, data.author);
            bmpList.append(elem);
            setStarState(docId, true);
          }).catch(function(err) {
            console.log(err);
          });
          Promise.all([p]);
        });
        bmpHintEmpty.hide();
        bmpButtons.show();
        updateListState(true);
      }
    }).catch(function(err) {
      // This code runs if there were any errors
      console.log(err);
    });
    Promise.all([p]);
  }

  function setStarState(docId, enabled) {
    let star = $('[id="d-' + docId + '"]').find(".result-star");
    if (enabled) {
      star.addClass("starred");
    } else {
      star.removeClass("starred");
    }
  }

  // re-evaluate size and sortable state
  function updateListState(forceClosed) {
    makeSortable();
    updateReadmore(0, forceClosed);
  }

  function makeSortable() {
    // evil hack, but I haven't found a better way to wait for the dom to update
    // MutationObserver is not useful in this case, because the sortable lib mutates the dom all the time...
    setTimeout(function() {
      sortable(bmpList);
    }, 100);
  }

  function updateReadmore(extraDelay, forceClosed) {
    setTimeout(function() {
      let props = readmoreDefaultProps;
      let expanded = bmpList.height() > 300;
      // if list is large and forceClosed is false (or undefined), leave it open
      if (expanded && (typeof forceClosed === 'undefined' || !forceClosed)) {
        props = $.extend({startOpen: true}, readmoreDefaultProps);
      }
      bmpList.readmore(props);
    }, typeof extraDelay === 'undefined' ? 100 : 100 + extraDelay);
  }

  // intialize: read from localstorage (if bookmark list container exists)
  if (bmpList.length) {
    restoreBookmarkList();
  }

});

// text effect
$(function () {

  lettersFadeIn($('#search-heading').find('.version').first(), 7, 2);

  function lettersFadeIn(element, delay, repeat) {
    let elem = $(element);
    let text = elem.text();
    if (typeof delay === 'undefined')
      delay = 10;
    if (typeof repeat === 'undefined')
      repeat = 3;
    editCharTimeout(elem, text, repeat, delay, new Array(text.length).fill(0), repeat, 0);
  }

  function editCharTimeout(elem, text, repeatNew, delay, fixedAr, repeat, fixedCount) {
    if (repeat-1 === 0 && fixedCount+1 === text.length) {
      elem.text(text);
    } else {
      repeat -= 1;
      if (repeat <= 0) {
        let offset = Math.floor(Math.random() * fixedAr.length);
        for (let i=0; i<fixedAr.length; i++) {
          let idx = (i + offset) % fixedAr.length;
          if (fixedAr[idx] === 0) {
            fixedAr[idx] = 1;
            break;
          }
        }
        repeat = repeatNew;
        fixedCount += 1;
      }

      let partlyRandomString = fixedAr.map(function(val, i) {
        if (val === 0) {
          return randChar();
        } else {
          return text[i];
        }
      }).join("");

      elem.text(partlyRandomString);
      setTimeout(function() {
        editCharTimeout(elem, text, repeatNew, delay, fixedAr, repeat, fixedCount)
      }, delay);
    }
  }

  function randChar() {
    return (Math.random()+1).toString(36).slice(2,3);
  }

});

// dropdown
$(function () {
  // store last state
  let lastDropdownButton;
  let lastDropdownContent;
  // actual click event handler
  $(document).click(function(e) {
    let target = $(e.target)
    if (target.is(".dropdown")) {
      e.preventDefault();
      if (lastDropdownContent != undefined && target[0] == lastDropdownButton[0]) {
        // same button, just toggle
        lastDropdownContent.toggle();
      } else {
        // different button. close last dropdown and display the new one
        if (lastDropdownContent != undefined) {
          lastDropdownContent.hide();
        }
        lastDropdownContent = target.find(".dropdown-content");
        lastDropdownContent.show();
        lastDropdownButton = target;
      }
    } else if (target.is(".dropdown-content")) {
      // keep it open
    } else {
      // hide the content
      if (lastDropdownContent != undefined) {
        lastDropdownContent.hide();
      }
    }
  })
});

// copy stuff to clipboard
$(function () {

  // copy bookmarks to the clipboard
  new Clipboard('.bookmarks-copy', {
    text: function(trigger) {
      btn = $(trigger);
      // animate the button so we see something happened
      btn.fadeTo(50, 0.5, function() { $(this).fadeTo(200, 1.0); });
      bookmarkList = btn.closest('.tabs-panel').find('.bookmark-list');
      return bookmarksToCsv(bookmarkList);
    }
  });
  
  function bookmarksToCsv(bookmarkList) {
    let csv = "id\ttitle\tauthor\tyear\tlink\n";
    for (let bookmark of bookmarkList.find('.bookmark')) {
      let bm = $(bookmark);
      let id = bm.data("docId");
      let title = bm.find(".title").text();
      let author = bm.find(".author").text();
      let year = bm.find(".year").text();
      let url = bm.find("a").attr("href");
      csv += "{0}\t{1}\t{2}\t{3}\t{4}\n".format(id, title, author, year, url);
    }
    return csv;
  }
});

// topic graph
$(function () {

  // settings
  let aspectRatio = 1.45;
  let yMulti = Math.round((1.0 / aspectRatio) * 1000) / 1000;
  let width = 1000;
  let height = width * yMulti;

  // find data
  if (typeof topicGraph === 'undefined') {
    if ($('#result-sidebar').length)
      console.log("topic graph was not found");
    return;
  }
  let jsonData = topicGraph;

  // hide the notice
  d3.select("#topic-centroid-graph > p").attr("style", "display: none;");

  // define the svg
  let svg = d3.select("#topic-centroid-graph").append("svg")
    .attr("width", "100%")
    .attr("viewBox", "0 0 1000 " + 1000*yMulti)
    .attr("preserveAspectRatio", "xMidYMin");

  let force = d3.layout.force()
    .linkDistance(150)
    .linkStrength(0.1)
    .friction(0.93)
    .charge(-1500)
    .chargeDistance(500)
    .gravity(0.03)
    .size([width, height]);

  let drag = force.drag()
    .on("dragstart", dragStart);

  // adjust y positions according to the selected aspect ratio
  jsonData.nodes.forEach(function(d) {
    d.y = Math.round(d.y * yMulti)
  });

  force
    .nodes(jsonData.nodes)
    .links(jsonData.links)
    .start();

  let link = svg.selectAll(".link")
    .data(jsonData.links)
    .enter().append("line")
    .attr("class", "link");

  let node = svg.selectAll(".node")
    .data(jsonData.nodes)
    .enter().append("g")
    .attr("class", "node")
    .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
    .call(force.drag);

  node.append("circle")
    .attr({"r":45})
    .style("fill",function(d,i){return "#ffbcb3";});

  // name
  node.append("text")
    .attr("dy", -5)
    .attr("text-anchor", "middle")
    .attr("fill", "#fff")
    .attr("font-size", 30)
    .attr("font-weight", "bold")
    .classed("name", true)
    .text(function(d) {
      // shorten the name
      let idx = d.name.indexOf("-");
      return d.name.slice(0, idx > 0 ? 2 : 4);
    });

  // sub-name (optional)
  node.append("text")
    .attr("dy", 20)
    .attr("text-anchor", "middle")
    .attr("fill", "#fff")
    .attr("font-size", 22)
    .classed("name", true)
    .text(function(d) {
      // shorten the name
      let idx = d.name.indexOf("-");
      return idx > 0 ? d.name.slice(idx+1) : null;
    });

  for (let i=0; i<6; i++) {
    let xOffset = (i%3) === 1 ? 55 : 45;
    let dx = i<3 ? xOffset : -xOffset;
    let dy = -26 + (i%3) * 35;
    node.append("text")
      .attr("dx", dx)
      .attr("dy", dy)
      .attr("text-anchor", i<3 ? "start" : "end")
      .classed("token", true)
      .text(function(d) {
        if (d.tokens instanceof Array && d.tokens.length > i) {
          let text = d.tokens[i];
          return text.length > 12 ? text.slice(0,10) + "…" : text;
        } else {
          return null;
        }
      });
  }

  force.on("tick", function(e) {
    let k = 6 * e.alpha;

    jsonData.links.forEach(function(d, i) {
      d.source.y -= k;
      d.target.y += k;
    });

    link.attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });

    node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
  });

  function dragStart(d) {
    d.fixed = true;
    d3.select(this).classed("fixed", true);
  }

});



