# Reddit NLP Package [![Build Status](https://travis-ci.org/jaijuneja/reddit-nlp.svg?branch=master)](https://travis-ci.org/jaijuneja/reddit-nlp)

A lightweight Python module that performs tokenization and processing of text on Reddit. It allows you to analyze users, titles, comments and subreddits to understand their vocabulary. The module comes packaged with its own inverted index builder for storing vocabularies and word frequencies, such that you can generate and manipulate large corpora of tf-idf weighted words without worrying about implementation. This is especially useful if you're running scripts over long periods and wish to save intermediary results.

## License

Copyright 2014 Jai Juneja.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

## Installation

### Using pip or easy_install

You can download the latest release version using `pip` or `easy_install`:

```
pip install redditnlp
```

### Latest development version
You can alternatively download the latest development version directly from GitHub:

```
git clone https://github.com/jaijuneja/reddit-nlp.git
```

Change into the root directory:

```
cd reddit-nlp
```

Then install the package:

```
python setup.py install
```

### Error: the required version of setuptools is not available

Upon running `pip install` or the `setup.py` script you might get a message like this:

```
The required version of setuptools (>=0.7) is not available, and can't be installed while this script is running. Please install a more recent version first, using 'easy_install -U setuptools'.
```

This is appearing because you have a very outdated version of the setuptools package. The redditnlp package typically bootstraps a newer version of setuptools during install, but it isn't working in this case. You need to update setuptools using `easy_install -U setuptools` (you may need to apply `sudo` to this command).

If the above command doesn't do anything then it is likely that your version of setuptools was installed using a package manager such yum, apt or pip. Check your package manager for a package called python-setuptools or try `pip install setuptools --upgrade` and then re-run the install.

## Usage

A more complex sample program using the redditnlp module can be found at `https://github.com/jaijuneja/reddit-nlp/blob/master/example.py`. Here we outline a basic word counter application.

The module consists of three classes: 
 
* A basic word counter class, `WordCounter`, which performs tokenization and counting on input strings
* A Reddit word counter, `RedditWordCounter`, which extends the `WordCounter` class to allow interaction with the Reddit API
* A tf-idf corpus builder, which allows storing of large word corpora in an inverted index

These three classes can be instantiated as follows:

```python
from redditnlp import WordCounter, RedditWordCounter, TfidfCorpus

word_counter = WordCounter()
reddit_counter = RedditWordCounter('your_username')
corpus = TfidfCorpus()
```

To adhere to the Reddit API rules, it is asked that you use your actual Reddit username in place of `'your_username'` above.

For further information on the attributes and methods of these two classes you can run:

```python
help(WordCounter)
help(RedditWordCounter)
help(TfidfCorpus)
```

Next, we can tokenize 1000 comments from a selection of subreddits, extract the most common words and save all of our data to disk:

```python
for subreddit in ['funny', 'aww', 'pics']:
    # Tokenize and count words for 1000 comments
    word_counts = counter.subreddit_comments(subreddit, limit=1000)
    
    # Add the word counts to our corpus
    corpus.add_document(word_counts, subreddit)

# Save the corpus to a specified path (must be JSON)
corpus.save(path='word_counts.json')

# Save the top 50 words (by tf-idf score) from each subreddit to a text file
for subreddit in corpus.get_document_list():
    top_words = corpus.get_top_terms(document, num_terms=50)
    with open('top_words.txt', 'ab') as f:
        f.write(document + '\n' + '\n'.join(top_words.keys()))
```

### Machine learning

`redditnlp` now supports some of scikit-learn's machine learning capability. Several in-built functions enable the user to:

* Convert a TfidfCorpus object into a scipy sparse feature matrix (using `build_feature_matrix()`)
* Train a classifier using the documents contained in a TfidfCorpus (with `train_classifier()`) and thereafter classify new documents (with `classify_document()`)

Below is an example of a simple machine learning application that loads a corpus of subreddit comment data, uses it to train a classifier and determines which subreddit a user's comments most closely match:

```python
# Load the corpus of subreddit comment data and use it to train a classifier
corpus = TfidfCorpus('path/to/subreddit_corpus.json')
corpus.train_classifier(classifier_type='LinearSVC', tfidf=True)

# Tokenize all of your comments
counter = RedditWordCounter('your_username')
user_comments = counter.user_comments('your_username')

# Classify your comments against the documents in the corpus
print corpus.classify_document(user_comments)
```

### Multiprocessing

`redditnlp` uses the [PRAW](https://github.com/praw-dev/praw) Reddit API wrapper. It supports multiprocessing, such that you can run multiple instances of `RedditWordCounter` without exceeding Reddit's rate limit. There is more information about this in the [PRAW documentation](https://praw.readthedocs.org/en/latest/pages/multiprocess.html) but for the sake of completeness an example is included below.

First, you must initialise a request handling server on your local machine. This is done using the terminal/command line:

```
praw-multiprocess
```

Next, you can instantiate multiple `RedditWordCounter` objects and set the parameter `multiprocess=True` so that outgoing API calls are handled:

```
counter = RedditWordCounter('your_username', multiprocess=True)
```

## Contact

If you have any questions or have encountered an error, feel free to contact me at `jai -dot- juneja -at- gmail -dot- com`.