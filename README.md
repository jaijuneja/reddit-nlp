# Reddit NLP Package

A lightweight Python module that performs tokenization and processing of text on Reddit. It allows you to analyze users, titles, comments and subreddits to understand their vocabulary. The module comes packaged with its own inverted index builder for storing vocabularies and word frequencies, such that it can generate large corpora of tf-idf weighted words. This means that you don't have to worry about storing and reading word counts if you're running scripts over long periods.

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

### Usage:

A more complex sample program using the redditnlp module can be found at `https://github.com/jaijuneja/reddit-nlp/blob/master/example.py`. Here we outline a basic word counter.

The module consists of two classes: a Reddit word counter and a tf-idf corpus builder. These can be instantiated as follows:

```python
from redditnlp import RedditWordCounter, TfidfCorpus

counter = RedditWordCounter('your_username')
corpus = TfidfCorpus()
```

To adhere to the Reddit API rules, it is asked that you use your actual Reddit username in place of `'your_username'` above.

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