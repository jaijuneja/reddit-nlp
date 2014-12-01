# Reddit NLP Package

A lightweight Python module that performs tokenization and processing of text on Reddit. It allows you to analyze users, titles, comments and subreddits to understand their vocabulary. The module comes packaged with its own inverted index builder for storing vocabularies and word frequencies, such that it can generate large corpora of tf-idf weighted words. This means that you don't have to worry about storing and reading word counts if you're running scripts over long periods.

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
The required version of setuptools (>=7.0) is not available, and can't be installed while this script is running. Please install a more recent version first, using 'easy_install -U setuptools'.
```

This is appearing because you have a very outdated version of the setuptools package. The redditnlp package typically bootstraps a newer version of setuptools during install, but it isn't working in this case. You need to update setuptools using `easy_install -U setuptools` (you may need to apply `sudo` to this command).

If the above command doesn't do anything then it is likely that your version of setuptools was installed using a package manager such yum, apt or pip. Check your package manager for a package called python-setuptools or try `pip install setuptools --upgrade` and then re-run the install.

## Usage:

A more complex sample program using the redditnlp module can be found at `https://github.com/jaijuneja/reddit-nlp/blob/master/example.py`. Here we outline a basic word counter application.

The module consists of two classes: a Reddit word counter and a tf-idf corpus builder. These can be instantiated as follows:

```python
from redditnlp import RedditWordCounter, TfidfCorpus

counter = RedditWordCounter('your_username')
corpus = TfidfCorpus()
```

To adhere to the Reddit API rules, it is asked that you use your actual Reddit username in place of `'your_username'` above.

For further information on the attributes and methods of these two classes you can run:

```python
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

## Contact

If you have any questions or have encountered an error, feel free to contact me at `jai -dot- juneja -at- gmail -dot- com`.