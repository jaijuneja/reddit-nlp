from __future__ import division
from __future__ import print_function

import os
import math
import errno
import json
import operator
import numpy as np
import praw
import urllib2
import nltk

from nltk.stem.porter import PorterStemmer
from collections import Counter, OrderedDict
from time import time, sleep
from string import punctuation
from praw.handlers import MultiprocessHandler


class RedditWordCounter(object):
    """Performs word counting of comments and titles in Reddit using the Reddit API.

    To initialise a new RedditWordCounter instance:
    >>> counter = RedditWordCounter('your_username')

    To adhere to the Reddit API rules, please provide your Reddit username in place of 'your_username' above.
    This will ensure that the app doesn't get banned from Reddit!

    Data Attributes:
        user_agent (str): required to connect to Reddit
        reddit: instance of the Reddit API connection
        stemmer: Porter stemmer used optionally to perform stemming of extracted words
        stopwords (list): list of stop words used to reject common words such as 'and'

    Methods:
        subreddit_comments: word count from comments of a given subreddit
        subreddit_titles: word count from titles of a given subreddit
        user_comments: word count from comments of a given user
        get_word_count: return tokenized word counts given an input string
        remove_punctuation
        remove_stopwords
        stem_tokens: perform Porter stemming on a list of words
        check_connection: check that there is a working connection to Reddit
    """

    def __init__(
            self,
            user,
            multiprocess=False
    ):
        """Initialise a RedditWordCounter object.

        :param user: your Reddit username
        :param multiprocess: if True, will handle requests from multiple RedditWordCounter objects (False by default)
        :return:
        """
        handler = MultiprocessHandler() if multiprocess else None
        self.user_agent = 'redditvocab/0.1 bot by {0}'.format(user)
        self.reddit = praw.Reddit(user_agent=self.user_agent, handler=handler)
        self.stemmer = PorterStemmer()

        # Load stop-words
        application_root = os.path.dirname(__file__)
        stopwords = os.path.join(application_root, 'words/stopwords_english.txt')
        with open(stopwords, 'rb') as stopwords_file:
            self.stopwords = [word.strip('\n') for word in stopwords_file.readlines()]

    def subreddit_comments(self, subreddit_name, limit=1000, stemming=False, get_all_comments=False):
        """Retrieve the vocabulary from the comments of a subreddit.

        :param subreddit_name: name of the subreddit excluding '/r/'
        :param limit: number of comments to retrieve (1000 by default) - note that at present the limit is approximate
        :param stemming: if True, performs stemming on tokenized words (False by default)
        :param get_all_comments: if True, retrieves all comments per submission. Note that this requires descending the
        comment tree, which drastically increases the number of API calls and reduces performance due to rate-limiting.
        :return: Counter (dict) of comment vocabulary in the form {'term1': freq, 'term2': freq, ...}
        """

        def get_vocabulary(comments):

            vocab = Counter()
            num_comments = 0
            for comment in comments:
                if isinstance(comment, praw.objects.Comment):
                    try:
                        # Get the word counts for the comment
                        vocab += self.get_word_count(comment.body, stemming=stemming)
                        num_comments += 1

                    except ValueError:
                        pass
                elif isinstance(comment, praw.objects.MoreComments) and get_all_comments:
                    new_vocab, num_new_comments = get_vocabulary(comment.comments)
                    vocab += new_vocab
                    num_comments += num_new_comments

            return vocab, num_comments

        subreddit = self.reddit.get_subreddit(subreddit_name)

        # Initialise loop variables
        vocabulary = Counter()
        comments_processed = 0

        for submission in subreddit.get_hot(limit=None):
            comments = praw.helpers.flatten_tree(submission.comments)

            # Run over all comments
            submission_vocab, num_new_comments = get_vocabulary(comments)
            vocabulary += submission_vocab
            comments_processed += num_new_comments

            print("Comments processed for subreddit '{0}': {1}".format(subreddit_name, comments_processed), end="\r")

            if limit and comments_processed >= limit:
                break

        print('\n')
        return vocabulary

    def subreddit_titles(self, subreddit_name, limit=1000, stemming=False):
        """Retrieve the vocabulary from the titles in a subreddit.

        :param subreddit_name: name of the subreddit excluding '/r/'
        :param limit: number of submissions to process (1000 by default - note that this is the maximum)
        :param stemming: if True, performs stemming on tokenized words (False by default)
        :return: Counter (dict) of title vocabulary in the form {'term1': freq, 'term2': freq, ...}
        """

        subreddit = self.reddit.get_subreddit(subreddit_name)

        # Initialise loop variables
        vocabulary = Counter()
        submissions_processed = 0

        for submission in subreddit.get_hot(limit=limit):
            try:
                # Update the word counter to include the comment
                vocabulary += self.get_word_count(submission.title, stemming=stemming)
                submissions_processed += 1

                if submissions_processed % 100 == 0 or submissions_processed >= limit:
                    print("Titles processed for subreddit '{0}': {1}".format(subreddit_name, submissions_processed),
                          end="\r")

            except ValueError:
                pass

        print('\n')
        return vocabulary

    def user_comments(self, username, limit=1000, stemming=False):
        """Retrieve the vocabulary of a user's comments.

        :param username: user's Reddit username excluding '/u/'
        :param limit: number of comments to process (1000 by default - note that this is the maxmimum)
        :param stemming: if True, performs stemming on tokenized words (False by default)
        :return: Counter (dict) of user's vocabulary in the form {'term1': freq, 'term2': freq, ...}
        """
        user = self.reddit.get_redditor(username)

        vocabulary = Counter()
        comments_processed = 0
        for comment in user.get_comments(limit=limit):
            try:
                # Get the word counts for the comment
                vocabulary += self.get_word_count(comment.body, stemming=stemming)
                comments_processed += 1

                if comments_processed % 100 == 0 or comments_processed >= limit:
                    print("Comments processed for user '{0}': {1}".format(username, comments_processed), end="\r")

            except ValueError:
                pass

        print('\n')
        return vocabulary

    def get_word_count(self, text, stop_words=True, stemming=False):
        text = text.lower()
        punctuation_removed = self.remove_punctuation(text)
        tokens = nltk.word_tokenize(punctuation_removed)

        # Remove stop words
        if stop_words:
            tokens = self.remove_stopwords(tokens)

        if stemming:
            tokens = self.stem_tokens(tokens)

        return Counter(tokens)

    @staticmethod
    def remove_punctuation(text, replacement=' ', exclude="'"):
        """Remove punctuation from an input string """
        text = text.replace("'", "")  # Single quote always stripped out
        for p in set(list(punctuation)) - set(list(exclude)):
            text = text.replace(p, replacement)

        text = ' '.join(text.split())  # Remove excess whitespace
        return text

    def remove_stopwords(self, tokens):
        """Remove all stopwords from a list of word tokens."""
        return [word for word in tokens if word not in self.stopwords]

    def stem_tokens(self, tokens):
        """Perform porter stemming on a list of word tokens."""
        return [self.stemmer.stem(word) for word in tokens]

    def check_connection(self, timeout=10):
        """Wait for a server response."""
        header = {'User-Agent': self.user_agent}
        start = time()
        while True:
            try:
                request = urllib2.Request("http://www.reddit.com/", headers=header)
                response = urllib2.urlopen(request)
                response.read()
                sleep(2)  # Adhere to Reddit API rule of 30 requests per minute
                if response.getcode() == 200:
                    return True
            except urllib2.HTTPError as err:
                print(err)
            finally:
                if time() - start > timeout:
                    return False


class TfidfCorpus(object):
    """Stores features (e.g. words) and their document frequencies in an inverted index. Useful for NLP and machine
    learning applications.

    To initialise a new TfidfCorpus instance:
    >>> corpus = TfidfCorpus()

    By default the corpus will save to 'tfidf_corpus/corpus.json'. You can specify an existing file to load
    or a specific save path as follows:
    >>> corpus = TfidfCorpus(corpus_path='path/to/corpus.json')

    Data Attributes:
        corpus_path (str): save/load path of the corpus
        document_list (list): list of strings indicating the documents stored in the corpus
        document_lengths (dict): sum of word frequencies contained in each document, takes the form:
            {
                "document1": int,
                "document2": int,
                ...
            }
        corpus (dict): dict of Counters that takes the form:
            {
                "term1": {
                    "document1": int,
                    "document2": int
                },
                "term2": {
                    "document1": int,
                    "document2": int,
                },
                ...
            }

    Methods:
        save
        load
        get_corpus_path
        get_document_list
        add_document
        get_document
        delete_document
        append_document
        get_idf
        get_tfidf
        get_document_tfidfs
        get_top_terms
        build_feature_array
    """

    def __init__(self, corpus_path='corpus.json'):

        # Check that the corpus path is valid
        self.check_corpus_path(corpus_path)
        self.corpus_path = corpus_path
        self.document_list = list()
        self.document_lengths = dict()
        self.corpus = dict()

        if os.path.isfile(corpus_path):
            self.load()

    def save(self, path=''):
        """Save the corpus to a JSON file at the path specified in self.corpus_path.

        :param path: you can specify a save path (must end in .json), which will change self.corpus_path
        """
        if path:
            self.check_corpus_path(path)
            self.corpus_path = path

        with open(self.corpus_path, 'wb') as save_file:
            json.dump(
                {
                    'document_list': self.document_list,
                    'document_lengths': self.document_lengths,
                    'corpus': self.corpus
                },
                save_file
            )

    def load(self):
        """Load the corpus from a JSON file. File path defined in self.corpus_path."""
        with open(self.corpus_path, 'rb') as load_file:
            data = json.load(load_file)

        try:
            self.document_list = data['document_list']
            self.document_lengths = data['document_lengths']
            self.corpus = data['corpus']

            # Make sure that frequency dicts in corpus are Counter objects
            for term in self.corpus.iterkeys():
                self.corpus[term] = Counter(self.corpus[term])
        except KeyError as err:
            print('Provided file does not have expected structure')
            raise err

    def get_corpus_path(self):
        return self.corpus_path

    def set_corpus_path(self, path):
        if not path.lower().endswith('.json'):
            raise Exception('Corpus path must be a JSON file (.json extension).')
        self.corpus_path = path

    def get_document_list(self):
        return self.document_list

    def get_vocabulary(self):
        """Return the full list of terms in the corpus."""
        return self.corpus.keys()

    def get_document(self, document_name):
        """Retrieve a document from the corpus."""
        if document_name not in self.document_list:
            raise Exception("No document with name '{0}' found in corpus".format(document_name))
        return Counter({
            term: freqs[document_name] for term, freqs in self.corpus.iteritems() if freqs.get(document_name, 0)
        })

    def add_document(self, document, document_name):
        """Load a document into the corpus.

        :param document: takes the form {'term1': freq1, 'term2', freq2, ...}
        :param document_name: string which uniquely identifies the document
        """
        if document_name in self.document_list:
            print("Document with name '{0}' already exists in corpus." \
                  "Do you wish to replace it?".format(document_name))
            while True:
                replace_doc = raw_input("Response (y/n): ")
                if replace_doc in ['y', 'yes', 'ye']:
                    self.delete_document(document_name)
                    break
                elif replace_doc in ['n', 'no']:
                    return
                else:
                    print('Could not interpret response. Try again.')

        for term, freq in document.iteritems():
            if not self.corpus.get(term, False):
                self.corpus[term] = Counter()

            self.corpus[term][document_name] = freq

        self.document_list.append(document_name)
        self.document_lengths[document_name] = sum(document.itervalues())

    def delete_document(self, document_name):
        """Delete a document from the corpus.

        :param document_name: string indicating document's name in the corpus - should exist in self.document_list
        """
        if document_name not in self.document_list:
            return
        [freqs.pop(document_name) for term, freqs in self.corpus.iteritems() if freqs.get(document_name, 0)]
        self.document_list.remove(document_name)
        self.document_lengths.pop(document_name)

    def append_document(self, document, document_name):
        """Add new counts to an existing document. If the document doesn't exist in the corpus then it is added.

        :param document: dict or Counter of word counts, e.g. {'i': 1, 'like': 2, 'cheese': 1}
        :param document_name: string indicating document's name in the corpus - should exist in self.document_list
        """
        if document_name not in self.document_list:
            self.add_document(document, document_name)
        else:
            for term, freq in document.iteritems():
                if not self.corpus.get(term, False):
                    self.corpus[term] = Counter()

                self.corpus[term][document_name] += freq

        self.document_lengths[document_name] += sum(document.itervalues())

    def get_idf(self, term):
        """Get inverse document frequency of a given term in the corpus."""
        num_documents = len(self.document_list)
        docs_containing_term = len(self.corpus[term])
        return math.log(num_documents / (1 + docs_containing_term))

    def get_tfidf(self, term, document_name):
        """Get tf-idf score given a term and document in the corpus."""
        tf = self.corpus[term].get(document_name, '') / self.document_lengths[document_name]
        idf = self.get_idf(term)
        return tf * idf

    def get_document_tfidfs(self, document_name, l2_norm=True):
        """Get tf-idf scores for all terms in a document.

        :param document_name: string indicating document's name in the corpus - should exist in self.document_list
        :param l2_norm: if True, applies Euclidean normalization to tf-idf scores of the document
        :return: Counter of tf-idf scores for each term
        """
        tfidfs = {
            term: self.get_tfidf(term, document_name) for term, freq in self.corpus.iteritems()
            if freq.get(document_name, '')
        }

        if l2_norm:
            normalization = np.linalg.norm(tfidfs.values(), axis=0)
            for key, value in tfidfs.items():
                tfidfs[key] = value / normalization

        return Counter(tfidfs)

    def get_top_terms(self, document_name, num_terms=30):
        """Get the top terms for a given document by tf-idf score.

        :param document_name: string indicating document's name in the corpus - should exist in self.document_list
        :param num_terms: number of top terms to return (30 by default)
        :return: dict of top terms and their corresponding tf-idf scores
        """
        tfidfs = self.get_document_tfidfs(document_name)
        sorted_tfidfs = sorted(tfidfs.items(), key=operator.itemgetter(1), reverse=True)
        return OrderedDict(sorted_tfidfs[:num_terms])

    def build_vectorizer(self, tfidf=True):
        """Transforms the corpus into a scikit-learn vectorizer object which can be used for machine learning.

        :param tfidf: if True, applies TfidfTransformer to vectorized features
        :return: scikit-learn vectorizer
        """
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer

        index = [self.get_document(document) for document in self.document_list]
        vectorizer = DictVectorizer()
        vectorizer.fit_transform(index)

        if tfidf:
            vectorizer = TfidfTransformer().fit_transform(vectorizer)

        return vectorizer

    def count_words_from_list(self, document_name, word_list, normalize=True):
        """Given a list of input words, return the counts of these words in a specified document."""
        document = self.get_document(document_name)
        word_counts = [document[word] for word in word_list]
        total_count = sum(word_counts)
        if normalize:
            total_count /= self.document_lengths[document_name]

        return total_count

    def get_mean_word_length(self, document_name, upper_limit=12):
        """Get the average word length for all words in a given document."""
        document = self.get_document(document_name)
        return sum([len(term) * freq for term, freq in document.iteritems()
                    if len(term) <= upper_limit]) / sum(document.itervalues())

    @staticmethod
    def check_corpus_path(corpus_path):
        if not corpus_path.lower().endswith('.json'):
            raise Exception('corpus_path provided is not a valid JSON file.')
        make_path(corpus_path)


def make_path(path):
    """Check if path exists. If it doesn't, create the necessary folders."""

    # Remove file name from path
    base_name = os.path.basename(path)
    if '.' in base_name:
        path = path[:-len(base_name)]

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    return path


def get_word_corpora():
    """Returns a list of paths to all word corpora installed in the module."""
    application_root = get_root_dir()
    words_dir = os.path.join(application_root, 'words')
    return os.listdir(words_dir)


def get_root_dir():
    return os.path.dirname(__file__)