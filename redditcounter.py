import praw
from collections import Counter
from time import time, sleep
import urllib2
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation


class RedditWordCounter(object):
    user_agent = 'redditvocab/0.1 by TheRedfather'

    def __init__(self):
        self.reddit = praw.Reddit(user_agent=self.user_agent)
        self.stemmer = PorterStemmer()

    def subreddit_comments(self, subreddit_name, limit=1000):

        subreddit = self.reddit.get_subreddit(subreddit_name)
        # Initialise loop variables
        vocabulary = Counter()
        comments_processed = 0

        for submission in subreddit.get_hot(limit=None):
            comments = praw.helpers.flatten_tree(submission.comments)

            # Run over all comments
            for comment in comments:
                if isinstance(comment, praw.objects.Comment):
                    try:
                        # Get the word counts for the comment
                        vocabulary += self.get_word_count(comment.body)
                        comments_processed += 1

                        if comments_processed % 100 == 0:
                            print "{0} comments processed for subreddit " \
                                  "'{1}'...".format(comments_processed, subreddit_name)

                    except ValueError:
                        pass

            if comments_processed >= limit:
                break

        print "Processed {0} comments for subreddit '{1}'".format(comments_processed, subreddit_name)
        return vocabulary

    def subreddit_titles(self, subreddit_name, limit=1000):

        subreddit = self.reddit.get_subreddit(subreddit_name)
        # Initialise loop variables
        vocabulary = Counter()
        submissions_processed = 0

        for submission in subreddit.get_hot(limit=limit):
            try:
                # Update the word counter to include the comment
                vocabulary += self.get_word_count(submission.title)
                submissions_processed += 1

                if submissions_processed % 100 == 0:
                    print "{0} titles processed for subreddit '{1}'".format(submissions_processed, subreddit_name)

            except ValueError:
                pass

        print "Processed {0} titles for subreddit {1}".format(submissions_processed, subreddit_name)
        return vocabulary

    def get_word_count(self, text, stop_words=True, stemming=True):
        text = text.lower()
        punctuation_removed = self.remove_punctuation(text)
        # punctuation_removed = text.translate(None, string.punctuation)
        tokens = nltk.word_tokenize(punctuation_removed)

        # Remove stop words
        if stop_words:
            tokens = self.remove_stopwords(tokens)

        if stemming:
            tokens = self.stem_tokens(tokens)

        return Counter(tokens)

    @staticmethod
    def remove_punctuation(text, replacement='', exclude="-"):
        """Remove punctuation from an input string """
        text = text.replace('-', ' ')  # Always replace hyphen with space
        for p in set(list(punctuation)) - set(list(exclude)):
            text = text.replace(p, replacement)

        text = ' '.join(text.split())  # Remove excess whitespace
        return text

    @staticmethod
    def remove_stopwords(tokens, language='english'):
        return [word for word in tokens if word not in stopwords.words(language)]

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    @staticmethod
    def check_connection(timeout=10):
        """Wait for a server response."""
        start = time()
        while True:
            try:
                request = urllib2.Request("http://www.reddit.com/")
                response = urllib2.urlopen(request)
                response.read()
                sleep(2)  # Adhere to Reddit API rule of 30 requests per minute
                if response.getcode() == 200:
                    return True
            except urllib2.HTTPError as err:
                print err
            finally:
                if time() - start > timeout:
                    return False