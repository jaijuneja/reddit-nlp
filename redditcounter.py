import praw
from collections import Counter
from time import time, sleep
import urllib2
import nltk
from nltk.stem.porter import PorterStemmer
from string import punctuation
from praw.handlers import MultiprocessHandler


class RedditWordCounter(object):

    def __init__(self, user, stopwords='words/stopwords_english.txt', multiprocess=False):
        handler = MultiprocessHandler() if multiprocess else None
        self.user_agent = 'redditvocab/0.1 bot by {0}'.format(user)
        self.reddit = praw.Reddit(user_agent=self.user_agent, handler=handler)
        self.stemmer = PorterStemmer()

        # Load stop-words
        with open(stopwords, 'rb') as stopwords_file:
            self.stopwords = [word.strip('\n') for word in stopwords_file.readlines()]

    def subreddit_comments(self, subreddit_name, limit=1000, stemming=False, get_all_comments=False):

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
        print "Comments processed for subreddit '{}': 0".format(subreddit_name),

        # Initialise loop variables
        vocabulary = Counter()
        comments_processed = 0

        for submission in subreddit.get_hot(limit=None):
            comments = praw.helpers.flatten_tree(submission.comments)

            # Run over all comments
            submission_vocab, num_new_comments = get_vocabulary(comments)
            vocabulary += submission_vocab
            comments_processed += num_new_comments

            print "{}...".format(comments_processed),

            if limit and comments_processed >= limit:
                break

        print "{}. Finished!".format(comments_processed)
        return vocabulary

    def subreddit_titles(self, subreddit_name, limit=1000, stemming=False):

        subreddit = self.reddit.get_subreddit(subreddit_name)
        print "Titles processed for subreddit {}: 0".format(subreddit_name),

        # Initialise loop variables
        vocabulary = Counter()
        submissions_processed = 0

        for submission in subreddit.get_hot(limit=limit):
            try:
                # Update the word counter to include the comment
                vocabulary += self.get_word_count(submission.title, stemming=stemming)
                submissions_processed += 1

                if submissions_processed % 100 == 0:
                    print "{}...".format(submissions_processed),

            except ValueError:
                pass

        print "{}. Finished!".format(submissions_processed)
        return vocabulary

    def user_comments(self, username, limit=1000, stemming=False):
        user = self.reddit.get_redditor(username)

        vocabulary = Counter()
        comments_processed = 0
        for comment in user.get_comments(limit=limit):
            try:
                # Get the word counts for the comment
                vocabulary += self.get_word_count(comment.body, stemming=stemming)
                comments_processed += 1

                if comments_processed % 100 == 0:
                    print "{0} comments processed for user " \
                          "'{1}'...".format(comments_processed, username)

            except ValueError:
                pass

        print "Processed {0} comments for user {1}".format(comments_processed, username)
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
    def remove_punctuation(text, replacement=' ', exclude=""):
        """Remove punctuation from an input string """
        for p in set(list(punctuation)) - set(list(exclude)):
            text = text.replace(p, replacement)

        text = ' '.join(text.split())  # Remove excess whitespace
        return text

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stopwords]

    def stem_tokens(self, tokens):
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
                print err
            finally:
                if time() - start > timeout:
                    return False