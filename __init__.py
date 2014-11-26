import config
from redditcounter import RedditWordCounter
from tfidf import TfidfCorpus
import requests

if __name__ == '__main__':
    # Initialise Reddit word counter instance
    reddit_counter = RedditWordCounter()

    # Initialise tf-idf corpus instance
    comment_corpus = TfidfCorpus(config.SAVE_DIR)

    # Extract the vocabulary for each of the subreddits specified
    failed_subreddits = []
    for subreddit in config.SUBREDDITS:

        connected = reddit_counter.check_connection(timeout=20)
        if not connected:
            failed_subreddits.append(subreddit)
            continue

        try:
            vocabulary = reddit_counter.subreddit_comments(subreddit, limit=config.COMMENTS_PER_SUBREDDIT)
        except requests.exceptions.HTTPError as err:
            print err
            failed_subreddits.append(subreddit)
            continue

        comment_corpus.add_document(vocabulary, subreddit)
        comment_corpus.save_corpus()

    # # Check out the top terms for a given subreddit
    # comment_corpus = TfidfCorpus(config.SAVE_DIR)
    # print comment_corpus.get_top_terms('books')