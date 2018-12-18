import ratebeer
import sys

# 'Stouts and porters': 68, 'India Pale Ale (IPA)': 17
IPA_ID = 17
STOUTS_ID = 68
MAX_REVIEWS_PER_BEER = 150


def cleaned_up(review):
    """
    Clean up a review.
    :param review: Review to clean up.
    :return: Cleaned up review.
    """
    # Remove the Beer Buddy suffix.
    suffix = '---Rated via Beer Buddy for iPhone'
    if review.endswith(suffix):
        review = review[:len(review) - len(suffix)]

    # Remove multiple spaces.
    review = ' '.join(review.split()) + '\n\n'

    # Delete the review if it's too short.
    if len(review.split()) < 5:
        review = ''

    return review


def save_reviews(rb, style_id):
    """
    Save reviews of fifty best beers of a given style to a file.
    :param rb: RateBeer object
    :param style_id: int, identifies beer style to save reviews of
    """
    styles = rb.beer_style_list()
    beers = rb.beer_style(style_id)

    style = list(styles.keys())[list(styles.values()).index(style_id)]
    filename = '_'.join(style.split()) + '.txt'

    with open(filename, 'w') as f:
        for i, beer in enumerate(beers):
            sys.stdout.write('\rSaving reviews of beer number {0}.'.format(i))
            sys.stdout.flush()

            review_cnt = 0
            reviews = beer.get_review_comments()

            for r in reviews:
                # Do not save more than 150 reviews of a given beer to keep the corpus short.
                if review_cnt >= MAX_REVIEWS_PER_BEER:
                    break

                # Do not save reviews containing non-ASCII characters.
                try:
                    r.encode('ascii')
                except UnicodeEncodeError:
                    continue

                # Clean up the review and save it.
                review_cnt = review_cnt + 1
                f.write(cleaned_up(r))

    print('\nBeer reviews saved in ' + filename)


def main():
    rb = ratebeer.RateBeer()
    save_reviews(rb, IPA_ID)
    save_reviews(rb, STOUTS_ID)


if __name__ == '__main__':
    main()
