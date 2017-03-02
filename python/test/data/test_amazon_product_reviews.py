import unittest
from paddle.data import amazon_product_reviews


class AmazonReviewsTest(unittest.TestCase):
    def test_read_data(self):
        dataset = amazon_product_reviews.dataset(
            category=amazon_product_reviews.Categories.AmazonInstantVideo,
            positive_threshold=4,
            negative_threshold=3)

        sample_num = 0

        for _ in dataset.train_data():
            sample_num += 1

        for _ in dataset.test_data():
            sample_num += 1

        self.assertEqual(37126, sample_num)


if __name__ == '__main__':
    unittest.main()
