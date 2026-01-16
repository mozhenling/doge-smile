
from utils.data.stats import stats_with_threshold, stats_classifier_single

class Human():
    def __init__(self, config, examples=None):
        # use to differentiate LLM and non-LLM algorithms
        self.is_llm = False

        self.config = config
        self.examples = examples


    def predict(self, x):
        """
        prediction
        :param x: a batch of test data
        :return: a batch of predicted labels
        """
        return [stats_classifier_single(stats_with_threshold(xi), examples=self.examples) for xi in x ]
