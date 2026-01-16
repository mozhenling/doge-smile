
from utils.data.stats import stats_with_threshold, stats_classifier_single
from algorithms.Smile import Smile

class SmileAblation(Smile):

    def __init__(self, config, examples=None, metadatabase_preview_retriever=None,
                 adapt_examples=None, agent_part=False):
        super(SmileAblation, self).__init__(config, examples, metadatabase_preview_retriever, adapt_examples, agent_part)

    def predict(self, x, is_batch=True, database=None, database_path=None):
        """
        prediction
        :param x: an instance or a batch of test data, a list of lists with each inner list as a list of floats
        :return: a batch of predicted labels
        """
        database = self.database if database is None else database
        database_path = self.database_path if database_path is None else database_path

        if is_batch:
            stats_batch = [stats_with_threshold(xi, approx=self.config["approximation_adjustment"]) for xi in x]
            if self.config["similarity_adjustment"]:
                exampls_batch = [self.similar_examples(stats, database, database_path=database_path) for stats in stats_batch]
            else:
                exampls_batch = [None] * len(x)
            return [stats_classifier_single(stats, examples=egs,
                                            f_sensitivity=self.config["f_sensitivity"],
                                            test_time_extra_database=self.test_time_extra_database,
                                            test_ulb_max=self.config["smile_test_ulb_max"],
                                            knn=self.config["knn"],
                                            single_group_classifier= self.config["single_group_classifier"],
                                            stats_single_name=self.config["stats_single_name"]) for stats, egs in zip(stats_batch, exampls_batch) ]

        else:
            stats = stats_with_threshold(x, approx= self.config["approximation_adjustment"])
            egs = self.similar_examples(stats, database, database_path=database_path) if self.config["similarity_adjustment"] else None
            return stats_classifier_single(stats, examples=egs,
                                           f_sensitivity=self.config["f_sensitivity"],
                                           test_time_extra_database=self.test_time_extra_database,
                                           test_ulb_max=self.config["smile_test_ulb_max"],
                                           knn=self.config["knn"],
                                           single_group_classifier=self.config["single_group_classifier"],
                                           stats_single_name=self.config["stats_single_name"])


