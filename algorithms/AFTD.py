
from algorithms.SmileAblation import SmileAblation

class AFTD(SmileAblation):
    """
    Assemble Fixed Threshold Detector

    Ref.:
    [1] J. Antoni et al., “On the design of Optimal Health Indicators for
        early fault detection and their statistical thresholds,” Mechanical
        Systems and Signal Processing, vol. 218, p. 111518, Sep. 2024,
        doi: 10.1016/j.ymssp.2024.111518.
    """
    def __init__(self, config, examples=None, metadatabase_preview_retriever=None,
                 adapt_examples=None, agent_part=False):

        # Recover the fixed threshold detectors
        config["single_group_classifier"]= False # no similarity based signal group classifier
        config["similarity_adjustment"] = False # turn off similarity threshold detectors
        config["approximation_adjustment"] = False # turn off simulation approximation adjustment
        config["knn"] = False # no knn mode
        config ["stats_single_name"] = None # assemble mode

        super(AFTD, self).__init__(config, examples, metadatabase_preview_retriever, adapt_examples, agent_part)


