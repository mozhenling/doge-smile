import copy

import numpy as np
import scipy as spy
from collections import defaultdict
from scipy.stats import skew, kurtosis
from scipy.special import gamma, digamma, polygamma
from utils.output.seed import seed_hash

def stats_with_threshold(x, alpha=0.05, scale=1e3, offset=0, approx=True):
    """
    Compute statistical measures with threshold based on [1]

    parameters:
        - x: the preprocessed signal
        - alpha: the risk of rejecting the null hypothesis
                  The alpha value gives us the probability of a type I error.
                  Type I errors occur when we reject a null hypothesis that is actually true.
        - scale: scaling the stats before keeping its decimal to avoid meaningless zeros
        - offset: offsetting the threshold before scaling it and keeping its decimal

    return: a dictionary of statistical measures with their thresholds

    Ref.:
    [1] J. Antoni et al., “On the design of Optimal Health Indicators for
        early fault detection and their statistical thresholds,” Mechanical
        Systems and Signal Processing, vol. 218, p. 111518, Sep. 2024,
        doi: 10.1016/j.ymssp.2024.111518.

    """
    # Register stats
    stats_names = [
        "gk_kurtosis",  # Gaussian kernel
        "gk_shape_factor", # Gaussian kernel
        "gk_negentropy",  # Gaussian kernel
        "gk_JB_stats", # Gaussian kernel

        "ggk_kurtosis", # Generalized Gaussian kernel
        "ggk_crest_factor",  # Generalized Gaussian kernel
        "ggk_smoothness_factor",  # Generalized Gaussian kernel
        "ggk_negentropy",  # Generalized Gaussian kernel
    ]

    stats = {k: {"value": 0, "threshold": 0} for k in stats_names}

    sigLen = len(x)
    eps = 10 ** -16  # a tiny number for numerical stability
    # make sure x is np.array
    x = np.array(x)

    # chi-square distribution of 1 degree of freedom with confidence 1-alpha
    chi2df1 = spy.stats.chi2.isf(q=1 - alpha, df=1, loc=0, scale=1)
    chi2df2 = spy.stats.chi2.isf(q=1 - alpha, df=2, loc=0, scale=1)

    # -- handle approximation error
    # Create a statistical threshold based on simulation
    # Generate 1000 points from the Gaussian distribution. You may customize xg.
    if approx:
        rng = np.random.default_rng(
            seed=seed_hash(x))  # For the same x, make sure we have the same simulated gaussian signal
        xg = rng.normal(loc=np.mean(x), scale=np.std(x), size=sigLen)
    else:
        xg = None

    def val_func(x, xg, func):
        # If func(x) - func(xg) is negative, then no faults.
        return func(x) - func(xg) if xg is not None else func(x)

    # statistical measures derived based on the generalized likelihood ratio
    # -- eq.(43) of [1]
    # By default, it is Fisher’s definition, 0 for gaussian normal (by substracting 3)
    def gk_kurtosis(x):
        return spy.stats.kurtosis(x, fisher=True) ** 2

    stats["gk_kurtosis"]["value"] = val_func(x, xg, gk_kurtosis)
    stats["gk_kurtosis"]["threshold"] = 24 * chi2df1 / sigLen

    # -- eqs.(48) and (51) of [1]
    def gk_shape_factor(x):
        return (np.mean(np.abs(x)) / (eps + np.sqrt(np.mean(x ** 2 + eps))) - np.sqrt((2 / np.pi))) ** 2

    stats["gk_shape_factor"]["value"] = val_func(x, xg, gk_shape_factor)
    stats["gk_shape_factor"]["threshold"] = (1 - 3 / np.pi) * chi2df1 / sigLen

    # -- eq.(58) of [1]
    def gk_negentropy(x):
        c = 0.7296
        x_sqr_avg_norm = x ** 2 / (np.mean(x ** 2) + eps)
        h2_hat = np.mean(x_sqr_avg_norm * np.log(x_sqr_avg_norm + eps))
        return (h2_hat - c) ** 2

    stats["gk_negentropy"]["value"] = val_func(x, xg, gk_negentropy)
    stats["gk_negentropy"]["threshold"] = 0.8044 * chi2df1 / sigLen

    # -- eq.(72) of [1]
    def gk_JB_stats(x):
        return spy.stats.skew(x) ** 2 / 12 + spy.stats.kurtosis(x) ** 2 / 48

    stats["gk_JB_stats"]["value"] = val_func(x, xg, gk_JB_stats)
    stats["gk_JB_stats"]["threshold"] = 0.5 * chi2df2 / sigLen

    # -----------------------------------Generalized Gaussian Kernel Based
    # -- eq.(88) of [1]
    def ggk_kurtosis(x):
        return (1 / np.sqrt(eps + spy.stats.kurtosis(x, fisher=False)) - 0.6760) ** 2

    stats["ggk_kurtosis"]["value"] = val_func(x, xg, ggk_kurtosis)
    stats["ggk_kurtosis"]["threshold"] = 0.0861 * chi2df1 / sigLen

    # -- eqs.(95) and (101) of [1]
    def ggk_crest_factor(x):
        s2inf_hat = np.mean(np.abs(x) ** 2) / (eps + np.max(np.abs(x)) ** 2)
        return (s2inf_hat - 1 / 3) ** 2

    stats["ggk_crest_factor"]["value"] = val_func(x, xg, ggk_crest_factor)
    stats["ggk_crest_factor"]["threshold"] = 4 * chi2df1 / (45 * sigLen)

    # -- eqs.(112) and (113) of [1]
    smo_q = 2
    def ggk_smoothness_factor(x):
        s_hat = np.mean(np.log(np.abs(x) ** smo_q + eps)) - np.log(np.mean(np.abs(x) ** smo_q) + eps)
        return (s_hat - np.log(smo_q) - digamma(1 / smo_q)) ** 2

    stats["ggk_smoothness_factor"]["value"] = val_func(x, xg, ggk_smoothness_factor)
    stats["ggk_smoothness_factor"]["threshold"] = (digamma(1 / smo_q) - smo_q) * chi2df1 / sigLen

    # -- eqs.(105) and (106) of [1]
    neg_q = 2

    def ggk_negentropy(x):
        x_sqr_avg_qnorm = np.abs(x) ** neg_q / (np.mean(np.abs(x) ** neg_q) + eps)
        hq_hat = np.mean(x_sqr_avg_qnorm * np.log(x_sqr_avg_qnorm + eps))
        return (hq_hat - np.log(neg_q) - neg_q - digamma(1 / neg_q)) ** 2

    stats["ggk_negentropy"]["value"] = val_func(x, xg, ggk_negentropy)
    stats["ggk_negentropy"]["threshold"] = ((1 + neg_q) * polygamma(n=3, x=1 / neg_q) - neg_q * (
                1 + neg_q + neg_q ** 2)) * chi2df1 / sigLen

    # scaling and offsetting before keeping the decimals
    for s in stats:
        value = scale * stats[s]["value"]
        stats[s]["value"] = value  # float(value) if tofloat else value
        threshold = scale * (stats[s]["threshold"] + offset)
        stats[s]["threshold"] = threshold  # if tofloat else threshold

    return stats

def stats_hmax_fmin_threshold(h_stats, f_stats):
    """Greedy method for finding bounds based on similar examples"""

    # Initialize dicts to track max/min values
    # Create dictionaries with default values for missing keys
    h_bound = defaultdict(lambda: float('-inf'))
    f_bound = defaultdict(lambda: float('inf'))

    # Update max values for healthy examples
    for stats_dict in h_stats:
        for k, v in stats_dict.items():
            h_bound[k] = max(h_bound[k], v["value"])  #

    # Update min values for faulty examples
    for stats_dict in f_stats:
        for k, v in stats_dict.items():
            f_bound[k] = min(f_bound[k], v["value"])

    return h_bound, f_bound

def stats_classifier_single(stats, examples, test_time_extra_database, test_ulb_max,
                            f_sensitivity = 0.5,
                            single_group_classifier=True,
                            knn=False,
                            stats_single_name=None ):
    """
    Classify machine states into 1 (faulty) or 0 (healthy)
    """
    # -------------------------------------
    # add test time examples if the current number from the dictionary of test_time_extra_database is less than test_ulb_max 
    def _add_test_time_examples(stats, label):
        f_len = len(test_time_extra_database["f_examples"])
        n_len = len(test_time_extra_database["n_examples"])

        current_len = f_len if label == 1 else n_len

        if current_len <= test_ulb_max - 1:
            current_key = "f_examples" if label == 1 else "n_examples"
            id = seed_hash(stats, label)
            new_eg = {"stats": stats, "label": label, "metadata": {"source": "test_time_adjustment", "id": id}}
            print(f"""Test_time_extra:\n\tf_examples_num={f_len}\n\tn_examples_num={n_len}""")
            ids = [d["metadata"]["id"] for d in test_time_extra_database[current_key]] if current_len > 0 else []
            if id not in ids:
                test_time_extra_database[current_key].append(new_eg)
    #-------------------------------------
    # If similar examples are not none, we use the similarity threshold detector; otherwise, the fixed threshold one
    if examples is not None:

        # Separate examples by label
        h_stats = [e["stats"] for e in examples if e["label"] == 0]
        f_stats = [e["stats"] for e in examples if e["label"] == 1]
        h_num, f_num = len(h_stats), len(f_stats)

        # if single_group_classifier is true and there is only one class of examples
        # return the label as this class
        if ( h_num == 0 or f_num == 0 ) and single_group_classifier:
            label = int(h_num==0)
            
            _add_test_time_examples(stats, label)

            return label
        # otherwise, classify the test sample based on the hmax and fmin of the
        # similar state measures from the training set
        else:
            if knn:
                return int(f_num>h_num) # majority voting of KNN
            else:
                h_bound, f_bound = stats_hmax_fmin_threshold(h_stats, f_stats)
                # Update the thresholds in the original stats
                for k in stats:
                    stats_temp = []
                    if h_bound[k] != float('-inf'):
                        stats_temp.append(h_bound[k])
                    if f_bound[k] != float('inf'):
                        stats_temp.append(f_bound[k])
                    stats[k]["threshold"] = np.mean(stats_temp)

    # if not specifying a certain state measure, use all the available measures
    if stats_single_name is None:
        # Classify based on final thresholds
        states = [float(v["value"] > v["threshold"]) for v in stats.values()]
        # you may adjust the voting sensitivity
        assert f_sensitivity < 1 and f_sensitivity >0, "f_sensitivity should be within [0, 1]!"
        label = int( sum(states) > len(states) * (1-f_sensitivity) )
    # otherwise, use a specified measure
    else:
        label = int(stats[stats_single_name]["value"] > stats[stats_single_name]["threshold"])

    return label


def common_health_indicators(x):
    """
    Calculate common health indicators used in bearing fault detection.

    Parameters:
    x (array-like): Vibration signal (time-domain)

    Returns:
    dict: Dictionary with health indicators and their values
    """
    x = np.asarray(x)
    # Basic statistics
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    # mean = np.mean(x)
    abs_mean = np.mean(np.abs(x))
    # std_dev = np.std(x)

    # Health indicators
    crest_factor = peak / rms if rms != 0 else np.nan
    impulse_factor = peak / abs_mean if abs_mean != 0 else np.nan
    clearance_factor = peak / (np.mean(np.sqrt(np.abs(x))) ** 2) if np.mean(np.sqrt(np.abs(x))) != 0 else np.nan
    shape_factor = rms / abs_mean if abs_mean != 0 else np.nan
    skewness = skew(x)
    kurt = kurtosis(x)

    return {
        "RMS": rms,
        "Peak Value": peak,
        "Crest Factor": crest_factor,
        "Impulse Factor": impulse_factor,
        "Clearance Factor": clearance_factor,
        "Shape Factor": shape_factor,
        "Skewness": skewness,
        "Kurtosis": kurt
    }