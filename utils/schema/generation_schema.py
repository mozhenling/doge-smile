from pydantic import BaseModel, Field

class StatsValueThreshold(BaseModel):
    """value and threshold of a statistical measure"""
    value:float = Field(..., description="The value of a statistical measure")
    threshold:float = Field(..., description="The threshold of a statistical measure")

class StatisticalMeasures(BaseModel):
    """This is for generating eight statistical measures:
    1. 'gk_kurtosis',
    2. 'gk_shape_factor',
    3. 'gk_negentropy',
    4. 'gk_JB_stats',
    5. 'ggk_kurtosis',
    6. 'ggk_crest_factor',
    7. 'ggk_smoothness_factor',
    8. 'ggk_negentropy' .
    Each statistical measure is a dictionary with 'value' and 'threshold', both being floats.
    The prefixes, 'gk' and 'ggk' means 'Gaussian kernel' and 'generalized Gaussian kernel', respectively"""
    gk_kurtosis:StatsValueThreshold = Field(...,
      description="gk_kurtosis metric, a dictionary having 'value' and 'threshold' as keys and both their values being floats.")
    gk_shape_factor:StatsValueThreshold = Field(...,
      description="gk_shape_factor metric, a dictionary having 'value' and 'threshold' as keys and both their values being floats.")
    gk_negentropy: StatsValueThreshold = Field(...,
      description="gk_negentropy, a dictionary having 'value' and 'threshold' as keys and both their values being floats.")
    gk_JB_stats: StatsValueThreshold = Field(...,
      description="gk_JB_stats metric, a dictionary having 'value' and 'threshold' as keys and both their values being floats.")

    ggk_kurtosis: StatsValueThreshold = Field(...,
      description="ggk_kurtosis, a dictionary having 'value' and 'threshold' as keys and both their values being floats.")
    ggk_crest_factor: StatsValueThreshold = Field(...,
      description="ggk_crest_factor, a dictionary having 'value' and 'threshold' as keys and both their values being floats.")
    ggk_smoothness_factor: StatsValueThreshold = Field(...,
      description="ggk_smoothness_factor, a dictionary having 'value' and 'threshold' as keys and both their values being floats.")
    ggk_negentropy: StatsValueThreshold = Field(...,
      description="ggk_negentropy, a dictionary having 'value' and 'threshold' as keys and both their values being floats.")

class StatisticalMeasureList(BaseModel):
    """A list of statistical measures.
    Each entry of the list is a dictionary where each key maps to a dictionary with 'value' and 'threshold', both being floats."""
    stats_list:list[StatisticalMeasures]=Field(..., description="Each entry of the list is a dictionary with eight keys where each key maps to a dictionary with 'value' and 'threshold', both being floats.")