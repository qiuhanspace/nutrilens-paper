"""NutriLens: an image-derived digital biomarker of the postprandial glucose
trajectory.

Analysis code for "Image-based digital biomarkers predict postprandial glucose
responses across glycaemic states".

Modules
-------
config      paths, cohort definitions and every locked analysis constant
curves      trajectory reconstruction, per-participant error, clinical readouts
stats       difference and equivalence testing, effect sizes, trend test
training    leave-one-subject-out training and per-participant calibration
recourse    counterfactual meal modifications

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""

__version__ = '1.0.0'
__author__ = 'Qiuhan Jin'
__email__ = 'qiuhan.jin@unisg.ch'

from . import config, curves, recourse, stats  # noqa: F401

__all__ = ['config', 'curves', 'stats', 'recourse', 'training']
