import numpy as np

from typing import *
import pandas as pd
import math



class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, certify_results: pd.DataFrame):
        self.certify_results = certify_results

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        return np.array([self.at_radius(self.certify_results, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, certify_results: pd.DataFrame, alpha: float, rho: float):
        self.certify_results = certify_results
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        return np.array([self.at_radius(self.certify_results, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))