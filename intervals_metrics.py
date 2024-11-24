from typing import List, Union, Tuple, Type, Dict, Callable
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Interval:
    start: float
    end: float

    def length(self) -> float:
        return self.end - self.start

    def overlaps(self, other: 'Interval') -> bool:
        return not (self.end <= other.start or other.end <= self.start)

    def intersection(self, other: 'Interval') -> float:
        if not self.overlaps(other):
            return 0
        return min(self.end, other.end) - max(self.start, other.start)

    def union_length(self, other: 'Interval') -> float:
        return self.length() + other.length() - self.intersection(other)


@dataclass
class CompositeInterval:
    intervals: List[Interval]

    def length(self) -> float:
        """Total length of non-overlapping parts"""
        if not self.intervals:
            return 0

        # Sort intervals by start point
        sorted_intervals = sorted(self.intervals, key=lambda x: x.start)
        total_length = 0
        current_end = float('-inf')

        for interval in sorted_intervals:
            if interval.start > current_end:
                # New non-overlapping interval
                total_length += interval.length()
            else:
                # Overlapping interval
                total_length += max(0, interval.end - current_end)
            current_end = max(current_end, interval.end)

        return total_length

    def overlaps(self, other: 'CompositeInterval') -> bool:
        """Check if any part of this interval overlaps with another"""
        return any(interval.overlaps(other_interval) for interval in self.intervals for other_interval in other.intervals)

    def intersection(self, other: 'CompositeInterval') -> float:
        """Calculate total intersection length with another composite interval"""
        total_intersection = 0
        for int1 in self.intervals:
            for int2 in other.intervals:
                total_intersection += int1.intersection(int2)
        return total_intersection

    def union_length(self, other: 'CompositeInterval') -> float:
        """Calculate total union length with another composite interval"""
        # Combine all intervals
        all_intervals = self.intervals + other.intervals
        combined = CompositeInterval(all_intervals)
        return combined.length()


class IntervalSimilarityMethod(ABC):
    """Base class for interval similarity methods"""

    @abstractmethod
    def calculate(self, int1: CompositeInterval, int2: CompositeInterval, **kwargs) -> float:
        """Calculate similarity between two composite intervals"""
        pass

    @classmethod
    def get_name(cls) -> str:
        """Get method name (defaults to lowercase class name without 'method' suffix)"""
        name = cls.__name__.lower()
        if name.endswith('method'):
            name = name[:-6]
        return name


class SimilarityRegistry:
    """Registry for similarity calculation methods"""
    _methods: Dict[str, Type[IntervalSimilarityMethod]] = {}

    @classmethod
    def register(cls, method_class: Type[IntervalSimilarityMethod]) -> Type[IntervalSimilarityMethod]:
        """Register a new similarity method"""
        name = method_class.get_name()
        cls._methods[name] = method_class
        return method_class

    @classmethod
    def get_method(cls, name: str) -> Type[IntervalSimilarityMethod]:
        """Get similarity method by name"""
        if name not in cls._methods:
            raise ValueError(f"Unknown similarity method: {name}")
        return cls._methods[name]

    @classmethod
    def get_available_methods(cls) -> List[str]:
        """Get list of available method names"""
        return list(cls._methods.keys())


def register_similarity_method(cls: Type[IntervalSimilarityMethod]) -> Type[IntervalSimilarityMethod]:
    """Decorator to register similarity methods"""
    return SimilarityRegistry.register(cls)


@register_similarity_method
class JaccardMethod(IntervalSimilarityMethod):
    def calculate(self, int1: CompositeInterval, int2: CompositeInterval, **kwargs) -> float:
        intersection = int1.intersection(int2)
        if intersection == 0:
            return 0
        return intersection / int1.union_length(int2)


@register_similarity_method
class OverlapMethod(IntervalSimilarityMethod):
    def calculate(self, int1: CompositeInterval, int2: CompositeInterval, **kwargs) -> float:
        intersection = int1.intersection(int2)
        if intersection == 0:
            return 0
        return intersection / min(int1.length(), int2.length())


@register_similarity_method
class HausdorffMethod(IntervalSimilarityMethod):
    def calculate(self, int1: CompositeInterval, int2: CompositeInterval, **kwargs) -> float:
        max_distance = kwargs.get('max_distance')

        # Calculate Hausdorff distance
        min_distance = float('inf')
        for sub1 in int1.intervals:
            for sub2 in int2.intervals:
                if sub1.overlaps(sub2):
                    return 1.0  # Normalized similarity for overlapping intervals
                distance = min(abs(sub1.end - sub2.start), abs(sub2.end - sub1.start))
                min_distance = min(min_distance, distance)

        # Calculate normalization factor if not provided
        if max_distance is None:
            max_distance = 0
            for sub1 in int1.intervals:
                for sub2 in int2.intervals:
                    distance = max(abs(sub1.end - sub2.start), abs(sub2.end - sub1.start))
                    max_distance = max(max_distance, distance)
            if max_distance == 0:
                return 1.0

        return 1 - (min_distance / max_distance)


def parse_intervals(
    intervals: Union[List[tuple], List[list], List[Interval], List[CompositeInterval]]
) -> List[CompositeInterval]:
    """Convert input intervals to list of CompositeInterval objects"""
    result = []

    for interval in intervals:
        if isinstance(interval, CompositeInterval):
            result.append(interval)
            continue

        if isinstance(interval, Interval):
            result.append(CompositeInterval([interval]))
            continue

        if isinstance(interval, (tuple, list)):
            # Check if it's a simple interval
            if len(interval) == 2 and all(isinstance(x, (int, float)) for x in interval):
                start, end = interval
                if start > end:
                    start, end = end, start
                result.append(CompositeInterval([Interval(float(start), float(end))]))
            # Check if it's a composite interval
            elif all(isinstance(x, (tuple, list)) and len(x) == 2 for x in interval):
                sub_intervals = []
                for start, end in interval:
                    if start > end:
                        start, end = end, start
                    sub_intervals.append(Interval(float(start), float(end)))
                result.append(CompositeInterval(sub_intervals))
            else:
                raise ValueError("Invalid interval format")

    return result


def calculate_similarity(
    int1: Union[tuple, list, Interval, CompositeInterval],
    int2: Union[tuple, list, Interval, CompositeInterval],
    method: str = 'overlap',
    methods: List[str] = None,
    weights: List[float] = None,
    **kwargs
) -> float:
    """
    Calculate similarity between two intervals or composite intervals

    Parameters:
    -----------
    int1, int2 : intervals in any supported format
    method : str
        Similarity method name or 'mixed' for weighted combination
    methods : List[str]
        Required for mixed method - list of methods to combine
    weights : List[float]
        Required for mixed method - weights for each method
    **kwargs : additional parameters passed to similarity methods

    Returns:
    --------
    float : Similarity value between 0 and 1
    """
    # Parse intervals
    parsed = parse_intervals([int1, int2])
    if len(parsed) != 2:
        raise ValueError("Expected exactly two intervals")

    int1, int2 = parsed

    if method == 'mixed':
        if not methods or not weights or len(methods) != len(weights):
            raise ValueError("For mixed method, must provide equal-length methods and weights lists")

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Calculate weighted sum
        similarity = 0
        for m, w in zip(methods, weights):
            method_class = SimilarityRegistry.get_method(m)
            similarity += w * method_class().calculate(int1, int2, **kwargs)
        return similarity

    # Use single method
    method_class = SimilarityRegistry.get_method(method)
    return method_class().calculate(int1, int2, **kwargs)


def calculate_similarity_matrix(
    intervals: List[Union[tuple, list, Interval, CompositeInterval]],
    method: str = 'overlap',
    methods: List[str] = None,
    weights: List[float] = None,
    **kwargs
) -> np.ndarray:
    """
    Calculate similarity matrix for all interval pairs

    Parameters:
    -----------
    intervals : List of intervals in any supported format
    method : str
        Similarity method name or 'mixed' for weighted combination
    methods : List[str]
        Required for mixed method - list of methods to combine
    weights : List[float]
        Required for mixed method - weights for each method
    **kwargs : additional parameters passed to similarity methods

    Returns:
    --------
    np.ndarray : Matrix of pairwise similarities
    """
    parsed_intervals = parse_intervals(intervals)
    n = len(parsed_intervals)

    if n < 2:
        raise ValueError("Need at least 2 intervals to create similarity matrix")

    result = np.zeros((n, n))

    if method == 'mixed':
        if not methods or not weights or len(methods) != len(weights):
            raise ValueError("For mixed method, must provide equal-length methods and weights lists")

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Calculate weighted sum of different metrics
        for m, w in zip(methods, weights):
            result += w * calculate_similarity_matrix(intervals, method=m, **kwargs)
        return result

    for i in range(n):
        result[i, i] = 1.0  # Diagonal elements
        for j in range(i + 1, n):
            sim = calculate_similarity(
                parsed_intervals[i],
                parsed_intervals[j],
                method=method,
                **kwargs
            )
            result[i, j] = result[j, i] = sim

    return result