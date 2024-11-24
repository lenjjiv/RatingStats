from typing import List, Tuple, Optional, Dict, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
from scipy.stats import norm

class RatingStats(BaseModel):
    """Статистика по оценкам"""
    mean: float = Field(..., description="Точечная оценка среднего")
    p_hat: float = Field(..., description="Оценка вероятности успеха")
    max_rating: int = Field(..., description="Максимально возможная оценка")
    p_hat_ci: Optional[Tuple[float, float]] = Field(None, description="Доверительный интервал для p_hat")
    std_error: Optional[float] = Field(None, description="Стандартная ошибка оценки")
    counts: Optional[Dict[int, int]] = Field(None, description="Количество каждой оценки {оценка: количество}")
    mean_ci: Optional[Tuple[float, float]] = Field(None, description="Доверительный интервал для среднего")
    method: str = Field("wilson", description="Использованный метод расчета ДИ")

    @field_validator('p_hat')
    @classmethod
    def validate_p_hat(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError('p_hat должно быть между 0 и 1')
        return v

    @field_validator('max_rating')
    @classmethod
    def validate_max_rating(cls, v: int) -> int:
        if v < 1:
            raise ValueError('max_rating должен быть положительным')
        return v

    @model_validator(mode='after')
    def validate_counts(self) -> 'RatingStats':
        counts = self.counts
        if counts is not None:
            if not all(k >= 0 and k <= self.max_rating for k in counts.keys()):
                raise ValueError(f'Все оценки должны быть между 0 и {self.max_rating}')
            if sum(counts.values()) <= 0:
                raise ValueError('Сумма counts должна быть положительной')
        return self


def calculate_wilson_score_interval(successes: int, n: int, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Вычисляет доверительный интервал методом Вильсона для пропорций.
    """
    if n == 0:
        return (0.0, 0.0)

    p_hat = successes / n
    z = norm.ppf(1 - (1 - alpha) / 2)

    z2 = z * z
    center = (p_hat + z2/(2*n)) / (1 + z2/n)
    spread = z * np.sqrt((p_hat*(1-p_hat) + z2/(4*n)) / n) / (1 + z2/n)

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return lower, upper


def calculate_wald_interval(successes: int, n: int, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Вычисляет доверительный интервал методом Вальда (нормальное приближение).
    """
    if n == 0:
        return (0.0, 0.0)

    p_hat = successes / n
    z = norm.ppf(1 - (1 - alpha) / 2)

    std_error = np.sqrt(p_hat * (1 - p_hat) / n)
    margin = z * std_error

    lower = max(0.0, p_hat - margin)
    upper = min(1.0, p_hat + margin)

    return lower, upper


def calculate_ci_from_ratings(
    ratings: List[int],
    max_rating: int = 2,
    alpha: float = 0.95,
    method: Literal["wilson", "wald"] = "wilson"
) -> RatingStats:
    """
    Рассчитывает доверительные интервалы для среднего на основе списка оценок.

    Args:
        ratings: Список оценок (от 0 до max_rating включительно)
        max_rating: Максимально возможная оценка
        alpha: Доверительная вероятность (по умолчанию 0.95)
        method: Метод расчета доверительного интервала ("wilson" или "wald")

    Returns:
        RatingStats: Объект с результатами расчетов
    """
    if not all(0 <= x <= max_rating for x in ratings):
        raise ValueError(f"Все оценки должны быть между 0 и {max_rating}")
    if not 0 < alpha < 1:
        raise ValueError("alpha должна быть между 0 и 1")
    if method not in ["wilson", "wald"]:
        raise ValueError("method должен быть 'wilson' или 'wald'")

    # Подсчитываем количество каждой оценки
    counts = {i: sum(1 for x in ratings if x == i) for i in range(max_rating + 1)}

    return calculate_ci_from_counts(counts, max_rating, alpha, method)


def calculate_ci_from_counts(
    counts: Dict[int, int],
    max_rating: int = 2,
    alpha: float = 0.95,
    method: Literal["wilson", "wald"] = "wilson"
) -> RatingStats:
    """
    Рассчитывает доверительные интервалы для среднего на основе количества каждой оценки.

    Args:
        counts: Словарь {оценка: количество}
        max_rating: Максимально возможная оценка
        alpha: Доверительная вероятность (по умолчанию 0.95)
        method: Метод расчета доверительного интервала ("wilson" или "wald")

    Returns:
        RatingStats: Объект с результатами расчетов
    """
    if not all(v >= 0 for v in counts.values()):
        raise ValueError("Количество оценок не может быть отрицательным")
    if not 0 < alpha < 1:
        raise ValueError("alpha должна быть между 0 и 1")
    if method not in ["wilson", "wald"]:
        raise ValueError("method должен быть 'wilson' или 'wald'")

    N = sum(counts.values())
    if N == 0:
        raise ValueError("Общее количество оценок должно быть положительным")

    # Рассчитываем p_hat
    successes = sum(rating * count for rating, count in counts.items())
    p_hat = successes / (max_rating * N)

    # Рассчитываем средний скор
    mean = max_rating * p_hat

    # Базовый результат
    result = {
        'mean': mean,
        'p_hat': p_hat,
        'max_rating': max_rating,
        'method': method
    }

    # Стандартная ошибка для p_hat
    std_error = np.sqrt(p_hat * (1 - p_hat) / N)

    # Выбираем метод расчета доверительного интервала
    ci_func = calculate_wilson_score_interval if method == "wilson" else calculate_wald_interval
    p_hat_ci = ci_func(successes, max_rating * N, alpha)

    result.update({
        'p_hat_ci': p_hat_ci,
        'std_error': std_error,
        'counts': counts
    })

    # Доверительные интервалы для среднего
    mean_ci = (max_rating * p_hat_ci[0], max_rating * p_hat_ci[1])
    result['mean_ci'] = mean_ci

    return RatingStats(**result)