"""고객군별 프로모션 반응 모델.

모델 형태 (고객군마다 하나씩 적합)
    log(주문수 + 1) = 상품 고정효과 + 시간대 + 주말
                     + b_disc * log(1 + 할인율/5) + b_inst * (무이자개월/6) + b_gift * (사은품가치비율*10)

설계 포인트
- 상품 고정효과: '같은 상품이 다른 조건으로 팔린 차이'로 효과를 추정 → 인기 상품에 혜택이 몰린 편향을 보정.
- 시간대·주말 통제: 프라임 시간대에 혜택이 몰린 편향을 보정.
- 혜택 효과는 음수가 나오면 0으로 고정(단조 제약)하고 경고를 남긴다.
- 부트스트랩으로 예측 범위를 낸다.
한계: 관측되지 않은 요인(예: 쇼호스트, 경쟁 방송)이 혜택 배정에 영향을 주면 이 보정으로는 부족하다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

PROMO_FEATURES = ["f_disc", "f_inst", "f_gift"]


def promo_features(discount_pct, installment_months, gift_ratio) -> np.ndarray:
    """혜택 조건을 모델 입력값으로 바꾼다 (데이터 생성기와 같은 함수 형태를 가정)."""
    d = np.asarray(discount_pct, dtype=float)
    m = np.asarray(installment_months, dtype=float)
    g = np.asarray(gift_ratio, dtype=float)
    return np.column_stack([np.log1p(d / 5), m / 6, g * 10])


@dataclass
class SegmentFit:
    coef: dict                       # 변수명 → 계수
    warnings: list = field(default_factory=list)


class ResponseModel:
    def __init__(self, n_bootstrap: int = 200, seed: int = 0):
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.products_: list[str] = []
        self.slots_: list[str] = [s for s in C.TIMESLOTS if s != "morning"]  # morning = 기준
        self.fits_: dict[str, SegmentFit] = {}
        self.boot_: dict[str, np.ndarray] = {}   # 고객군 → (n_bootstrap, n_params)
        self.observed_range_: dict = {}

    # ----------------------------------------------------------------- 설계 행렬
    def _design(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cols, names = [], []
        for pid in self.products_:
            cols.append((df["product_id"] == pid).to_numpy(float))
            names.append(f"product[{pid}]")
        for s in self.slots_:
            cols.append((df["timeslot"] == s).to_numpy(float))
            names.append(f"slot[{s}]")
        cols.append(df["weekend"].to_numpy(float))
        names.append("weekend")
        X_promo = promo_features(df["discount_pct"], df["installment_months"], df["gift_ratio"])
        X = np.column_stack(cols + [X_promo])
        return X, names + PROMO_FEATURES

    @staticmethod
    def _solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    # ----------------------------------------------------------------- 학습
    def fit(self, df: pd.DataFrame) -> "ResponseModel":
        self.products_ = sorted(df["product_id"].unique())
        X, names = self._design(df)
        rng = np.random.default_rng(self.seed)
        n = len(df)
        boot_idx = [rng.integers(0, n, n) for _ in range(self.n_bootstrap)]

        for seg in C.SEGMENTS:
            y = np.log(df[f"orders_{seg}"].to_numpy(float) + 1)
            beta = self._solve(X, y)
            coef = dict(zip(names, beta))
            warnings = []
            for f in PROMO_FEATURES:  # 단조 제약: 혜택이 커지면 주문이 줄지 않는다
                if coef[f] < 0:
                    warnings.append(f"{seg}: {f} 계수가 음수({coef[f]:.3f})라 0으로 고정")
                    coef[f] = 0.0
            self.fits_[seg] = SegmentFit(coef=coef, warnings=warnings)
            self.boot_[seg] = np.array([self._solve(X[i], y[i]) for i in boot_idx])

        # 외삽 판정을 위한 카테고리별 관측 범위
        for cat, g in df.groupby("category"):
            self.observed_range_[cat] = {
                "discount_pct": (g["discount_pct"].min(), g["discount_pct"].max()),
                "installment_months": (g["installment_months"].min(), g["installment_months"].max()),
                "gift_ratio": (g["gift_ratio"].min(), g["gift_ratio"].max()),
            }
        self.param_names_ = names
        return self

    def promo_coefficients(self) -> pd.DataFrame:
        """고객군별 혜택 효과 추정치와 95% 부트스트랩 구간."""
        rows = []
        idx = [self.param_names_.index(f) for f in PROMO_FEATURES]
        for seg in C.SEGMENTS:
            for f, j in zip(PROMO_FEATURES, idx):
                b = self.boot_[seg][:, j]
                rows.append(
                    {
                        "segment": seg,
                        "feature": f,
                        "estimate": self.fits_[seg].coef[f],
                        "ci_low": np.percentile(b, 2.5),
                        "ci_high": np.percentile(b, 97.5),
                    }
                )
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------- 예측
    def is_extrapolation(self, category: str, discount_pct, installment_months, gift_ratio) -> list[str]:
        rng_ = self.observed_range_.get(category)
        if rng_ is None:
            return ["학습 데이터에 없는 카테고리"]
        flags = []
        labels = {"discount_pct": "할인율", "installment_months": "무이자 개월", "gift_ratio": "사은품 비율"}
        for name, v in [("discount_pct", discount_pct), ("installment_months", installment_months), ("gift_ratio", gift_ratio)]:
            lo, hi = rng_[name]
            if v < lo or v > hi:
                flags.append(f"{labels[name]} {v:g}: 학습 범위 {lo:g}~{hi:g} 밖")
        return flags

    def _row_vector(self, product_id, timeslot, weekend, discount_pct, installment_months, gift_ratio) -> np.ndarray:
        if product_id not in self.products_:
            raise ValueError(f"학습 데이터에 없는 상품: {product_id}")
        v = [1.0 if p == product_id else 0.0 for p in self.products_]
        v += [1.0 if s == timeslot else 0.0 for s in self.slots_]
        v += [float(weekend)]
        v += list(promo_features([discount_pct], [installment_months], [gift_ratio])[0])
        return np.array(v)

    def predict_orders(self, product_id, timeslot, weekend, discount_pct, installment_months, gift_ratio) -> dict:
        """고객군별 예상 주문 수(점추정 + 90% 범위)."""
        x = self._row_vector(product_id, timeslot, weekend, discount_pct, installment_months, gift_ratio)
        out = {}
        for seg in C.SEGMENTS:
            beta = np.array([self.fits_[seg].coef[n] for n in self.param_names_])
            point = np.exp(x @ beta) - 1
            boot = np.exp(self.boot_[seg] @ x) - 1
            out[seg] = {"mean": point, "p05": np.percentile(boot, 5), "p95": np.percentile(boot, 95), "samples": boot}
        total_samples = sum(out[s]["samples"] for s in C.SEGMENTS)
        out["total"] = {
            "mean": sum(out[s]["mean"] for s in C.SEGMENTS),
            "p05": np.percentile(total_samples, 5),
            "p95": np.percentile(total_samples, 95),
            "samples": total_samples,
        }
        return out


def naive_estimates(df: pd.DataFrame) -> dict:
    """비교용: 상품·시간대 통제 없이 혜택 변수만 넣은 단순 회귀 (편향이 생기는 방식)."""
    X = np.column_stack([np.ones(len(df)), promo_features(df["discount_pct"], df["installment_months"], df["gift_ratio"])])
    out = {}
    for seg in C.SEGMENTS:
        y = np.log(df[f"orders_{seg}"].to_numpy(float) + 1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[seg] = dict(zip(PROMO_FEATURES, beta[1:]))
    return out
