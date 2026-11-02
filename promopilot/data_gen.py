"""정답 효과를 심어 둔 합성 방송 데이터 생성기.

핵심 의도
- 프로모션 효과를 config.TRUE_EFFECTS 로 미리 정해 둔다. → 모델이 이 값을 되찾는지 검증할 수 있다.
- 일부러 '선택 편향'을 넣는다: 인기 상품과 프라임 시간대에 더 큰 혜택이 몰리게 배정한다.
  → 통제 변수 없이 단순 비교하면 효과가 부풀려지는 현상을 재현하고, 모델의 보정이 동작하는지 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def make_products(n_per_category: int = 10, seed: int = 0) -> pd.DataFrame:
    """카테고리별 상품 목록과 상품별 기본 인기도(로그 주문량 기준)를 만든다."""
    rng = np.random.default_rng(seed)
    rows = []
    for cat, spec in C.CATEGORIES.items():
        lo, hi = spec["price_range"]
        for i in range(n_per_category):
            rows.append(
                {
                    "product_id": f"{cat}-{i + 1:02d}",
                    "category": cat,
                    "price": int(rng.uniform(lo, hi) // 1000 * 1000),
                    "popularity": rng.normal(0.0, 0.4),  # 상품 고유 인기도 (모델은 이 값을 모른다)
                    "base_log": spec["base_log"],
                    "return_rate": spec["return_rate"],
                }
            )
    return pd.DataFrame(rows)


def _pick_level(levels: list, score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """점수가 높을수록 큰 혜택 단계가 뽑히도록 배정한다 (선택 편향의 원천)."""
    noisy = score + rng.normal(0.0, 1.0, size=score.shape)
    qs = np.quantile(noisy, np.linspace(0, 1, len(levels) + 1)[1:-1])
    idx = np.searchsorted(qs, noisy)
    return np.asarray(levels)[idx]


def generate_broadcasts(n_broadcasts: int = 3000, seed: int = 42, products: pd.DataFrame | None = None) -> pd.DataFrame:
    """방송 단위 합성 데이터를 만든다. 한 행 = 방송 1회 × 상품 1개, 고객군별 주문 수 포함."""
    rng = np.random.default_rng(seed)
    if products is None:
        products = make_products(seed=seed)

    p = products.sample(n=n_broadcasts, replace=True, random_state=seed).reset_index(drop=True)
    slot = rng.choice(C.TIMESLOTS, size=n_broadcasts, p=[0.2, 0.2, 0.2, 0.2, 0.2])
    weekend = rng.random(n_broadcasts) < 2 / 7

    # 선택 편향: 인기 상품·프라임 시간대일수록 혜택이 크다
    bias_score = 2.0 * p["popularity"].to_numpy() + 0.8 * (slot == "prime")
    discount = _pick_level(C.OBSERVED_LEVELS["discount_pct"], bias_score, rng)
    installment = _pick_level(C.OBSERVED_LEVELS["installment_months"], bias_score, rng)
    gift = _pick_level(C.OBSERVED_LEVELS["gift_ratio"], bias_score, rng)

    df = pd.DataFrame(
        {
            "broadcast_id": np.arange(n_broadcasts),
            "product_id": p["product_id"],
            "category": p["category"],
            "price": p["price"],
            "return_rate": p["return_rate"],
            "timeslot": slot,
            "weekend": weekend.astype(int),
            "discount_pct": discount,
            "installment_months": installment,
            "gift_ratio": gift,
        }
    )

    # 기준 수요 (혜택이 없을 때)
    base = (
        p["base_log"].to_numpy()
        + p["popularity"].to_numpy()
        + np.vectorize(C.TRUE_TIMESLOT.get)(slot)
        + C.TRUE_WEEKEND * weekend
    )

    for seg, eff in C.TRUE_EFFECTS.items():
        promo = (
            eff["disc"] * np.log1p(discount / 5)
            + eff["inst"] * (installment / 6)
            + eff["gift"] * (gift * 10)
        )
        mu = base + eff["offset"] + promo + rng.normal(0.0, 0.10, n_broadcasts)  # 방송별 잡음
        df[f"orders_{seg}"] = rng.poisson(np.exp(mu))

    df["orders_total"] = df[[f"orders_{s}" for s in C.SEGMENTS]].sum(axis=1)
    return df
