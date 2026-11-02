"""이익 계산과 전수 탐색 최적화 (AI가 아닌 규칙 기반 코드).

이익 정의 (홈쇼핑 관점, 모든 비용 파라미터는 config.DEFAULT_COSTS 로 교체 가능)
    순주문      = 주문 × (1 − 취소·반품률)
    순매출      = 순주문 × 판매가
    기본 이익   = 순매출 × 수익률
    혜택 비용   = 카드할인 비용 + 무이자 비용 + 사은품 비용 (각각 홈쇼핑 부담분만)
    이익        = 기본 이익 − 혜택 비용
    순증 이익   = 이익(해당 조합) − 이익(혜택 없음, 같은 방송 조건)
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config as C
from .model import ResponseModel


def profit_from_orders(orders, price, return_rate, discount_pct, installment_months, gift_ratio, costs: dict) -> dict:
    """주문 수(스칼라 또는 배열)를 받아 매출·비용·이익을 계산한다."""
    orders = np.asarray(orders, dtype=float)
    net_orders = orders * (1 - return_rate)
    net_sales = net_orders * price
    card_cost = net_sales * (discount_pct / 100) * costs["card_usage_rate"] * costs["card_share_hs"]
    inst_cost = net_sales * costs["inst_cost_per_month"] * installment_months * costs["inst_share_hs"]
    gift_cost = orders * price * gift_ratio * costs["gift_share_hs"]  # 사은품은 반품돼도 회수 못 한다고 가정
    promo_cost = card_cost + inst_cost + gift_cost
    profit = net_sales * costs["margin_rate"] - promo_cost
    return {
        "net_sales": net_sales,
        "promo_cost": promo_cost,
        "profit": profit,
        "profit_margin": np.divide(profit, net_sales, out=np.zeros_like(profit), where=net_sales > 0),
    }


def simulate(model: ResponseModel, product: dict, timeslot: str, weekend: int,
             discount_pct: float, installment_months: int, gift_ratio: float, costs: dict | None = None) -> dict:
    """한 가지 조건의 예상 성과 (점추정 + 90% 범위, 순증 이익 포함)."""
    costs = {**C.DEFAULT_COSTS, **(costs or {})}
    args = (product["product_id"], timeslot, weekend)
    pred = model.predict_orders(*args, discount_pct, installment_months, gift_ratio)
    base = model.predict_orders(*args, 0, 0, 0.0)

    def prof(samples, d, m, g):
        return profit_from_orders(samples, product["price"], product["return_rate"], d, m, g, costs)

    p_point = prof(pred["total"]["mean"], discount_pct, installment_months, gift_ratio)
    b_point = prof(base["total"]["mean"], 0, 0, 0.0)
    # 순증 이익의 범위: 같은 부트스트랩 표본끼리 빼서 계산
    p_s = prof(pred["total"]["samples"], discount_pct, installment_months, gift_ratio)["profit"]
    b_s = prof(base["total"]["samples"], 0, 0, 0.0)["profit"]
    inc = p_s - b_s

    return {
        "discount_pct": discount_pct,
        "installment_months": installment_months,
        "gift_ratio": gift_ratio,
        "orders": round(float(pred["total"]["mean"])),
        "orders_p05": round(float(pred["total"]["p05"])),
        "orders_p95": round(float(pred["total"]["p95"])),
        "orders_by_segment": {s: round(float(pred[s]["mean"])) for s in C.SEGMENTS},
        "baseline_orders": round(float(base["total"]["mean"])),
        "net_sales": round(float(p_point["net_sales"])),
        "promo_cost": round(float(p_point["promo_cost"])),
        "profit": round(float(p_point["profit"])),
        "profit_margin": round(float(p_point["profit_margin"]), 4),
        "incremental_profit": round(float(p_point["profit"] - b_point["profit"])),
        "incremental_profit_p05": round(float(np.percentile(inc, 5))),
        "incremental_profit_p95": round(float(np.percentile(inc, 95))),
        "extrapolation": model.is_extrapolation(product["category"], discount_pct, installment_months, gift_ratio),
    }


def optimize(model: ResponseModel, product: dict, timeslot: str, weekend: int,
             min_margin: float = 0.20, gift_stock: int | None = None, budget: float | None = None,
             objective: str = "incremental_profit", grid: dict | None = None, costs: dict | None = None,
             top_k: int = 3, allow_extrapolation: bool = False) -> dict:
    """후보 조합을 모두 계산하고 제약을 통과한 것 중 목표 기준 상위안을 돌려준다.

    allow_extrapolation=False(기본)이면 학습 범위 밖 조합은 추천에서 빼고,
    대신 'best_extrapolated'로 따로 알려 준다 → 실험(A/B)으로 확인할 후보.
    """
    grid = grid or C.CANDIDATE_GRID
    rows = []
    for d, m, g in itertools.product(grid["discount_pct"], grid["installment_months"], grid["gift_ratio"]):
        r = simulate(model, product, timeslot, weekend, d, m, g, costs)
        reasons = []
        if r["profit_margin"] < min_margin:
            reasons.append(f"이익률 {r['profit_margin']:.1%} < 최소 {min_margin:.0%}")
        if gift_stock is not None and g > 0 and r["orders_p95"] > gift_stock:
            reasons.append(f"주문 상단 {r['orders_p95']}건 > 사은품 재고 {gift_stock}개")
        if budget is not None and r["promo_cost"] > budget:
            reasons.append(f"혜택 비용 {r['promo_cost']:,}원 > 예산 {budget:,.0f}원")
        r["constraint_ok"] = not reasons
        if r["extrapolation"] and not allow_extrapolation:
            reasons.append("학습 범위 밖 조건 (실험으로 확인 후 사용 권장)")
        r["feasible"] = not reasons
        r["rejected_reasons"] = reasons
        rows.append(r)

    table = pd.DataFrame(rows)
    feasible = table[table["feasible"]].sort_values(objective, ascending=False)
    baseline = table[(table.discount_pct == 0) & (table.installment_months == 0) & (table.gift_ratio == 0)].iloc[0]

    # 학습 범위 밖이지만 제약은 통과하고, 추천 1위보다 좋아 보이는 조합 → 실험 후보로 별도 표시
    extra = table[table["constraint_ok"] & table["extrapolation"].map(bool)].sort_values(objective, ascending=False)
    best_extra = None
    if len(extra) and (feasible.empty or extra.iloc[0][objective] > feasible.iloc[0][objective]):
        best_extra = extra.iloc[0].to_dict()

    return {
        "objective": objective,
        "n_candidates": len(table),
        "n_feasible": len(feasible),
        "top": feasible.head(top_k).to_dict("records"),
        "baseline": baseline.to_dict(),
        "best_extrapolated": best_extra,
        "table": table,
    }
