"""LLM 에이전트에 제공할 도구 정의와 검증 장치.

- TOOL_SPECS: OpenAI 등 LLM의 함수 호출(tool calling)에 그대로 넘길 수 있는 JSON 스키마.
- ToolBox: LLM이 고른 도구 이름과 인자를 받아 실제 계산 함수를 실행한다.
- verify_numbers: LLM이 쓴 설명문 속 숫자가 모두 도구 결과에 있는 숫자인지 대조한다.
  → "LLM은 숫자를 만들지 않는다"는 원칙을 코드로 강제하는 장치.
본선에서는 여기에 LLM 호출 루프만 붙이면 에이전트가 된다.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from . import config as C
from .model import ResponseModel
from .optimizer import optimize, simulate

_CONDITION_PROPS = {
    "product_id": {"type": "string", "description": "상품 ID (예: 가전-03)"},
    "timeslot": {"type": "string", "enum": C.TIMESLOTS, "description": "방송 시간대"},
    "weekend": {"type": "integer", "enum": [0, 1], "description": "주말 방송이면 1"},
}

TOOL_SPECS = [
    {
        "type": "function",
        "name": "simulate_promotion",
        "description": "한 가지 혜택 조건의 예상 주문·매출·순증 이익을 범위와 함께 계산한다.",
        "parameters": {
            "type": "object",
            "properties": {
                **_CONDITION_PROPS,
                "discount_pct": {"type": "number", "minimum": 0, "maximum": 30},
                "installment_months": {"type": "integer", "enum": [0, 3, 6]},
                "gift_ratio": {"type": "number", "minimum": 0, "maximum": 0.2, "description": "사은품 가치 / 판매가"},
            },
            "required": ["product_id", "timeslot", "weekend", "discount_pct", "installment_months", "gift_ratio"],
        },
    },
    {
        "type": "function",
        "name": "optimize_promotion",
        "description": "제약 조건 안에서 후보 혜택 조합을 모두 비교해 상위안을 추천한다.",
        "parameters": {
            "type": "object",
            "properties": {
                **_CONDITION_PROPS,
                "min_margin": {"type": "number", "description": "최소 이익률 (예: 0.2)"},
                "gift_stock": {"type": "integer", "description": "사은품 재고 수량"},
                "budget": {"type": "number", "description": "홈쇼핑 부담 혜택 비용 상한(원)"},
                "card_share_hs": {"type": "number", "description": "카드 할인 비용 중 홈쇼핑 부담 비율"},
            },
            "required": ["product_id", "timeslot", "weekend", "min_margin"],
        },
    },
]


class ToolBox:
    def __init__(self, model: ResponseModel, products: pd.DataFrame):
        self.model = model
        self.products = products.set_index("product_id")

    def _product(self, product_id: str) -> dict:
        if product_id not in self.products.index:
            raise ValueError(f"없는 상품입니다: {product_id}")
        return {"product_id": product_id, **self.products.loc[product_id].to_dict()}

    def call(self, name: str, args: dict) -> dict:
        """입력 검증 → 계산 실행. 잘못된 입력은 계산 전에 차단한다."""
        if args.get("timeslot") not in C.TIMESLOTS:
            raise ValueError(f"알 수 없는 시간대: {args.get('timeslot')}")
        product = self._product(args["product_id"])
        if name == "simulate_promotion":
            return simulate(self.model, product, args["timeslot"], args["weekend"],
                            args["discount_pct"], args["installment_months"], args["gift_ratio"])
        if name == "optimize_promotion":
            costs = {"card_share_hs": args["card_share_hs"]} if "card_share_hs" in args else None
            res = optimize(self.model, product, args["timeslot"], args["weekend"],
                           min_margin=args["min_margin"], gift_stock=args.get("gift_stock"),
                           budget=args.get("budget"), costs=costs)
            res.pop("table")  # LLM에는 요약만 전달
            return res
        raise ValueError(f"알 수 없는 도구: {name}")


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers_in(obj) -> set[float]:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return {float(m.replace(",", "")) for m in _NUM.findall(text)}


def verify_numbers(explanation: str, tool_outputs: list[dict], rel_tol: float = 0.01) -> dict:
    """설명문의 숫자가 도구 결과 숫자와 일치(상대오차 1% 이내, 백분율·만원 단위 변환 허용)하는지 검사."""
    allowed = _numbers_in(tool_outputs)
    # 같은 값을 다른 단위로 쓰는 경우 허용: 0.123 → 12.3(%), 1,234,567원 → 123.5(만원)
    expanded = set(allowed)
    for v in allowed:
        expanded.update({v * 100, v / 10_000, v / 100_000_000})
    unmatched = []
    for m in _NUM.findall(explanation):
        v = float(m.replace(",", ""))
        if abs(v) < 10 and float(v).is_integer():  # '3안', '1위' 같은 작은 정수는 서수로 보고 건너뜀
            continue
        if not any(abs(v - a) <= rel_tol * max(abs(a), 1e-9) for a in expanded):
            unmatched.append(m)
    return {"ok": not unmatched, "unmatched": unmatched}
