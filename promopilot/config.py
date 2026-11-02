"""PromoPilot 공통 설정.

합성 데이터의 '정답 효과'(TRUE_EFFECTS)와 후보 프로모션 조건(CANDIDATE_GRID)을 한곳에 모아 둔다.
실제 데이터로 바꿀 때는 이 파일의 정답 효과는 쓰지 않고, 비용·제약 기본값만 현업 값으로 교체한다.
"""

# 고객군: 신규 / 기존 / 휴면
SEGMENTS = ["new", "existing", "dormant"]
SEGMENT_LABELS = {"new": "신규", "existing": "기존", "dormant": "휴면"}

# 방송 시간대 (기준 범주는 'morning')
TIMESLOTS = ["morning", "daytime", "afternoon", "prime", "late"]
TIMESLOT_LABELS = {
    "morning": "아침",
    "daytime": "오전",
    "afternoon": "오후",
    "prime": "저녁 프라임",
    "late": "심야",
}

# 카테고리별 가격대(원)와 취소·반품률
CATEGORIES = {
    "가전": {"price_range": (150_000, 600_000), "return_rate": 0.05, "base_log": 5.6},
    "뷰티": {"price_range": (40_000, 120_000), "return_rate": 0.04, "base_log": 6.3},
    "식품": {"price_range": (30_000, 90_000), "return_rate": 0.02, "base_log": 6.5},
    "패션": {"price_range": (60_000, 250_000), "return_rate": 0.15, "base_log": 6.0},
}

# ---------------------------------------------------------------------------
# 합성 데이터의 정답 효과 (로그 주문량 기준)
#   프로모션 효과 = disc * log(1 + 할인율/5) + inst * (무이자개월/6) + gift * (사은품가치비율*10)
#   - 할인 효과는 로그 형태라서 할인율이 커질수록 증가폭이 줄어든다(체감 효과).
#   - 모델이 이 값을 다시 찾아내는지가 파라미터 복원 테스트의 기준이다.
# ---------------------------------------------------------------------------
TRUE_EFFECTS = {
    "new":      {"offset": -1.2, "disc": 0.45, "inst": 0.10, "gift": 0.20},
    "existing": {"offset":  0.0, "disc": 0.30, "inst": 0.15, "gift": 0.35},
    "dormant":  {"offset": -1.6, "disc": 0.55, "inst": 0.05, "gift": 0.25},
}
TRUE_TIMESLOT = {"morning": 0.0, "daytime": -0.25, "afternoon": -0.10, "prime": 0.45, "late": -0.40}
TRUE_WEEKEND = 0.15

# 과거 데이터에서 실제로 쓰인 조건 값 (학습 범위)
OBSERVED_LEVELS = {
    "discount_pct": [0, 3, 5, 7, 10],
    "installment_months": [0, 3, 6],
    "gift_ratio": [0.0, 0.03, 0.05],
}

# 시뮬레이션에서 비교할 후보 조건 (15% 할인은 학습 범위 밖 → 외삽 경고 대상)
CANDIDATE_GRID = {
    "discount_pct": [0, 3, 5, 7, 10, 15],
    "installment_months": [0, 3, 6],
    "gift_ratio": [0.0, 0.03, 0.05],
}

# ---------------------------------------------------------------------------
# 비용 구조 기본값 (모두 가정값이며, 실제 계약 조건으로 바꿔 넣는 파라미터)
# ---------------------------------------------------------------------------
DEFAULT_COSTS = {
    "margin_rate": 0.30,          # 순매출 대비 홈쇼핑 수익률(수수료 또는 마진)
    "card_usage_rate": 0.80,      # 주문 중 할인 대상 카드로 결제하는 비율
    "card_share_hs": 1.00,        # 카드 할인 비용 중 홈쇼핑 부담 비율 (보수적으로 전액 부담 가정)
    "inst_cost_per_month": 0.008, # 무이자 1개월당 비용(판매가 대비)
    "inst_share_hs": 1.00,        # 무이자 비용 중 홈쇼핑 부담 비율
    "gift_share_hs": 1.00,        # 사은품 비용 중 홈쇼핑 부담 비율
}
