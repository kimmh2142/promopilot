"""PromoPilot 검증 테스트.

1. 파라미터 복원: 합성 데이터에 심어 둔 정답 효과를 모델이 다시 찾아내는가?
2. 편향 재현: 통제 없는 단순 회귀는 효과를 부풀리는가? (보정이 왜 필요한지 보여 줌)
3. 최적화: 제약을 어긴 조합이 추천되지 않는가? 학습 범위 밖 조합이 걸러지는가?
4. 수치 대조: 도구 결과에 없는 숫자가 설명문에 있으면 잡아내는가?
"""

import pytest

from promopilot import config as C
from promopilot.agent_tools import ToolBox, verify_numbers
from promopilot.data_gen import generate_broadcasts, make_products
from promopilot.model import PROMO_FEATURES, ResponseModel, naive_estimates
from promopilot.optimizer import optimize, simulate

TRUE_KEY = {"f_disc": "disc", "f_inst": "inst", "f_gift": "gift"}


@pytest.fixture(scope="module")
def setup():
    products = make_products(seed=42)
    df = generate_broadcasts(3000, seed=42, products=products)
    model = ResponseModel(n_bootstrap=100, seed=0).fit(df)
    return products, df, model


def _product(products, pid):
    return {"product_id": pid, **products.set_index("product_id").loc[pid].to_dict()}


def test_parameter_recovery(setup):
    """모든 고객군·혜택 변수에서 정답이 95% 구간 안에 있고, 오차가 0.05 이내."""
    _, _, model = setup
    coefs = model.promo_coefficients()
    for _, r in coefs.iterrows():
        truth = C.TRUE_EFFECTS[r["segment"]][TRUE_KEY[r["feature"]]]
        assert abs(r["estimate"] - truth) < 0.05, (r["segment"], r["feature"], r["estimate"], truth)
        assert r["ci_low"] - 0.01 <= truth <= r["ci_high"] + 0.01, (r["segment"], r["feature"])


def test_naive_model_is_biased(setup):
    """통제 없는 단순 비교는 할인 효과를 정답보다 크게 추정한다 (선택 편향)."""
    _, df, _ = setup
    naive = naive_estimates(df)
    for seg in C.SEGMENTS:
        assert naive[seg]["f_disc"] > C.TRUE_EFFECTS[seg]["disc"] + 0.1


def test_recovery_on_other_seeds():
    """시드를 바꿔도 복원이 안정적인지 확인."""
    for seed in (1, 7):
        products = make_products(seed=seed)
        df = generate_broadcasts(3000, seed=seed, products=products)
        model = ResponseModel(n_bootstrap=20, seed=seed).fit(df)
        for seg in C.SEGMENTS:
            for f in PROMO_FEATURES:
                truth = C.TRUE_EFFECTS[seg][TRUE_KEY[f]]
                assert abs(model.fits_[seg].coef[f] - truth) < 0.06


def test_optimizer_respects_constraints(setup):
    products, _, model = setup
    p = _product(products, "가전-03")
    res = optimize(model, p, "prime", 0, min_margin=0.20, gift_stock=300)
    assert res["n_feasible"] > 0
    for t in res["top"]:
        assert t["profit_margin"] >= 0.20
        assert not t["extrapolation"]
        if t["gift_ratio"] > 0:
            assert t["orders_p95"] <= 300
    # 상위안은 순증 이익 내림차순
    vals = [t["incremental_profit"] for t in res["top"]]
    assert vals == sorted(vals, reverse=True)


def test_extrapolation_flagged(setup):
    products, _, model = setup
    p = _product(products, "가전-03")
    r = simulate(model, p, "prime", 0, 15, 0, 0.0)
    assert r["extrapolation"], "15% 할인은 학습 범위(0~10%) 밖이므로 경고가 나와야 한다"
    r2 = simulate(model, p, "prime", 0, 7, 0, 0.0)
    assert not r2["extrapolation"]


def test_baseline_has_zero_incremental_profit(setup):
    products, _, model = setup
    r = simulate(model, _product(products, "뷰티-02"), "daytime", 1, 0, 0, 0.0)
    assert r["incremental_profit"] == 0


def test_toolbox_rejects_bad_input(setup):
    products, _, model = setup
    tb = ToolBox(model, products)
    with pytest.raises(ValueError):
        tb.call("simulate_promotion", {"product_id": "없는상품", "timeslot": "prime", "weekend": 0,
                                       "discount_pct": 5, "installment_months": 0, "gift_ratio": 0})
    with pytest.raises(ValueError):
        tb.call("simulate_promotion", {"product_id": "가전-03", "timeslot": "midnight", "weekend": 0,
                                       "discount_pct": 5, "installment_months": 0, "gift_ratio": 0})


def test_verify_numbers(setup):
    products, _, model = setup
    tb = ToolBox(model, products)
    out = tb.call("simulate_promotion", {"product_id": "가전-03", "timeslot": "prime", "weekend": 0,
                                         "discount_pct": 7, "installment_months": 0, "gift_ratio": 0})
    good = f"7% 할인 시 예상 주문은 {out['orders']:,}건, 순증 이익은 {out['incremental_profit']:,}원입니다."
    assert verify_numbers(good, [out])["ok"]
    bad = f"7% 할인 시 예상 주문은 {out['orders'] * 2:,}건입니다."
    check = verify_numbers(bad, [out])
    assert not check["ok"] and check["unmatched"]
