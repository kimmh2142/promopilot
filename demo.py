"""PromoPilot 시제품 End-to-End 데모.

실행: python demo.py
결과: 화면 출력 + results/demo_report.md 저장

흐름
 1) 정답 효과를 심은 합성 데이터 생성
 2) 반응 모델 학습 → 정답 복원 확인 (단순 회귀와 비교)
 3) MD 요청 시나리오: "화요일 저녁 프라임 방송, 이익률 20% 이상, 사은품 300개 안에서 최적안"
 4) 조건 변경: "카드사가 할인 비용의 50%를 분담하면?"
 5) 설명문 작성 → 수치 대조 검증 (일부러 틀린 설명도 넣어 잡아내는지 확인)
LLM 호출 없이 에이전트가 할 도구 호출 순서를 그대로 재현한다.
"""

from pathlib import Path

from promopilot import config as C
from promopilot.agent_tools import ToolBox, verify_numbers
from promopilot.data_gen import generate_broadcasts, make_products
from promopilot.model import ResponseModel, naive_estimates

TRUE_KEY = {"f_disc": "disc", "f_inst": "inst", "f_gift": "gift"}
FEATURE_LABEL = {"f_disc": "카드 할인", "f_inst": "무이자 할부", "f_gift": "사은품"}

lines: list[str] = []


def out(s: str = ""):
    print(s)
    lines.append(s)


def won(v) -> str:
    return f"{int(round(v)):,}원"


def combo(r) -> str:
    gift = f"사은품 {r['gift_ratio']:.0%}" if r["gift_ratio"] else "사은품 없음"
    inst = f"무이자 {int(r['installment_months'])}개월" if r["installment_months"] else "무이자 없음"
    return f"할인 {r['discount_pct']:g}% · {inst} · {gift}"


def main():
    # 1) 데이터 -------------------------------------------------------------
    products = make_products(seed=42)
    df = generate_broadcasts(3000, seed=42, products=products)
    out("# PromoPilot 시제품 데모 결과")
    out()
    out(f"- 합성 방송 데이터: {len(df):,}건, 상품 {df.product_id.nunique()}개, 카테고리 {df.category.nunique()}개")
    out("- 혜택은 인기 상품·프라임 시간대에 더 많이 배정되도록 **선택 편향**을 일부러 넣었습니다.")
    out()

    # 2) 학습과 복원 검증 ---------------------------------------------------
    model = ResponseModel(n_bootstrap=200, seed=0).fit(df)
    coefs = model.promo_coefficients()
    naive = naive_estimates(df)
    out("## 1. 파라미터 복원 테스트 (정답 효과를 되찾는가)")
    out()
    out("| 고객군 | 혜택 | 정답 | PromoPilot 추정 (95% 구간) | 단순 회귀 추정 |")
    out("|---|---|---|---|---|")
    for _, r in coefs.iterrows():
        truth = C.TRUE_EFFECTS[r.segment][TRUE_KEY[r.feature]]
        out(f"| {C.SEGMENT_LABELS[r.segment]} | {FEATURE_LABEL[r.feature]} | {truth:.2f} | "
            f"{r.estimate:.3f} ({r.ci_low:.3f}~{r.ci_high:.3f}) | {naive[r.segment][r.feature]:.3f} |")
    out()
    out("상품·시간대를 통제한 모델은 정답을 구간 안에서 되찾고, 통제 없는 단순 회귀는 효과를 크게 부풀립니다.")
    out()

    # 3) MD 요청 시나리오 ---------------------------------------------------
    tb = ToolBox(model, products)
    pid = "가전-03"
    price = products.set_index("product_id").loc[pid, "price"]
    request = {"product_id": pid, "timeslot": "prime", "weekend": 0, "min_margin": 0.20, "gift_stock": 300}
    out("## 2. MD 요청 시나리오")
    out()
    out(f'> "{pid}(판매가 {price:,}원) 화요일 저녁 프라임 방송, 이익률 20% 이상, 사은품 300개 안에서 순증 이익 최대 조합 찾아줘"')
    out()
    out(f"에이전트가 구조화한 도구 호출: `optimize_promotion({request})`")
    out()
    res = tb.call("optimize_promotion", request)
    base = res["baseline"]
    out(f"- 후보 {res['n_candidates']}개 중 제약을 통과한 조합 {res['n_feasible']}개")
    out(f"- 혜택 없음 기준선: 주문 {base['orders']:,}건, 이익 {won(base['profit'])}")
    out()
    out("| 순위 | 조합 | 예상 주문 (90% 범위) | 이익률 | 순증 이익 (90% 범위) |")
    out("|---|---|---|---|---|")
    for i, t in enumerate(res["top"], 1):
        out(f"| {i} | {combo(t)} | {t['orders']:,} ({t['orders_p05']:,}~{t['orders_p95']:,}) | "
            f"{t['profit_margin']:.1%} | {won(t['incremental_profit'])} "
            f"({won(t['incremental_profit_p05'])}~{won(t['incremental_profit_p95'])}) |")
    out()
    seg = res["top"][0]["orders_by_segment"]
    out("1위안 고객군별 예상 주문: " + ", ".join(f"{C.SEGMENT_LABELS[s]} {seg[s]:,}건" for s in C.SEGMENTS))
    out()

    # 4) 조건 변경 ---------------------------------------------------------
    out("## 3. 조건 변경: 카드사가 할인 비용의 50%를 분담하면?")
    out()
    res2 = tb.call("optimize_promotion", {**request, "card_share_hs": 0.5})
    t2 = res2["top"][0]
    out(f"- 추천 1위: {combo(t2)}, 순증 이익 {won(t2['incremental_profit'])}")
    if res2["best_extrapolated"]:
        e = res2["best_extrapolated"]
        out(f"- 참고: {combo(e)} 조합은 순증 이익이 {won(e['incremental_profit'])}으로 더 높게 계산되지만, "
            f"학습 범위 밖({'; '.join(e['extrapolation'])})이라 추천에서 빼고 **A/B 실험 후보**로 표시합니다.")
    out()

    # 5) 설명문 + 수치 대조 -------------------------------------------------
    out("## 4. 설명문 수치 대조 (LLM이 숫자를 지어내지 않는지 검사)")
    out()
    t1 = res["top"][0]
    explanation = (f"{combo(t1)} 조합이 1위입니다. 예상 주문은 {t1['orders']:,}건, "
                   f"순증 이익은 {t1['incremental_profit']:,}원입니다.")
    wrong = f"{combo(t1)} 조합이 1위입니다. 예상 주문은 {t1['orders'] + 500:,}건입니다."
    for label, text in [("정상 설명", explanation), ("틀린 숫자를 넣은 설명", wrong)]:
        chk = verify_numbers(text, [res])
        verdict = "통과" if chk["ok"] else f"차단 (도구 결과에 없는 숫자: {', '.join(chk['unmatched'])})"
        out(f"- {label}: \"{text}\" → **{verdict}**")
    out()
    out("> 합성 데이터 결과는 구현이 정확한지 확인하는 용도이며, 실제 프로모션 효과 크기를 뜻하지 않습니다.")

    Path("results").mkdir(exist_ok=True)
    Path("results/demo_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n[저장] results/demo_report.md")


if __name__ == "__main__":
    main()
