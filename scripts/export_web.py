"""웹 데모(docs/index.html)가 쓰는 모델 결과를 docs/model.json 으로 내보낸다.

실행: python scripts/export_web.py
웹 데모는 이 파일의 숫자만으로 계산하므로, 파이썬 모델과 같은 결과를 보여 준다.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promopilot import config as C  # noqa: E402
from promopilot.data_gen import generate_broadcasts, make_products  # noqa: E402
from promopilot.model import PROMO_FEATURES, ResponseModel, naive_estimates  # noqa: E402

TRUE_KEY = {"f_disc": "disc", "f_inst": "inst", "f_gift": "gift"}


def main():
    products = make_products(seed=42)
    df = generate_broadcasts(3000, seed=42, products=products)
    model = ResponseModel(n_bootstrap=200, seed=0).fit(df)
    naive = naive_estimates(df)

    seg = {}
    for s in C.SEGMENTS:
        coef = model.fits_[s].coef
        seg[s] = {
            "product": {p: coef[f"product[{p}]"] for p in model.products_},
            "slot": {t: (0.0 if t == "morning" else coef[f"slot[{t}]"]) for t in C.TIMESLOTS},
            "weekend": coef["weekend"],
            "promo": {f: coef[f] for f in PROMO_FEATURES},
        }

    coefs = model.promo_coefficients()
    recovery = [
        {
            "segment": C.SEGMENT_LABELS[r.segment],
            "feature": r.feature,
            "truth": C.TRUE_EFFECTS[r.segment][TRUE_KEY[r.feature]],
            "estimate": round(r.estimate, 3),
            "ci": [round(r.ci_low, 3), round(r.ci_high, 3)],
            "naive": round(float(naive[r.segment][r.feature]), 3),
        }
        for r in coefs.itertuples()
    ]

    out = {
        "products": [
            {"id": r.product_id, "category": r.category, "price": int(r.price), "return_rate": float(r.return_rate)}
            for r in products.itertuples()
        ],
        "timeslots": [{"id": t, "label": C.TIMESLOT_LABELS[t]} for t in C.TIMESLOTS],
        "segments": [{"id": s, "label": C.SEGMENT_LABELS[s]} for s in C.SEGMENTS],
        "model": seg,
        "observed_range": {k: {n: [float(v[0]), float(v[1])] for n, v in d.items()} for k, d in model.observed_range_.items()},
        "costs": C.DEFAULT_COSTS,
        "recovery": recovery,
        "n_broadcasts": len(df),
    }
    Path("docs").mkdir(exist_ok=True)
    Path("docs/model.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("[저장] docs/model.json")


if __name__ == "__main__":
    main()
