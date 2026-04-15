"""
battle_predictor.py

战斗仿真器（第5版）：
  - 并行模拟 B 局战斗，精确估算攻方胜率
  - 精度远高于神经网络近似，B=1000 时标准误差约 ±1.5%，运行时间约 0.02s
  - 接口与原 ML 版本完全一致，可直接替换

支持单位（攻方7种 + 守方额外1种AA炮）：
  infantry, mech_infantry, artillery, armour/tank,
  fighter, tactical_bomber, strategic_bomber/bomber
  守方额外：aa_gun / aaGun
  [Do Not Need to Know]
"""

from __future__ import annotations

import numpy as np

# 每次调用的模拟局数（精度 vs 速度权衡）
# B=500:  误差≈2.2%，极快;  B=1000: 误差≈1.5%，推荐;  B=2000: 误差≈1.1%
_DEFAULT_B = 1000

# 全局随机数生成器（固定种子保证可复现性）
_rng = np.random.default_rng(42)

# TripleA 单位名 → 模拟器内部参数名 映射
_UNIT_NAME_MAP: dict[str, str] = {
    "infantry":             "infantry",
    "mech_infantry":        "mech_infantry",
    "mechanized_infantry":  "mech_infantry",
    "mechanizedinfantry":   "mech_infantry",
    "artillery":            "artillery",
    "armour":               "tank",
    "armor":                "tank",
    "tank":                 "tank",
    "fighter":              "fighter",
    "tactical_bomber":      "tactical_bomber",
    "tacticalbomber":       "tactical_bomber",
    "tac_bomber":           "tactical_bomber",
    "strategic_bomber":     "strategic_bomber",
    "strategicbomber":      "strategic_bomber",
    "bomber":               "strategic_bomber",
    "aa_gun":               "aa_gun",
    "aagun":                "aa_gun",
    "aaguns":               "aa_gun",
    "aaGun":                "aa_gun",
    "anti_air":             "aa_gun",
    "antiaircraft":         "aa_gun",
}


def _parse_units(units_dict: dict) -> dict:
    """把 TripleA 单位 dict 规范化为模拟器参数 dict。"""
    result: dict[str, int] = {
        "infantry": 0, "mech_infantry": 0, "artillery": 0,
        "tank": 0, "fighter": 0, "tactical_bomber": 0,
        "strategic_bomber": 0, "aa_gun": 0,
    }
    for raw_name, count in units_dict.items():
        key = _UNIT_NAME_MAP.get(raw_name.lower().replace(" ", "_"))
        if key and count > 0:
            result[key] = result.get(key, 0) + int(count)
    return result


def sim_parallel(
    B: int,
    # 攻方
    ai: int, am: int, aa: int, at: int, af: int, atb: int, asb: int,
    # 守方
    di: int, dm: int, da: int, dt: int, df: int, dtb: int, dsb: int, daa: int,
) -> float:
    """
    并行蒙特卡洛战斗仿真，返回攻方胜率 [0.0, 1.0]。

    参数（攻方）:
      ai  — infantry（步兵）
      am  — mech_infantry（机械化步兵，并入步兵计算）
      aa  — artillery（炮兵，支援等数步兵使其命中率 1/6→1/3）
      at  — tank/armour（坦克）
      af  — fighter（战斗机）
      atb — tactical_bomber（战术轰炸机）
      asb — strategic_bomber（战略轰炸机）

    参数（守方）:
      di,dm,da,dt,df,dtb,dsb — 同攻方含义
      daa — aa_gun（AA炮，先手射击飞机，每门最多打3架，命中率1/6，只打第一轮）

    战斗规则（A&A Pacific 1940）:
      进攻：步兵（无支援）1/6；被炮兵支援 1/3；炮兵 1/3；坦克 1/2
            战机 1/2；战术轰炸机（配对战机/坦克）2/3，否则 1/2
            战略轰炸机 2/3（进攻时）
      防守：步兵/炮兵 1/3；坦克 1/2；战机 2/3；战术/战略轰炸机 1/2,1/6
      伤亡优先级：步兵 → 炮兵 → 坦克 → 飞机
      胜利条件：守方全灭 且 攻方至少剩1个单位（平局算攻方失败）
    """
    rng = _rng

    # 机步并入步兵（攻击值相同）
    ai = ai + am
    di = di + dm

    # ── 初始化 ──
    A_i  = np.full(B, ai,  dtype=np.int16)
    A_a  = np.full(B, aa,  dtype=np.int16)
    A_t  = np.full(B, at,  dtype=np.int16)
    A_f  = np.full(B, af,  dtype=np.int16)
    A_tb = np.full(B, atb, dtype=np.int16)
    A_sb = np.full(B, asb, dtype=np.int16)

    D_i  = np.full(B, di,  dtype=np.int16)
    D_a  = np.full(B, da,  dtype=np.int16)
    D_t  = np.full(B, dt,  dtype=np.int16)
    D_f  = np.full(B, df,  dtype=np.int16)
    D_tb = np.full(B, dtb, dtype=np.int16)
    D_sb = np.full(B, dsb, dtype=np.int16)
    D_aa = np.full(B, daa, dtype=np.int16)

    alive    = np.ones(B, dtype=bool)
    aa_fired = np.zeros(B, dtype=bool)

    # ── 战斗循环 ──
    while alive.any():
        idx = np.where(alive)[0]

        # AA炮先手射击（仅第一轮，只打飞机）
        need_aa = (~aa_fired[idx]) & (D_aa[idx] > 0) & ((A_f[idx] + A_tb[idx] + A_sb[idx]) > 0)
        if need_aa.any():
            j = idx[need_aa]
            shots = np.minimum(3 * D_aa[j], A_f[j] + A_tb[j] + A_sb[j]).astype(np.int16)
            hits  = rng.binomial(shots, 1/6).astype(np.int16)
            k = hits.copy()
            kill_f  = np.minimum(A_f[j],  k); A_f[j]  -= kill_f;  k -= kill_f
            kill_tb = np.minimum(A_tb[j], k); A_tb[j] -= kill_tb; k -= kill_tb
            kill_sb = np.minimum(A_sb[j], k); A_sb[j] -= kill_sb
            aa_fired[j] = True

        # 取当前存活批次数据
        Ai, Aa, At = A_i[idx], A_a[idx], A_t[idx]
        Af, Atb, Asb = A_f[idx], A_tb[idx], A_sb[idx]
        Di, Da, Dt = D_i[idx], D_a[idx], D_t[idx]
        Df, Dtb, Dsb = D_f[idx], D_tb[idx], D_sb[idx]

        # ── 攻方命中 ──
        sup   = np.minimum(Ai, Aa)   # 被炮兵支援的步兵数
        unsup = Ai - sup              # 未被支援的步兵数
        boosted_tb = np.minimum(Atb, Af + At)   # 配对战轰
        normal_tb  = Atb - boosted_tb

        a_hits = (
            rng.binomial(sup,        1/3) +
            rng.binomial(unsup,      1/6) +
            rng.binomial(Aa,         1/3) +
            rng.binomial(At,         1/2) +
            rng.binomial(Af,         1/2) +
            rng.binomial(boosted_tb, 2/3) +
            rng.binomial(normal_tb,  1/2) +
            rng.binomial(Asb,        2/3)
        )

        # ── 守方命中 ──
        d_hits = (
            rng.binomial(Di,  1/3) +
            rng.binomial(Da,  1/3) +
            rng.binomial(Dt,  1/2) +
            rng.binomial(Df,  2/3) +
            rng.binomial(Dtb, 1/2) +
            rng.binomial(Dsb, 1/6)
        )

        # ── 攻方减员（优先步兵→炮兵→坦克→飞机）──
        rem = d_hits.astype(np.int16)
        kill_Ai  = np.minimum(Ai,  rem); rem -= kill_Ai
        kill_Aa  = np.minimum(Aa,  rem); rem -= kill_Aa
        kill_At  = np.minimum(At,  rem); rem -= kill_At
        kill_Af  = np.minimum(Af,  rem); rem -= kill_Af
        kill_Atb = np.minimum(Atb, rem); rem -= kill_Atb
        kill_Asb = np.minimum(Asb, rem)

        # ── 守方减员 ──
        rem = a_hits.astype(np.int16)
        kill_Di  = np.minimum(Di,  rem); rem -= kill_Di
        kill_Da  = np.minimum(Da,  rem); rem -= kill_Da
        kill_Dt  = np.minimum(Dt,  rem); rem -= kill_Dt
        kill_Df  = np.minimum(Df,  rem); rem -= kill_Df
        kill_Dtb = np.minimum(Dtb, rem); rem -= kill_Dtb
        kill_Dsb = np.minimum(Dsb, rem)

        A_i[idx]  = Ai  - kill_Ai
        A_a[idx]  = Aa  - kill_Aa
        A_t[idx]  = At  - kill_At
        A_f[idx]  = Af  - kill_Af
        A_tb[idx] = Atb - kill_Atb
        A_sb[idx] = Asb - kill_Asb

        D_i[idx]  = Di  - kill_Di
        D_a[idx]  = Da  - kill_Da
        D_t[idx]  = Dt  - kill_Dt
        D_f[idx]  = Df  - kill_Df
        D_tb[idx] = Dtb - kill_Dtb
        D_sb[idx] = Dsb - kill_Dsb

        # 判断本局结束
        atk_dead = (
            (A_i[idx] <= 0) & (A_a[idx] <= 0) & (A_t[idx] <= 0) &
            (A_f[idx] <= 0) & (A_tb[idx] <= 0) & (A_sb[idx] <= 0)
        )
        def_dead = (
            (D_i[idx] <= 0) & (D_a[idx] <= 0) & (D_t[idx] <= 0) &
            (D_f[idx] <= 0) & (D_tb[idx] <= 0) & (D_sb[idx] <= 0)
        )
        alive[idx[atk_dead | def_dead]] = False

    # ── 胜负统计（守方全灭 且 攻方仍有单位 = 进攻胜） ──
    def_dead_all = (
        (D_i <= 0) & (D_a <= 0) & (D_t <= 0) &
        (D_f <= 0) & (D_tb <= 0) & (D_sb <= 0)
    )
    atk_alive = (
        (A_i > 0) | (A_a > 0) | (A_t > 0) |
        (A_f > 0) | (A_tb > 0) | (A_sb > 0)
    )
    wins = int(np.sum(def_dead_all & atk_alive))
    return wins / B


def predict_winrate(
    attacker_units: dict,
    defender_units: dict,
    B: int = _DEFAULT_B,
) -> float:
    """
    预测攻方获胜概率（蒙特卡洛仿真版本）。

    attacker_units: 攻方单位字典，如 {"infantry": 3, "armour": 1}
    defender_units: 守方单位字典，如 {"infantry": 2, "aaGun": 1}
    B: 模拟局数（默认 1000，误差约 ±1.5%）
    返回值: 0.0~1.0（越高表示攻方胜率越高）
    """
    a = _parse_units(attacker_units)
    d = _parse_units(defender_units)
    return sim_parallel(
        B,
        ai=a["infantry"],        am=a["mech_infantry"],   aa=a["artillery"],
        at=a["tank"],            af=a["fighter"],          atb=a["tactical_bomber"],
        asb=a["strategic_bomber"],
        di=d["infantry"],        dm=d["mech_infantry"],   da=d["artillery"],
        dt=d["tank"],            df=d["fighter"],          dtb=d["tactical_bomber"],
        dsb=d["strategic_bomber"], daa=d["aa_gun"],
    )


# ─────────────────────────────────────────────────────────────
# check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=== BattlePredictor (Monte Carlo Simulator) 测试 ===\n")
    cases = [
        ("3步兵 vs 2步兵",          {"infantry": 3},                    {"infantry": 2}),
        ("5步兵+2坦克 vs 2步兵+1炮", {"infantry": 5, "armour": 2},       {"infantry": 2, "artillery": 1}),
        ("1步兵 vs 3步兵+1AA炮",     {"infantry": 1},                    {"infantry": 3, "aaGun": 1}),
        ("2战机+3步兵 vs 2步兵+3AA", {"fighter": 2, "infantry": 3},      {"infantry": 2, "aaGun": 3}),
        ("坦克+步兵+炮兵 vs 2步兵",  {"armour": 1, "infantry": 1, "artillery": 1}, {"infantry": 2}),
    ]
    for desc, atk, dfn in cases:
        t0 = time.perf_counter()
        p  = predict_winrate(atk, dfn, B=2000)
        ms = (time.perf_counter() - t0) * 1000
        print(f"场景：{desc}")
        print(f"  攻方：{atk}")
        print(f"  守方：{dfn}")
        print(f"  胜率：{p:.1%}  (耗时 {ms:.1f}ms)\n")
