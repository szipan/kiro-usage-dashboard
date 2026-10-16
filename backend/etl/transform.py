"""ETL 核心转换：Kiro per-user CSV -> Parquet 长表。

产出两张事实表（按 year/month/day 分区）：
  fact_user_daily        一行 / 用户 / 天（跨 client 聚合 + 派生字段 + 部门）
  fact_user_model_daily  一行 / 用户 / 天 / 模型（模型消息数长表）

设计见 design.md §3.3。此模块不依赖 AWS，可本地对 mock 数据运行；
Lambda handler（handler.py）负责 S3 读写与调用本模块。
"""
from __future__ import annotations

import io
import glob
import os
from typing import Iterable

import pandas as pd

# 与 shared.derivations 保持一致；ETL 打包时把 shared 一并纳入
try:
    from backend.shared import derivations as dv
except ImportError:  # Lambda 内 shared 与本模块同级
    import derivations as dv  # type: ignore

STATIC_NUMERIC = [
    "Chat_Conversations", "Credits_Used", "Overage_Cap",
    "Overage_Credits_Used", "Total_Messages", "Usage_Limit",
]


def _read_csv(buf) -> pd.DataFrame:
    df = pd.read_csv(buf, dtype=str).fillna("")
    # 早期导出 / mock 生成器没有 Usage_Limit 列：补 0，下游即按 Overage_Cap 回退（见 transform 派生段）
    if "Usage_Limit" not in df.columns:
        df["Usage_Limit"] = "0"
    numeric_cols = STATIC_NUMERIC + [c for c in df.columns if c.endswith("_messages")]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 布尔列：真实 CSV 里可能缺列，或被解析成非 Series；统一稳健处理。
    def _bool_col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].astype(str).str.strip().str.lower() == "true"
        return pd.Series([False] * len(df), index=df.index)

    df["Overage_Enabled"] = _bool_col("Overage_Enabled")
    df["New_User"] = _bool_col("New_User")

    # 邮箱标准化
    if "User_Email" in df.columns:
        df["User_Email"] = df["User_Email"].astype(str).str.strip().str.lower()

    # 档位大小写规范化：真实数据是 POWER/PRO 等全大写，统一为 Pro/ProPlus/ProMax/Power。
    if "Subscription_Tier" in df.columns:
        df["Subscription_Tier"] = df["Subscription_Tier"].astype(str).map(_normalize_tier)
    return df


_TIER_MAP = {
    "pro": "Pro", "proplus": "ProPlus", "pro_plus": "ProPlus", "pro plus": "ProPlus",
    "promax": "ProMax", "pro_max": "ProMax", "pro max": "ProMax", "power": "Power",
}


def _normalize_tier(v: str) -> str:
    key = (v or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return _TIER_MAP.get(key, v.strip().title() if v else "")


def transform(
    frames: Iterable[pd.DataFrame],
    department_map: dict[str, dict],
    credit_unit_price: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把多个 CSV DataFrame 规整为 (fact_user_daily, fact_user_model_daily)。"""
    df = pd.concat(list(frames), ignore_index=True)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    model_cols = [c for c in df.columns if c.endswith("_messages")]

    # --- 跨 client 按 (Date, User_Email) 聚合 ---
    agg = {
        "Chat_Conversations": "sum",
        "Credits_Used": "sum",
        "Overage_Cap": "max",            # 同一用户上限一致，取 max 稳健
        "Overage_Credits_Used": "sum",
        "Total_Messages": "sum",
        "Usage_Limit": "max",            # 套餐额度（每用户固定，取 max）
        "Overage_Enabled": "max",        # 任一 client 开启即视为开启
        "New_User": "max",
        "Subscription_Tier": "first",
        "UserId": "first",
        "ProfileId": "first",
    }
    for mc in model_cols:
        agg[mc] = "sum"

    daily = df.groupby(["Date", "User_Email"], as_index=False).agg(agg)

    # --- 部门 join（来自 Identity Center 同步）---
    def _dept(email: str) -> str:
        info = department_map.get(email)
        return info.get("department", "未分组") if info else "未分组"

    def _name(email: str) -> str:
        info = department_map.get(email)
        return (info or {}).get("display_name", "") or email.split("@")[0]

    daily["department"] = daily["User_Email"].map(_dept)
    daily["display_name"] = daily["User_Email"].map(_name)

    # --- 派生字段（统一口径：基于 Usage_Limit 套餐额度）---
    # 回退：若 Usage_Limit 缺失/为0（早期 CSV 无此列），退回用 Overage_Cap。
    def _limit(r):
        ul = r.get("Usage_Limit", 0) or 0
        return ul if ul > 0 else r["Overage_Cap"]
    daily["usage_rate"] = daily.apply(
        lambda r: dv.usage_rate(r["Credits_Used"], _limit(r)), axis=1)
    daily["balance"] = daily.apply(
        lambda r: dv.balance(r["Credits_Used"], _limit(r)), axis=1)
    daily["is_overage"] = daily.apply(
        lambda r: dv.is_overage(r["Credits_Used"], _limit(r)), axis=1)
    # est_cost 在日表里只放**当天的 overage 成本**（可按天累加）；
    # 订阅月费是席位级整月固定，不能逐行加，由聚合查询按档位补。
    # overage 成本 = Overage_Credits_Used * 单价（未开 overage 的客户恒=0）。
    daily["est_cost"] = daily["Overage_Credits_Used"].map(
        lambda oc: dv.overage_cost(oc, credit_unit_price))

    # 规范列名为 snake_case，便于 Athena/前端
    fact_user_daily = daily.rename(columns={
        "Date": "date", "User_Email": "email", "UserId": "user_id",
        "Chat_Conversations": "chat_conversations", "Credits_Used": "credits_used",
        "Overage_Cap": "overage_cap", "Overage_Credits_Used": "overage_credits_used",
        "Overage_Enabled": "overage_enabled", "ProfileId": "profile_id",
        "Subscription_Tier": "subscription_tier", "Total_Messages": "total_messages",
        "New_User": "new_user", "Usage_Limit": "usage_limit",
    })[[
        "date", "user_id", "email", "display_name", "department",
        "subscription_tier", "chat_conversations", "credits_used", "overage_cap",
        "overage_credits_used", "overage_enabled", "total_messages", "new_user",
        "usage_rate", "balance", "est_cost", "profile_id", "usage_limit", "is_overage",
    ]]

    # 【口径：仅显示有用量的用户】看板只统计出现在 Kiro CSV 里的用户
    # （= 实际激活并有活动的用户）。pending/未激活的订阅用户不会出现在 CSV，
    # 因此天然被排除，激活率等指标才准确。故此处**不**为全体订阅用户补 0 行。

    # 强制 Parquet 列类型与 Glue 表 schema 一致（否则 Athena 报 type incompatible）。
    # bigint 列须为 int64；double 列为 float64；boolean 为 bool；其余 string。
    fact_user_daily = _cast_types(fact_user_daily, {
        "chat_conversations": "int64", "total_messages": "int64",
        "credits_used": "float64", "overage_cap": "float64",
        "overage_credits_used": "float64", "usage_rate": "float64",
        "balance": "float64", "est_cost": "float64", "usage_limit": "float64",
        "overage_enabled": "bool", "new_user": "bool", "is_overage": "bool",
    })

    # --- 模型长表 ---
    long = daily.melt(
        id_vars=["Date", "User_Email"],
        value_vars=model_cols,
        var_name="model",
        value_name="model_messages",
    )
    long["model"] = long["model"].str.replace("_messages$", "", regex=True)
    long = long[long["model_messages"] > 0]
    fact_user_model_daily = long.rename(
        columns={"Date": "date", "User_Email": "email"})[
        ["date", "email", "model", "model_messages"]]
    fact_user_model_daily = _cast_types(fact_user_model_daily, {"model_messages": "int64"})

    return fact_user_daily.reset_index(drop=True), fact_user_model_daily.reset_index(drop=True)


def _cast_types(df: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    """按目标类型强制转换列，确保写出的 Parquet 类型与 Glue 表 schema 一致。"""
    if df.empty:
        return df
    for col, t in types.items():
        if col not in df.columns:
            continue
        if t == "int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype("int64")
        elif t == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
        elif t == "bool":
            df[col] = df[col].astype("bool")
    return df


def run_local(csv_glob: str, identity_json: str, credit_unit_price: float = 0.15):
    """本地便捷入口：对 mock 数据跑转换并返回两张表。"""
    import json
    with open(identity_json, encoding="utf-8") as f:
        dept_map = json.load(f)
    files = glob.glob(csv_glob, recursive=True)
    frames = [_read_csv(p) for p in files]
    return transform(frames, dept_map, credit_unit_price)
