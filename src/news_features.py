"""
Extract stock-level and industry-level news features from financial news data.

Produces:
  (trade_date, ts_code) features: news_count, news_polarity              — 个股新闻
  (trade_date, ts_code) features: industry_news_count, industry_polarity — 所属行业新闻
"""

import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DATA_DIR, NEWS_DIR, PROCESSED_DIR

# 中文金融情感关键词
POSITIVE_KEYWORDS = [
    "涨停", "大涨", "利好", "盈利", "增长", "中标", "突破", "回购",
    "增持", "新高", "上升", "预增", "扭亏", "放量", "上攻", "超额",
    "创新高", "扩张", "签约", "交付", "获批",
]
NEGATIVE_KEYWORDS = [
    "跌停", "大跌", "利空", "亏损", "减持", "立案", "调查", "下跌",
    "风险", "违规", "处罚", "预警", "降级", "暂停",
    "违约", "冻结", "查封", "ST", "退市", "炸板",
]

# 行业新闻关键词 → industry 名（对应 basic.csv 中的 industry 字段）
INDUSTRY_KEYWORDS: dict[str, set[str]] = {
    "半导体": {"半导体"},
    "芯片": {"半导体"},
    "集成电路": {"半导体"},
    "新能源": {"电气设备"},
    "光伏": {"电气设备"},
    "锂电池": {"电气设备"},
    "锂电": {"电气设备"},
    "储能": {"电气设备"},
    "人工智能": {"软件服务"},
    "AI": {"软件服务"},
    "云计算": {"软件服务"},
    "大模型": {"软件服务"},
    "医药": {"医药", "化学制药", "生物制药", "医疗保健"},
    "医疗": {"医疗保健", "医药"},
    "创新药": {"医药"},
    "汽车": {"汽车", "汽车配件"},
    "新能源汽车": {"汽车"},
    "白酒": {"白酒", "食品"},
    "食品": {"食品"},
    "军工": {"航空", "船舶", "军工"},
    "航天": {"航空"},
    "房地产": {"房地产"},
    "地产": {"房地产"},
    "银行": {"银行"},
    "券商": {"证券"},
    "保险": {"保险"},
    "煤炭": {"煤炭"},
    "钢铁": {"钢铁"},
    "电力": {"电力"},
    "化工": {"化工"},
    "新材料": {"化工"},
    "通信": {"通信设备"},
    "5G": {"通信设备"},
    "传媒": {"影视音像", "广告包装", "互联网"},
    "游戏": {"互联网"},
    "机器人": {"机械基件", "专用机械"},
    "环保": {"环境保护"},
    "氢能": {"化工"},
    "黄金": {"黄金"},
    "有色": {"有色"},
    "农业": {"农业", "农林牧渔"},
    "养殖": {"农业", "农林牧渔"},
    "航运": {"水运", "空运", "运输"},
    "旅游": {"旅游景点", "旅游服务"},
}


def load_next_trade_date_map(trade_cal_path: Path) -> dict[str, str]:
    """日历日期 → 下一个交易日 的映射"""
    cal = pd.read_csv(trade_cal_path, dtype={"cal_date": str, "pretrade_date": str})
    cal = cal[cal["exchange"] == "SSE"].sort_values("cal_date")

    trading_dates = set(cal[cal["is_open"] == 1]["cal_date"].tolist())
    all_dates = sorted(cal["cal_date"].tolist())

    date_map = {}
    for d in all_dates:
        next_trade = d
        while next_trade not in trading_dates:
            idx = all_dates.index(next_trade)
            if idx + 1 < len(all_dates):
                next_trade = all_dates[idx + 1]
            else:
                next_trade = None
                break
        date_map[d] = next_trade

    return date_map


def build_stock_index(basic_path: Path) -> tuple[dict, dict]:
    """构建股票索引：
    - name_to_code: 公司全名 → ts_code
    - industry_to_codes: 行业名 → {ts_code, ...}
    - short_name_to_code: 简称（去尾部字母）→ ts_code
    """
    basic = pd.read_csv(basic_path, dtype={"ts_code": str})

    name_to_code = {}
    short_name_to_code = {}
    industry_to_codes: dict[str, set[str]] = {}

    for _, row in basic.iterrows():
        name = str(row["name"]).strip()
        code = row["ts_code"]
        industry = str(row["industry"]).strip() if pd.notna(row["industry"]) else ""

        if name and len(name) >= 2:
            name_to_code[name] = code

            # 简称：去尾部全角字母
            clean = name.rstrip("ＡＢＣＤＥＦＧＨ")
            if clean != name and len(clean) >= 2:
                short_name_to_code[clean] = code

            # 也处理 万科Ａ→万科 这类
            clean2 = name.replace("Ａ", "A").replace("＊", "*")
            for sf in "ABCDEFGH":
                idx = clean2.find(sf)
                if idx >= 2:
                    candidate = clean2[:idx].strip()
                    if len(candidate) >= 2:
                        short_name_to_code[candidate] = code

        if industry:
            if industry not in industry_to_codes:
                industry_to_codes[industry] = set()
            industry_to_codes[industry].add(code)

    # 按名称长度降序排列，优先匹配最长名称
    name_to_code = dict(sorted(name_to_code.items(), key=lambda x: -len(x[0])))
    short_name_to_code = dict(sorted(short_name_to_code.items(), key=lambda x: -len(x[0])))

    return name_to_code, short_name_to_code, industry_to_codes


def extract_stock_codes(text: str) -> list[str]:
    """从文本提取股票代码 000001.SZ / 603011.SH 格式"""
    return list(set(re.findall(r"(\d{6}\.(?:SZ|SH))", text)))


def match_stock_by_name(text: str, name_to_code: dict, short_name_to_code: dict) -> set[str]:
    """用公司名 + 简称匹配股票代码"""
    found: set[str] = set()
    for name, code in name_to_code.items():
        if name in text:
            found.add(code)
            break  # 全名匹配到一个就够了（最长的先匹配）
    for name, code in short_name_to_code.items():
        if name in text:
            found.add(code)
            break
    return found


def match_industry(text: str, industry_to_codes: dict) -> set[str]:
    """匹配行业新闻 → 返回该行业所有股票代码"""
    matched_industries = set()
    for kw, industries in INDUSTRY_KEYWORDS.items():
        if kw in text:
            matched_industries.update(industries)

    codes: set[str] = set()
    for ind in matched_industries:
        if ind in industry_to_codes:
            codes.update(industry_to_codes[ind])
    return codes


def simple_sentiment(text: str) -> float:
    """基于关键词的简单情感打分"""
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    return float(pos - neg)


def process_news_file(
    news_path: Path,
    name_to_code: dict,
    short_name_to_code: dict,
    industry_to_codes: dict,
    date_map: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    处理单日新闻文件，返回：
      stock_df: 个股级特征 (trade_date, ts_code, news_count, news_polarity)
      ind_df:   行业级特征 (trade_date, ts_code, industry_news_count, industry_polarity)
    """
    trade_date = news_path.stem
    mapped_date = date_map.get(trade_date)
    if mapped_date is None:
        return pd.DataFrame(), pd.DataFrame()

    try:
        df = pd.read_csv(news_path, dtype={"datetime": str})
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    stock_records = []
    industry_records = []

    for _, row in df.iterrows():
        content = str(row.get("content", "") or "")
        title = str(row.get("title", "") or "")
        full_text = content + " " + title

        # ── 个股级 ──
        codes = set(extract_stock_codes(full_text))
        codes |= match_stock_by_name(full_text, name_to_code, short_name_to_code)
        polarity = simple_sentiment(full_text)
        for code in codes:
            stock_records.append(
                {"trade_date": mapped_date, "ts_code": code, "polarity": polarity}
            )

        # ── 行业级 ── 如果没匹配到个股，再尝试行业匹配
        if not codes:
            ind_codes = match_industry(full_text, industry_to_codes)
            ind_polarity = simple_sentiment(full_text)
            for code in ind_codes:
                industry_records.append(
                    {"trade_date": mapped_date, "ts_code": code, "polarity": ind_polarity}
                )

    stock_df = _aggregate(stock_records, "news", mapped_date)
    ind_df = _aggregate(industry_records, "industry_news", mapped_date)
    return stock_df, ind_df


def _aggregate(records: list[dict], prefix: str, trade_date: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    agg = (
        result.groupby(["trade_date", "ts_code"])
        .agg(
            **{f"{prefix}_count": ("polarity", "count"), f"{prefix}_polarity": ("polarity", "sum")}
        )
        .reset_index()
    )
    agg[f"{prefix}_count"] = agg[f"{prefix}_count"].astype("float32")
    agg[f"{prefix}_polarity"] = agg[f"{prefix}_polarity"].astype("float32")
    agg["trade_date"] = agg["trade_date"].astype(str)
    return agg


def generate_news_features(
    news_dir: Path,
    basic_path: Path,
    trade_cal_path: Path,
    output_path: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """处理所有新闻文件，生成特征并保存"""
    print(">>> [新闻特征] 加载股票索引...")
    name_to_code, short_name_to_code, industry_to_codes = build_stock_index(basic_path)
    print(f"    {len(name_to_code)} 个全名 | {len(short_name_to_code)} 个简称 | {len(industry_to_codes)} 个行业")

    print(">>> [新闻特征] 构建交易日映射...")
    date_map = load_next_trade_date_map(trade_cal_path)

    print(">>> [新闻特征] 逐日处理新闻...")
    files = sorted(news_dir.glob("*.csv"))
    files = [f for f in files if start_date <= f.stem <= end_date]
    print(f"    {len(files)} 个新闻文件待处理")

    from tqdm import tqdm

    stock_frames = []
    ind_frames = []

    for p in tqdm(files, desc="Processing news"):
        stock_df, ind_df = process_news_file(
            p, name_to_code, short_name_to_code, industry_to_codes, date_map
        )
        if not stock_df.empty:
            stock_frames.append(stock_df)
        if not ind_df.empty:
            ind_frames.append(ind_df)

    result_frames = []

    if stock_frames:
        stock_result = pd.concat(stock_frames, ignore_index=True)
        print(f"    个股级新闻特征: {len(stock_result)} 条")
        result_frames.append(stock_result)
    if ind_frames:
        ind_result = pd.concat(ind_frames, ignore_index=True)
        print(f"    行业级新闻特征: {len(ind_result)} 条")
        result_frames.append(ind_result)

    if not result_frames:
        print("    ! 无新闻特征生成")
        return pd.DataFrame()

    result = pd.concat(result_frames, ignore_index=True)
    result = result.groupby(["trade_date", "ts_code"]).first().reset_index()
    print(f"    合并后总计: {len(result)} 条")

    result.to_parquet(output_path, index=False)
    print(f">>> [新闻特征] 已保存到 {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate news-based features.")
    parser.add_argument("--start", type=str, default="20190101")
    parser.add_argument("--end", type=str, default="20260529")
    parser.add_argument(
        "--out", type=str, default=str(PROCESSED_DIR / "news_features.parquet")
    )
    args = parser.parse_args()

    generate_news_features(
        news_dir=NEWS_DIR,
        basic_path=DATA_DIR / "basic.csv",
        trade_cal_path=DATA_DIR / "trade_cal.csv",
        output_path=Path(args.out),
        start_date=args.start,
        end_date=args.end,
    )


if __name__ == "__main__":
    main()
