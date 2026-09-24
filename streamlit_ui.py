"""Streamlit user interface for race-result analysis and race planning."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from race_planner import build_nutrition_plan, build_race_plan, parse_gpx, summarize_course
from race_score_model import RaceScoreModel
from scraper import scrape_itra_results


def configure_page() -> None:
    st.set_page_config(page_title="Trail Race Lab", page_icon="⛰️", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1440px; padding-top: 1.6rem; padding-bottom: 3rem;}
        [data-testid="stSidebar"] {border-right: 1px solid #e3eae6;}
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e1e9e4; border-radius: 14px;
            padding: 14px 16px; box-shadow: 0 2px 8px rgba(24, 58, 43, .04);
        }
        [data-testid="stForm"] {
            background: #ffffff; border: 1px solid #e1e9e4; border-radius: 16px;
            padding: 1.1rem 1.2rem 1.3rem;
        }
        .step-row {display:flex; gap:10px; margin:.4rem 0 1.2rem;}
        .step-pill {
            flex:1; padding:10px 12px; border-radius:12px; background:#edf4f0;
            color:#2f5e49; font-size:.88rem; font-weight:650; text-align:center;
        }
        .section-note {
            padding: 12px 14px; border-left: 4px solid #4f7d64; border-radius: 8px;
            background: #f0f6f2; color: #355544; margin: .5rem 0 1rem;
        }
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            min-height: 2.75rem; border-radius: 10px; font-weight: 650;
        }
        h1, h2, h3 {letter-spacing: -.02em;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_duration(value: str) -> int:
    parts = str(value).strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError("时间请使用 HH:MM 或 HH:MM:SS 格式。")
    try:
        hours, minutes = int(parts[0]), int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError("时间请使用 HH:MM 或 HH:MM:SS 格式。") from exc
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("时间数值无效。")
    return hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    total = max(int(round(float(seconds))), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def pace_label(minutes_per_km: float) -> str:
    total_seconds = int(round(float(minutes_per_km) * 60))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}/km"


def normalize_result_time(time_str: object) -> float | None:
    if time_str in (None, "N/A", ""):
        return None
    try:
        return float(parse_duration(str(time_str)))
    except ValueError:
        return None


def normalize_gender(gender: object) -> str | None:
    value = str(gender or "").strip().upper()
    if value == "M":
        return "男子"
    if value == "F":
        return "女子"
    return None


def normalize_age_group(age: object) -> str | None:
    value = str(age or "").strip()
    if value.upper() in {"", "-", "N/A", "NA", "NONE", "NAN"}:
        return None
    if re.fullmatch(r"\d{1,3}\s*-\s*\d{1,3}", value):
        start, end = re.findall(r"\d+", value)
        return f"{int(start)}-{int(end)}"
    try:
        numeric_age = int(float(value))
        return f"{(numeric_age // 10) * 10}-{(numeric_age // 10) * 10 + 9}"
    except ValueError:
        return None


def age_group_sort_key(value: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)", str(value))
    return (int(match.group(1)), str(value)) if match else (9999, str(value))


def prepare_results_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize scraped result fields for readable filtering and charts."""
    prepared = frame.copy()
    prepared["time_seconds"] = prepared["time"].apply(normalize_result_time)
    prepared["finish_hours"] = prepared["time_seconds"] / 3600.0
    prepared["position_numeric"] = pd.to_numeric(prepared["position"], errors="coerce")
    prepared["performance_index"] = pd.to_numeric(
        prepared["performance_index"], errors="coerce"
    )
    prepared["gender_label"] = prepared["gender"].apply(normalize_gender)
    prepared["age_group"] = prepared["age"].apply(normalize_age_group)
    return prepared


def apply_result_filters(
    frame: pd.DataFrame,
    genders: list[str],
    nationalities: list[str],
    age_groups: list[str],
) -> pd.DataFrame:
    """Apply only non-empty filters; an empty selection means show everything."""
    mask = pd.Series(True, index=frame.index)
    if genders:
        mask &= frame["gender_label"].isin(genders)
    if nationalities:
        mask &= frame["nationality"].isin(nationalities)
    if age_groups:
        mask &= frame["age_group"].isin(age_groups)
    return frame[mask]


def add_filtered_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Order selected results by overall finish and add a continuous rank."""
    ranked = frame.sort_values(
        ["position_numeric", "time_seconds"], kind="stable", na_position="last"
    ).copy()
    ranked["filtered_rank"] = range(1, len(ranked) + 1)
    return ranked


def age_distribution_figure(frame: pd.DataFrame, age_order: list[str]) -> go.Figure:
    """Create a stacked age distribution using only valid age and M/F data."""
    age_chart = frame.dropna(subset=["age_group", "gender_label"])
    age_chart = age_chart[age_chart["gender_label"].isin(["男子", "女子"])]
    figure = px.histogram(
        age_chart,
        x="age_group",
        color="gender_label",
        barmode="stack",
        title="年龄组分布",
        category_orders={"age_group": age_order, "gender_label": ["男子", "女子"]},
        color_discrete_map={"男子": "#2F6BFF", "女子": "#E0528D"},
        labels={"age_group": "年龄组", "gender_label": "性别", "count": "人数"},
    )
    figure.update_yaxes(title="人数", tickformat="d")
    figure.update_layout(legend_title_text="性别")
    return figure


def load_race_score_model() -> RaceScoreModel:
    # Keep this uncached: Streamlit can otherwise retain an instance created
    # from an older RaceScoreModel class after a hot reload.
    artifact = Path(__file__).resolve().parent / "model_artifacts" / "race_score_model_v1.json"
    return RaceScoreModel.load(artifact)


def estimate_race_scores(
    frame: pd.DataFrame,
    model: RaceScoreModel,
    *,
    distance_km: float,
    elevation_gain_m: float,
    anchor_time: str | float | None = None,
    anchor_score: float | None = None,
    anchors: list[tuple[str | float, float]] | None = None,
) -> pd.DataFrame:
    """Estimate a Race Score for every row with a valid finish time."""
    estimated = frame.copy()
    for column in [
        "estimated_race_score", "score_lower_80", "score_upper_80",
        "score_lower_95", "score_upper_95",
    ]:
        estimated[column] = np.nan
    estimated["score_confidence"] = None
    estimated["score_method"] = None

    for index, row in estimated.dropna(subset=["finish_hours"]).iterrows():
        if anchors is not None:
            result = model.estimate_with_anchors(
                finish_time=float(row["finish_hours"]), anchors=anchors
            )
        else:
            result = model.estimate(
                distance_km=distance_km,
                elevation_gain_m=elevation_gain_m,
                finish_time=float(row["finish_hours"]),
                anchor_time=anchor_time,
                anchor_score=anchor_score,
            )
        estimated.at[index, "estimated_race_score"] = result.score
        estimated.at[index, "score_lower_80"] = result.lower_80
        estimated.at[index, "score_upper_80"] = result.upper_80
        estimated.at[index, "score_lower_95"] = result.lower_95
        estimated.at[index, "score_upper_95"] = result.upper_95
        estimated.at[index, "score_confidence"] = result.confidence
        estimated.at[index, "score_method"] = result.method
    return estimated


def score_interval_label(row: pd.Series) -> str:
    if pd.isna(row.get("score_lower_80")) or pd.isna(row.get("score_upper_80")):
        return "—"
    return f"{row['score_lower_80']:.1f}–{row['score_upper_80']:.1f}"


def parse_anchor_table(table: pd.DataFrame) -> list[tuple[str, float]]:
    anchors: list[tuple[str, float]] = []
    for row in table.to_dict("records"):
        time_value = str(row.get("finish_time", "") or "").strip()
        score_value = row.get("race_score")
        if not time_value and pd.isna(score_value):
            continue
        if not time_value or pd.isna(score_value):
            raise ValueError("每个锚点必须同时填写完赛时间和 Race Score。")
        parse_duration(time_value)
        anchors.append((time_value, float(score_value)))
    if not anchors:
        raise ValueError("请至少填写一个有效锚点。")
    return anchors


def render_race_score_estimator(
    frame: pd.DataFrame,
    course_info: dict[str, float] | None = None,
) -> pd.DataFrame:
    st.subheader("Race Score 预估")
    st.caption(
        "这是单场比赛表现分的离线估算，不是选手个人 ITRA Index，也不是 ITRA 官方出分。"
    )
    mode = st.segmented_control(
        "估算方式",
        ["仅赛道参数（无需账号）", "同场成绩锚点（高置信度）"],
        default="仅赛道参数（无需账号）",
        key="race_score_mode",
    )
    with st.form("race_score_estimator"):
        if mode == "同场成绩锚点（高置信度）":
            st.markdown(
                '<div class="section-note"><b>高置信度</b><br>'
                '可输入同一场比赛多位选手的官方 Race Score 与对应完赛时间。'
                '模型稳健聚合“分数 × 时间”，并自动识别明显异常锚点；建议选择 3–5 个、'
                '覆盖不同完赛时段的锚点。</div>',
                unsafe_allow_html=True,
            )
            anchor_table = st.data_editor(
                pd.DataFrame([
                    {"finish_time": "10:00:00", "race_score": 700.0},
                ]),
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                key="race_score_anchors",
                column_config={
                    "finish_time": st.column_config.TextColumn(
                        "完赛时间", help="HH:MM:SS，可超过 24 小时", required=True
                    ),
                    "race_score": st.column_config.NumberColumn(
                        "官方 Race Score", min_value=1.0, max_value=1000.0,
                        step=1.0, format="%.1f", required=True,
                    ),
                },
            )
            anchors = None
            distance_km, elevation_gain_m = 1.0, 0.0
        else:
            course_info = course_info or {}
            has_course_info = {"distance_km", "elevation_gain_m"}.issubset(course_info)
            if has_course_info:
                distance_km = float(course_info["distance_km"])
                elevation_gain_m = float(course_info["elevation_gain_m"])
                st.success(
                    f"已从 ITRA 赛事详情自动读取：{distance_km:.2f} km / "
                    f"D+ {elevation_gain_m:.0f} m，无需手动输入。"
                )
            else:
                st.warning(
                    "该 ITRA 页面没有返回完整赛道参数，请补充缺失值。"
                    "当前验证 MAE 约 34.8 分，95% 误差约 ±67.7 分。"
                )
                course1, course2 = st.columns(2)
                with course1:
                    distance_km = st.number_input(
                        "比赛距离 km", min_value=1.0, max_value=500.0,
                        value=float(course_info.get("distance_km", 50.0)), step=1.0,
                    )
                with course2:
                    elevation_gain_m = st.number_input(
                        "累计爬升 m", min_value=0.0, max_value=30000.0,
                        value=float(course_info.get("elevation_gain_m", 2500.0)), step=100.0,
                    )
            anchors = None
        calculate = st.form_submit_button(
            "计算整场 Race Score", type="primary", use_container_width=True
        )

    if calculate:
        try:
            if mode == "同场成绩锚点（高置信度）":
                anchors = parse_anchor_table(anchor_table)
            estimated = estimate_race_scores(
                frame,
                load_race_score_model(),
                distance_km=float(distance_km),
                elevation_gain_m=float(elevation_gain_m),
                anchors=anchors,
            )
            calibration = (
                load_race_score_model().calibrate_anchors(anchors)
                if anchors is not None else None
            )
            st.session_state["results_frame"] = estimated
            st.session_state["race_score_meta"] = {
                "mode": mode,
                "anchor_count": calibration.anchor_count if calibration else 0,
                "anchor_used_count": calibration.used_count if calibration else 0,
                "anchor_rejected_count": calibration.rejected_count if calibration else 0,
                "distance_km": distance_km,
                "elevation_gain_m": elevation_gain_m,
            }
            st.success(f"已为 {estimated['estimated_race_score'].notna().sum():,} 名完赛者生成预估分数。")
            return estimated
        except Exception as exc:
            st.error(f"Race Score 估算失败：{exc}")
    return frame


@st.cache_data(show_spinner=False)
def load_course(payload: bytes) -> pd.DataFrame:
    return parse_gpx(payload)


def render_sidebar() -> str:
    with st.sidebar:
        st.title("Trail Race Lab")
        st.caption("比赛数据、配速与补给规划")
        page = st.radio(
            "功能",
            ["比赛计划与补给", "ITRA 成绩分析"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        if page == "比赛计划与补给":
            st.markdown("**快速流程**")
            st.caption("① 上传 GPX\n\n② 设置时间与站点\n\n③ 生成并下载计划")
            with st.expander("模型说明"):
                st.caption(
                    "使用平滑高程、坡度配速代价、线性疲劳与基础海拔修正。"
                    "输出是时间预算，不是逐秒 GPS 配速指令。"
                )
        else:
            st.caption("输入公开 ITRA 比赛结果链接，抓取并筛选全部完赛记录。")
    return page


def render_results_page() -> None:
    st.title("ITRA 比赛成绩分析")
    st.caption("抓取公开比赛结果，查看完赛时间、名次、年龄与国籍分布。")

    with st.form("results_query"):
        url = st.text_input(
            "ITRA 比赛结果 URL",
            placeholder="https://itra.run/Races/RaceResults/70K/2024/94006",
        )
        option1, option2 = st.columns([1, 1])
        with option1:
            include_pi = st.checkbox(
                "抓取个人 ITRA Index（较慢）",
                help="每名选手需要额外访问一次个人页面；Race Score 不是这个数值。",
            )
        with option2:
            pi_limit = st.number_input(
                "最多查询选手数", 1, 200, 50, 10,
                disabled=not include_pi,
            )
        submitted = st.form_submit_button("分析比赛成绩", type="primary", use_container_width=True)

    if submitted:
        if not url.strip():
            st.error("请先输入有效的 ITRA 比赛结果 URL。")
        else:
            try:
                with st.spinner("正在获取比赛结果..."):
                    results, course_info = scrape_itra_results(
                        url.strip(),
                        include_performance_index=include_pi,
                        performance_index_limit=int(pi_limit) if include_pi else 0,
                        return_course_info=True,
                    )
                if not results:
                    st.warning("没有找到比赛结果，请检查 URL。")
                else:
                    frame = prepare_results_frame(pd.DataFrame(results))
                    if {"distance_km", "elevation_gain_m"}.issubset(course_info):
                        frame = estimate_race_scores(
                            frame,
                            load_race_score_model(),
                            distance_km=float(course_info["distance_km"]),
                            elevation_gain_m=float(course_info["elevation_gain_m"]),
                        )
                        st.session_state["race_score_meta"] = {
                            "mode": "仅赛道参数（ITRA 自动读取）",
                            **course_info,
                        }
                    else:
                        st.session_state.pop("race_score_meta", None)
                    st.session_state["results_frame"] = frame
                    st.session_state["results_course_info"] = course_info
                    st.session_state["results_include_pi"] = include_pi
            except Exception as exc:
                st.error(f"获取失败：{exc}")

    frame = st.session_state.get("results_frame")
    if frame is None:
        st.info("输入比赛链接后开始分析。结果会保留，调整筛选条件不会重新抓取。")
        return
    # Migrate data already held in session state after UI/schema updates.
    frame = prepare_results_frame(frame)
    st.session_state["results_frame"] = frame

    frame = render_race_score_estimator(
        frame, st.session_state.get("results_course_info", {})
    )
    has_score = "estimated_race_score" in frame and frame["estimated_race_score"].notna().any()
    if has_score:
        meta = st.session_state.get("race_score_meta", {})
        score_values = frame["estimated_race_score"].dropna()
        score1, score2, score3 = st.columns(3)
        score1.metric("最高预估 Race Score", f"{score_values.max():.1f}")
        score2.metric("完赛者预估中位数", f"{score_values.median():.1f}")
        score3.metric("估算方式", "同场锚点" if "锚点" in meta.get("mode", "") else "仅赛道参数")

    st.subheader("筛选")
    filter1, filter2, filter3 = st.columns(3)
    genders = sorted(frame["gender_label"].dropna().unique())
    nations = sorted(
        value for value in frame["nationality"].dropna().unique()
        if str(value).strip().upper() not in {"", "-", "N/A", "NA"}
    )
    ages = sorted(frame["age_group"].dropna().unique(), key=age_group_sort_key)
    with filter1:
        selected_genders = st.multiselect("性别", genders, default=[])
    with filter2:
        selected_nations = st.multiselect("国籍", nations, default=[])
    with filter3:
        selected_ages = st.multiselect("年龄组", ages, default=[])
    filtered = apply_result_filters(
        frame, selected_genders, selected_nations, selected_ages
    )
    filters_active = bool(selected_genders or selected_nations or selected_ages)
    if filters_active:
        filtered = add_filtered_ranking(filtered)
    st.caption(
        "筛选框留空表示显示全部。赛事源数据中的非标准性别代码、缺失年龄组和无效分类"
        "不会出现在筛选选项或年龄组图表中。"
    )
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("当前选手", f"{len(filtered):,}")
    metric2.metric("总完赛人数", f"{len(frame):,}")
    metric3.metric("最快成绩", format_duration(filtered["time_seconds"].min()) if len(filtered) else "—")
    if filtered.empty:
        st.warning("当前筛选条件下没有数据。")
        return

    chart_tab, table_tab, score_tab = st.tabs(["图表", "成绩表", "Race Score"])
    with chart_tab:
        left, right = st.columns(2)
        with left:
            time_chart = filtered.dropna(subset=["finish_hours", "position_numeric"])
            fig = px.scatter(
                time_chart, x="finish_hours", y="position_numeric",
                hover_data=["name", "age", "nationality", "time"],
                title="完赛时间与名次",
                labels={
                    "finish_hours": "完赛时间（小时）",
                    "position_numeric": "总名次",
                },
            )
            fig.update_xaxes(tickformat=".1f")
            fig.update_yaxes(autorange="reversed", tickformat="d")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.plotly_chart(
                age_distribution_figure(filtered, ages), use_container_width=True
            )
            st.caption(
                "每个年龄组按性别累计显示：蓝色为男子、粉色为女子；"
                "只统计同时具有有效年龄组和标准 M/F 性别的记录，"
                "不需要额外抓取个人 ITRA Index。"
            )
        index_frame = filtered.dropna(subset=["time_seconds", "performance_index"])
        if not index_frame.empty:
            fig = px.scatter(
                index_frame, x="finish_hours", y="performance_index",
                hover_data=["name", "age", "nationality", "time"],
                title="比赛时间与个人 ITRA Index",
                labels={
                    "finish_hours": "完赛时间（小时）",
                    "performance_index": "个人 ITRA Index",
                },
            )
            fig.update_xaxes(tickformat=".1f")
            st.plotly_chart(fig, use_container_width=True)
    with table_tab:
        display = filtered.copy()
        display["time"] = display["time_seconds"].apply(format_duration)
        display_columns = [
            "position", "name", "time", "performance_index", "age_group",
            "gender_label", "nationality",
        ]
        if has_score:
            display["score_80_interval"] = display.apply(score_interval_label, axis=1)
            display_columns[3:3] = ["estimated_race_score", "score_80_interval"]
        if filters_active:
            display_columns.insert(0, "filtered_rank")
        st.dataframe(
            display[display_columns],
            column_config={
                "filtered_rank": "排名", "position": "总名次", "name": "姓名",
                "performance_index": "ITRA Index",
                "estimated_race_score": st.column_config.NumberColumn(
                    "预估 Race Score", format="%.1f"
                ),
                "score_80_interval": "80% 估算区间",
                "time": "完赛时间", "age_group": "年龄组", "gender_label": "性别",
                "nationality": "国籍",
            },
            hide_index=True,
            use_container_width=True,
        )
    with score_tab:
        if not has_score:
            st.info("请先在页面上方选择估算方式并点击“计算整场 Race Score”。")
        else:
            score_frame = filtered.dropna(subset=["finish_hours", "estimated_race_score"])
            score_fig = px.scatter(
                score_frame,
                x="finish_hours",
                y="estimated_race_score",
                hover_data=["name", "position", "time"],
                title="完赛时间与预估 Race Score",
                labels={
                    "finish_hours": "完赛时间（小时）",
                    "estimated_race_score": "预估 Race Score",
                },
            )
            score_fig.update_xaxes(tickformat=".1f")
            score_fig.update_yaxes(tickformat=".0f", range=[0, 1000])
            st.plotly_chart(score_fig, use_container_width=True)
            meta = st.session_state.get("race_score_meta", {})
            if "锚点" in meta.get("mode", ""):
                st.success(
                    f"已输入 {int(meta.get('anchor_count', 0))} 个同场锚点，"
                    f"稳健校准采用 {int(meta.get('anchor_used_count', 0))} 个，"
                    f"排除 {int(meta.get('anchor_rejected_count', 0))} 个明显异常值。"
                    "表中同时给出按有效锚点数量和锚点离散度调整后的 80% 估算区间。"
                )
            else:
                st.warning(
                    f"赛道参数：{float(meta.get('distance_km', 0)):.1f} km / "
                    f"D+ {float(meta.get('elevation_gain_m', 0)):.0f} m。"
                    "该模式验证误差较大，不能替代官方 Race Score。"
                )


def default_checkpoints(total_km: float) -> pd.DataFrame:
    if total_km < 12:
        distances = [total_km / 2]
    else:
        interval = 20.0 if total_km >= 45 else 10.0
        distances = list(np.arange(interval, total_km - 1.0, interval))
        if not distances:
            distances = [total_km / 2]
    return pd.DataFrame(
        [
            {"name": f"CP{i + 1}", "distance_km": round(distance, 1), "stop_min": 4.0}
            for i, distance in enumerate(distances[:8])
        ]
    )


def course_figure(course: pd.DataFrame, plan: pd.DataFrame | None = None) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=course["distance_km"], y=course["elevation_m"], mode="lines",
            name="海拔", fill="tozeroy", line={"color": "#3f7457", "width": 2},
            fillcolor="rgba(75, 125, 92, .18)",
        )
    )
    if plan is not None and not plan.empty:
        distances = plan["end_km"].to_numpy()
        elevations = np.interp(distances, course["distance_km"], course["elevation_m"])
        figure.add_trace(
            go.Scatter(
                x=distances, y=elevations, mode="markers+text", text=plan["to"],
                textposition="top center", name="站点",
                marker={"size": 9, "color": "#d87045"},
            )
        )
    figure.update_layout(
        height=360, margin={"l": 20, "r": 20, "t": 35, "b": 20},
        xaxis_title="距离 (km)", yaxis_title="海拔 (m)", hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def race_plan_display(plan: pd.DataFrame) -> pd.DataFrame:
    display = plan.copy()
    display["moving_time"] = display["moving_seconds"].apply(format_duration)
    display["avg_pace"] = display["avg_pace_min_km"].apply(pace_label)
    display["arrival"] = display["arrival_seconds"].apply(format_duration)
    display["departure"] = display["departure_seconds"].apply(format_duration)
    return display[[
        "segment", "distance_km", "gain_m", "loss_m", "moving_time",
        "avg_pace", "arrival", "stop_min", "departure",
    ]].rename(columns={
        "segment": "区间", "distance_km": "距离 km", "gain_m": "爬升 m",
        "loss_m": "下降 m", "moving_time": "移动时间", "avg_pace": "平均配速",
        "arrival": "到站时间", "stop_min": "停留 min", "departure": "出站时间",
    })


def nutrition_display(nutrition: pd.DataFrame) -> pd.DataFrame:
    display = nutrition.copy()
    rounded = [
        "duration_hours", "target_carbs_g", "planned_carbs_g", "target_fluid_ml",
        "drink_flask_equiv", "gel_servings", "solid_servings",
    ]
    display[rounded] = display[rounded].round(1)
    return display.rename(columns={
        "segment": "区间", "duration_hours": "时长 h", "target_carbs_g": "目标碳水 g",
        "planned_carbs_g": "计划碳水 g", "target_fluid_ml": "目标饮水 ml",
        "flasks_to_carry": "携带水壶数", "drink_flask_equiv": "饮用壶数",
        "gel_servings": "能量胶份数", "solid_servings": "固体食物份数",
    })


def pace_figure(plan: pd.DataFrame, chart_type: str) -> go.Figure:
    """Build a labelled segment-pace bar or smoothed line chart."""
    segments = plan["segment"].astype(str).tolist()
    pace_values = plan["avg_pace_min_km"].astype(float).tolist()
    pace_labels = [f"{value:.2f}" for value in pace_values]

    if chart_type == "折线图":
        positions = list(range(len(segments)))
        figure = go.Figure(
            go.Scatter(
                x=positions,
                y=pace_values,
                mode="lines+markers+text",
                line={"shape": "spline", "smoothing": 1.0, "width": 3, "color": "#3f7457"},
                marker={"size": 9, "color": "#d87045"},
                text=pace_labels,
                textposition="top center",
                hovertemplate="%{customdata}<br>预计配速 %{y:.2f} 分钟/公里<extra></extra>",
                customdata=segments,
                name="预计配速",
            )
        )
        figure.update_xaxes(tickmode="array", tickvals=positions, ticktext=segments)
    else:
        figure = go.Figure(
            go.Bar(
                x=segments,
                y=pace_values,
                text=pace_labels,
                textposition="outside",
                cliponaxis=False,
                marker={"color": pace_values, "colorscale": "Tealgrn", "showscale": False},
                hovertemplate="%{x}<br>预计配速 %{y:.2f} 分钟/公里<extra></extra>",
                name="预计配速",
            )
        )

    figure.update_layout(
        height=350,
        margin={"l": 20, "r": 20, "t": 20, "b": 25},
        xaxis_title="区间",
        yaxis_title="预计配速（分钟/公里）",
        showlegend=False,
    )
    lower_bound = max(0.0, min(pace_values) * 0.85) if pace_values else 0.0
    upper_bound = max(pace_values) * 1.15 if pace_values else 1.0
    figure.update_yaxes(tickformat=".2f", range=[lower_bound, upper_bound])
    return figure


def render_plan_results(
    course: pd.DataFrame,
    plan: pd.DataFrame,
    nutrition: pd.DataFrame,
    metadata: dict[str, float],
) -> None:
    st.markdown("#### 已生成的计划")
    summary = summarize_course(course)
    planned_score = load_race_score_model().estimate(
        distance_km=summary.distance_km,
        elevation_gain_m=summary.elevation_gain_m,
        finish_time=metadata["elapsed_seconds"] / 3600.0,
    )
    metric1, metric2, metric3, metric4, metric5 = st.columns(5)
    metric1.metric("预计完赛", format_duration(metadata["elapsed_seconds"]))
    metric2.metric("移动时间", format_duration(metadata["moving_seconds"]))
    metric3.metric("计划停留", format_duration(metadata["stop_seconds"]))
    metric4.metric("累计爬升", f"{summary.elevation_gain_m:,.0f} m")
    metric5.metric(
        "预估 Race Score",
        f"{planned_score.score:.1f}",
        help=(
            f"80% 估算区间 {planned_score.lower_80:.1f}–{planned_score.upper_80:.1f}。"
            "这是仅用赛道参数与预计完赛时间得到的非官方、低置信度估算。"
        ),
    )
    st.caption(
        f"Race Score 80% 估算区间：{planned_score.lower_80:.1f}–{planned_score.upper_80:.1f}。"
        "无需 ITRA 账号；当前为赛道参数模型，误差较大，仅供赛前参考。"
    )

    overview_tab, pace_tab, fuel_tab = st.tabs(["总览", "分段比赛计划", "补给与携带"])
    with overview_tab:
        st.plotly_chart(course_figure(course, plan), use_container_width=True)
        chart_heading, chart_switch = st.columns([3, 1], vertical_alignment="center")
        with chart_heading:
            st.markdown("#### 各段预计配速")
        with chart_switch:
            chart_type = st.segmented_control(
                "图表类型",
                ["柱状图", "折线图"],
                default="柱状图",
                key="pace_chart_type",
                label_visibility="collapsed",
            )
        st.plotly_chart(pace_figure(plan, chart_type), use_container_width=True)
        st.caption("配速是包含坡度影响的区间平均预算；陡坡路段应以体感强度和时间预算为主。")
    with pace_tab:
        st.dataframe(
            race_plan_display(plan), hide_index=True, use_container_width=True,
            column_config={
                "距离 km": st.column_config.NumberColumn(format="%.1f"),
                "爬升 m": st.column_config.NumberColumn(format="%.0f"),
                "下降 m": st.column_config.NumberColumn(format="%.0f"),
                "停留 min": st.column_config.NumberColumn(format="%.1f"),
            },
        )
    with fuel_tab:
        total1, total2, total3, total4 = st.columns(4)
        total1.metric("目标碳水", f"{nutrition['target_carbs_g'].sum():.0f} g")
        total2.metric("目标饮水", f"{nutrition['target_fluid_ml'].sum() / 1000:.1f} L")
        total3.metric("能量胶", f"{nutrition['gel_servings'].sum():.1f} 份")
        total4.metric("固体食物", f"{nutrition['solid_servings'].sum():.1f} 份")
        st.dataframe(nutrition_display(nutrition), hide_index=True, use_container_width=True)
        st.info(
            "水壶数表示从该段起点出发时需要的容量；到补给站后按下一段重新装载。"
            "半份食物可理解为半根能量棒或等量替代品。"
        )

    download1, download2 = st.columns(2)
    with download1:
        st.download_button(
            "下载分段计划 CSV", plan.to_csv(index=False).encode("utf-8-sig"),
            "race_plan.csv", "text/csv", use_container_width=True,
        )
    with download2:
        st.download_button(
            "下载补给计划 CSV", nutrition.to_csv(index=False).encode("utf-8-sig"),
            "nutrition_plan.csv", "text/csv", use_container_width=True,
        )


def render_planner_page() -> None:
    st.title("比赛计划与补给")
    st.caption("从 GPX 快速生成可执行的分段时间表和补给携带清单。")
    st.markdown(
        '<div class="step-row"><div class="step-pill">1 · 上传赛道</div>'
        '<div class="step-pill">2 · 设置目标与站点</div>'
        '<div class="step-pill">3 · 查看并下载计划</div></div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "上传 GPX 赛道文件", type=["gpx"], key="planner_gpx_upload",
        accept_multiple_files=False,
        help="可拖入文件，或点击 Browse files 选择。需要包含经纬度和海拔。",
    )
    if uploaded is None:
        st.markdown(
            '<div class="section-note"><b>从 GPX 开始</b><br>'
            '拖入 GPX 或点击 Browse files；载入后会立即显示距离、高程和累计爬升。</div>',
            unsafe_allow_html=True,
        )
        return

    payload = uploaded.getvalue()
    source_name = uploaded.name
    file_id = hashlib.sha1(payload).hexdigest()[:10]
    try:
        with st.spinner("正在读取 GPX..."):
            course = load_course(payload)
    except Exception as exc:
        st.error(f"GPX 解析失败：{exc}")
        return
    summary = summarize_course(course)
    st.success(f"已载入 {source_name}，共解析 {len(course):,} 个赛道采样点。")

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("赛道距离", f"{summary.distance_km:.1f} km")
    top2.metric("累计爬升", f"{summary.elevation_gain_m:,.0f} m")
    top3.metric("累计下降", f"{summary.elevation_loss_m:,.0f} m")
    top4.metric("最高海拔", f"{summary.max_elevation_m:,.0f} m")

    settings_col, preview_col = st.columns([0.9, 1.35], gap="large")
    result_key = f"planner_result_{file_id}"
    with settings_col:
        st.markdown("### 计划设置")
        st.markdown("#### 完赛目标")
        mode = st.radio(
            "时间来源", ["指定目标完赛时间", "参考成绩估算"], horizontal=True,
            key=f"target_mode_{file_id}",
        )
        with st.form(f"planner_form_{file_id}"):
            if mode == "指定目标完赛时间":
                target_text = st.text_input(
                    "目标完赛时间", "12:00:00", help="格式 HH:MM:SS，可超过 24 小时"
                )
                reference_distance = reference_text = None
            else:
                ref1, ref2 = st.columns(2)
                with ref1:
                    reference_distance = st.number_input(
                        "历史比赛距离 km", 1.0, 200.0, 42.195, 1.0
                    )
                with ref2:
                    reference_text = st.text_input(
                        "历史完赛时间", "04:00:00", help="格式 HH:MM:SS"
                    )
                target_text = None

            with st.expander("强度修正", expanded=False):
                fatigue_decay = st.slider(
                    "终点疲劳减速", 0, 40, 12, format="%d%%",
                    help="从起点 0% 线性增加到终点。首次使用建议保持 10–15%。",
                )
                altitude_effect = st.checkbox(
                    "计入 1000 m 以上海拔影响", True
                )

            st.markdown("#### 补给站")
            st.caption("已按赛道距离预填建议位置，请改成赛事真实站点；可直接增删行。")
            checkpoint_table = st.data_editor(
                default_checkpoints(summary.distance_km),
                num_rows="dynamic", hide_index=True, use_container_width=True,
                key=f"checkpoints_{file_id}",
                column_config={
                    "name": st.column_config.TextColumn("站点"),
                    "distance_km": st.column_config.NumberColumn(
                        "累计 km", min_value=0.0, max_value=summary.distance_km, format="%.1f"
                    ),
                    "stop_min": st.column_config.NumberColumn(
                        "停留 min", min_value=0.0, format="%.1f"
                    ),
                },
            )

            st.markdown("#### 每小时补给目标")
            fuel1, fuel2 = st.columns(2)
            with fuel1:
                carbs = st.number_input(
                    "碳水 g/h", 20, 100, 60, 5,
                    help="常见耐力目标 30–60 g/h；高碳策略需训练验证。",
                )
            with fuel2:
                fluid = st.number_input(
                    "饮水 ml/h", 200, 1200, 500, 50,
                    help="按个人汗率、温度和口渴调整。",
                )

            with st.expander("食物和水壶规格", expanded=False):
                spec1, spec2 = st.columns(2)
                with spec1:
                    flask_ml = st.number_input("水壶容量 ml", 200, 1000, 500, 50)
                    drink_carbs = st.number_input("每壶饮料碳水 g", 0, 100, 30, 5)
                with spec2:
                    gel_carbs = st.number_input("每份能量胶碳水 g", 5, 60, 25, 1)
                    solid_carbs = st.number_input("每份固体食物碳水 g", 5, 100, 40, 1)
                gel_share = st.slider("胶在非饮料碳水中的偏好", 0, 100, 70, 5, format="%d%%")

            submitted = st.form_submit_button(
                "生成 / 更新计划", type="primary", use_container_width=True
            )

        if submitted:
            try:
                checkpoints = checkpoint_table.to_dict("records")
                if mode == "指定目标完赛时间":
                    plan, metadata = build_race_plan(
                        course, checkpoints,
                        target_elapsed_seconds=parse_duration(target_text),
                        fatigue_decay_pct=fatigue_decay,
                        altitude_effect=altitude_effect,
                    )
                else:
                    plan, metadata = build_race_plan(
                        course, checkpoints,
                        reference_distance_km=float(reference_distance),
                        reference_time_seconds=parse_duration(reference_text),
                        fatigue_decay_pct=fatigue_decay,
                        altitude_effect=altitude_effect,
                    )
                nutrition = build_nutrition_plan(
                    plan, carbs_per_hour=carbs, fluid_ml_per_hour=fluid,
                    flask_ml=flask_ml, drink_carbs_per_flask=drink_carbs,
                    gel_carbs=gel_carbs, solid_carbs=solid_carbs,
                    gel_share=gel_share / 100.0,
                )
                st.session_state[result_key] = {
                    "plan": plan, "nutrition": nutrition, "metadata": metadata
                }
                st.success("计划已更新。")
            except Exception as exc:
                st.error(f"生成失败：{exc}")

    with preview_col:
        result = st.session_state.get(result_key)
        if result is None:
            st.markdown("### 赛道预览")
            st.plotly_chart(course_figure(course), use_container_width=True)
            st.info("赛道已解析。完成左侧设置并点击“生成 / 更新计划”。")
        else:
            render_plan_results(
                course, result["plan"], result["nutrition"], result["metadata"]
            )
            with st.expander("安全提示与模型限制"):
                st.warning(
                    "补给数值仅用于赛前规划。碳水摄入需在训练中逐步验证；饮水应按个人汗率、"
                    "天气与口渴调整，避免过量饮水。有代谢、肾脏、心血管或胃肠疾病时，请先咨询专业人士。"
                )
                st.caption(
                    "模型尚未计入路面技术难度、天气、夜间、拥堵、装备重量和个人下坡能力。"
                )


def main() -> None:
    configure_page()
    page = render_sidebar()
    if page == "比赛计划与补给":
        render_planner_page()
    else:
        render_results_page()
