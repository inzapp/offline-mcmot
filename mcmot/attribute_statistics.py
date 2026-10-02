from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _table(data: list[dict[str, str]], columns: list[str], limit: int = 1000) -> str:
    if not data:
        return '<div class="empty">데이터 없음</div>'
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in data[:limit]:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return (f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _integer(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _percent(value: str | None) -> str:
    number = _float(value)
    return f"{number:.4f}%" if number is not None else "-"


def _stat_map(summary: list[dict[str, str]]) -> dict[str, str]:
    return {row.get("metric", ""): row.get("value", "") for row in summary}


def _stat_value(values: dict[str, str], key: str, suffix: str = "") -> str:
    value = values.get(key, "")
    return f"{value}{suffix}" if value else "-"


def _distribution_rows(data: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"구분": row.get("category", ""), "인원": f"{_integer(row.get('count')):,}",
             "비율(%)": _percent(row.get("percent")),
             "분모": f"{_integer(row.get('denominator')):,}"} for row in data]


def _age_detail_rows(age_distribution: list[dict[str, str]],
                     age_gender_distribution: list[dict[str, str]]) -> list[dict[str, str]]:
    by_age_gender: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in age_gender_distribution:
        by_age_gender[row.get("age_group", "")][row.get("gender", "")] += _integer(row.get("count"))

    result = []
    for row in age_distribution:
        age_group = row.get("category", "")
        total = _integer(row.get("count"))
        female = by_age_gender[age_group].get("female", 0)
        male = by_age_gender[age_group].get("male", 0)
        gender_total = female + male
        result.append({
            "연령 구간": age_group,
            "인원": f"{total:,}",
            "전체 비율(%)": _percent(row.get("percent")),
            "여성(명)": f"{female:,}",
            "여성/전체 유효연령(%)": _percent(next(
                (item.get("percent_of_valid_age") for item in age_gender_distribution
                 if item.get("age_group") == age_group and item.get("gender") == "female"), None)),
            "여성/구간 내(%)": f"{female / total * 100:.2f}%" if total else "-",
            "남성(명)": f"{male:,}",
            "남성/전체 유효연령(%)": _percent(next(
                (item.get("percent_of_valid_age") for item in age_gender_distribution
                 if item.get("age_group") == age_group and item.get("gender") == "male"), None)),
            "남성/구간 내(%)": f"{male / total * 100:.2f}%" if total else "-",
            "성별 합계 검증": "일치" if gender_total == total else f"불일치 ({gender_total:,})",
        })
    return result


def _age_chart(age_distribution: list[dict[str, str]]) -> str:
    bars = []
    for row in age_distribution:
        label = row.get("category", "")
        count = _integer(row.get("count"))
        rate = _float(row.get("percent")) or 0
        width = min(100, max(0, rate))
        bars.append(
            f'<div class="bar-row"><div class="bar-label"><b>{html.escape(label)}</b>'
            f'<span>{count:,}명 · {_percent(row.get("percent"))} (유효 연령 스냅샷 대비)</span></div>'
            f'<div class="bar-track"><i style="width:{width:.4f}%"></i></div></div>')
    return '<div class="age-chart">' + "".join(bars) + "</div>"


def _summary_rows(summary: list[dict[str, str]]) -> list[dict[str, str]]:
    labels = {
        "total_snapshots": "전체 스냅샷",
        "successful_snapshots": "성공 스냅샷",
        "failed_snapshots": "실패 스냅샷",
        "age_count": "연령값 개수",
        "age_mean": "추정 연령 평균",
        "age_median": "추정 연령 중앙값",
        "age_min": "추정 연령 최솟값",
        "age_max": "추정 연령 최댓값",
        "age_std_population": "추정 연령 모집단 표준편차",
        "gender_score_count": "성별 score 개수",
        "gender_score_mean": "성별 score 평균",
        "gender_score_median": "성별 score 중앙값",
        "gender_score_min": "성별 score 최솟값",
        "gender_score_max": "성별 score 최댓값",
        "gender_score_std_population": "성별 score 모집단 표준편차",
        "detection_ms_count": "얼굴 검출 시간 측정 개수",
        "detection_ms_mean": "얼굴 검출 시간 평균",
        "detection_ms_median": "얼굴 검출 시간 중앙값",
        "detection_ms_min": "얼굴 검출 시간 최솟값",
        "detection_ms_max": "얼굴 검출 시간 최댓값",
        "detection_ms_std_population": "얼굴 검출 시간 모집단 표준편차",
        "detection_ms_p95_nearest_rank": "얼굴 검출 시간 P95",
        "preprocess_ms_count": "전처리 시간 측정 개수",
        "preprocess_ms_mean": "전처리 시간 평균",
        "preprocess_ms_median": "전처리 시간 중앙값",
        "preprocess_ms_min": "전처리 시간 최솟값",
        "preprocess_ms_max": "전처리 시간 최댓값",
        "preprocess_ms_std_population": "전처리 시간 모집단 표준편차",
        "preprocess_ms_p95_nearest_rank": "전처리 시간 P95",
        "inference_ms_count": "추론 시간 측정 개수",
        "inference_ms_mean": "추론 시간 평균",
        "inference_ms_median": "추론 시간 중앙값",
        "inference_ms_min": "추론 시간 최솟값",
        "inference_ms_max": "추론 시간 최댓값",
        "inference_ms_std_population": "추론 시간 모집단 표준편차",
        "inference_ms_p95_nearest_rank": "추론 시간 P95",
        "pipeline_ms_count": "파이프라인 시간 측정 개수",
        "pipeline_ms_mean": "파이프라인 시간 평균",
        "pipeline_ms_median": "파이프라인 시간 중앙값",
        "pipeline_ms_min": "파이프라인 시간 최솟값",
        "pipeline_ms_max": "파이프라인 시간 최댓값",
        "pipeline_ms_std_population": "파이프라인 시간 모집단 표준편차",
        "pipeline_ms_p95_nearest_rank": "파이프라인 시간 P95",
    }
    result = []
    for row in summary:
        key = row.get("metric", "")
        value = row.get("value", "")
        if key.endswith("_count") or key.endswith("_snapshots"):
            display = f"{_integer(value):,}"
        else:
            display = value or "-"
        if key.endswith("_ms_mean") or key.endswith("_ms_median") or key.endswith("_ms_min") \
                or key.endswith("_ms_max") or key.endswith("_ms_std_population") \
                or key.endswith("_ms_p95_nearest_rank"):
            unit = "ms"
        elif key.startswith("age_"):
            unit = "세" if key != "age_count" else "개"
        elif key.startswith("gender_score_"):
            unit = "score" if key != "gender_score_count" else "개"
        else:
            unit = "개"
        result.append({"지표": labels.get(key, key), "원본 키": key,
                       "값": display, "단위": unit})
    return result


def _raw_csv_blocks(statistics_root: Path) -> str:
    names = ["summary.csv", "age_distribution.csv", "age_gender_distribution.csv",
             "gender_distribution.csv", "input_mode_distribution.csv",
             "face_status_distribution.csv", "errors.csv"]
    blocks = []
    for name in names:
        path = statistics_root / name
        if not path.exists():
            continue
        raw = html.escape(path.read_text(encoding="utf-8-sig"))
        blocks.append(f'<details><summary>{html.escape(name)} 원본 CSV</summary><pre>{raw}</pre></details>')
    return "".join(blocks)


def build_attribute_statistics_section(statistics_root: Path, integrated: bool = False) -> str:
    """Render the aggregate attribute statistics without mixing units with global IDs."""
    summary = _rows(statistics_root / "summary.csv")
    age_distribution = _rows(statistics_root / "age_distribution.csv")
    age_gender_distribution = _rows(statistics_root / "age_gender_distribution.csv")
    gender_distribution = _rows(statistics_root / "gender_distribution.csv")
    input_mode_distribution = _rows(statistics_root / "input_mode_distribution.csv")
    face_status_distribution = _rows(statistics_root / "face_status_distribution.csv")
    errors = _rows(statistics_root / "errors.csv")
    values = _stat_map(summary)

    total = _integer(values.get("total_snapshots"))
    age_count = _integer(values.get("age_count"))
    gender_counts = {row.get("category", ""): _integer(row.get("count"))
                     for row in gender_distribution}
    age_detail = _age_detail_rows(age_distribution, age_gender_distribution)
    age_gender_table = [{
        "연령 구간": row.get("age_group", ""),
        "성별": row.get("gender", ""),
        "인원": f"{_integer(row.get('count')):,}",
        "유효 연령 전체 대비(%)": _percent(row.get("percent_of_valid_age")),
    } for row in age_gender_distribution]
    stat_cards = [
        ("새 모델 스냅샷", f"{total:,}", "summary.csv total_snapshots"),
        ("추정 연령 평균", _stat_value(values, "age_mean", "세"), "연속형 age 추정값"),
        ("추정 연령 중앙값", _stat_value(values, "age_median", "세"), "연속형 age 추정값"),
        ("추정 연령 범위", f"{_stat_value(values, 'age_min')}–{_stat_value(values, 'age_max')}세",
         "최솟값–최댓값"),
        ("남성", f"{gender_counts.get('male', 0):,}명", "gender_distribution.csv"),
        ("여성", f"{gender_counts.get('female', 0):,}명", "gender_distribution.csv"),
        ("성공률", f"{_integer(values.get('successful_snapshots')) / total * 100:.2f}%" if total else "-",
         "successful / total"),
        ("연령 표준편차", _stat_value(values, "age_std_population", "세"), "모집단 기준"),
    ]
    cards = "".join(
        f'<div class="card"><small>{html.escape(label)}</small><b>{html.escape(value)}</b>'
        f'<span>{html.escape(note)}</span></div>' for label, value, note in stat_cards)

    denominator_note = (f"연령 구간의 분모는 age_distribution.csv의 denominator인 {age_count:,}건이며, "
                       "percent는 원본 CSV 값을 표시합니다. 성별별 ‘구간 내 비율’만 인원으로부터 추가 계산했습니다.")
    provenance_note = (
        "이 섹션은 기존 global/persons.csv의 대표 crop별 집계가 아니라 "
        "attributes/statistics에 저장된 새 모델의 스냅샷 단위 집계입니다. 따라서 기존 보고서의 "
        "Global ID 기준 연령 수치와 직접 합산하거나 비교하지 않아야 합니다."
    )
    if integrated:
        provenance_note = ("이 섹션은 MiVOLO의 crop별 추론 처리 통계입니다. 같은 Global ID에 연결된 여러 crop이 "
                           "각각 집계될 수 있습니다. 방문객·동선·시선 집계는 Global ID의 대표 crop 결과를 사용하므로 "
                           "두 집계의 건수를 합산하지 않습니다.")
    return f'''<section id="attribute-statistics"><h2>새 모델 성별·연령 통계</h2>
<p class="note">{provenance_note} 총 {total:,}개 스냅샷 중 성공 {html.escape(values.get("successful_snapshots", "-"))}건, 실패 {html.escape(values.get("failed_snapshots", "-"))}건입니다.</p>
<div class="cards">{cards}</div>
<h3>연령 분포 — 세부 구간</h3><p class="note">{denominator_note}</p>
{_age_chart(age_distribution)}
{_table(age_detail, ["연령 구간", "인원", "전체 비율(%)", "여성(명)", "여성/전체 유효연령(%)", "여성/구간 내(%)", "남성(명)", "남성/전체 유효연령(%)", "남성/구간 내(%)", "성별 합계 검증"])}
<h3>연령×성별 원자료 집계</h3>
<p class="note">‘유효 연령 전체 대비’는 age_gender_distribution.csv의 percent_of_valid_age를 그대로 표시합니다. 0건인 invalid_negative 행도 누락하지 않았습니다.</p>
{_table(age_gender_table, ["연령 구간", "성별", "인원", "유효 연령 전체 대비(%)"])}
<h3>성별 분포</h3>
{_table(_distribution_rows(gender_distribution), ["구분", "인원", "비율(%)", "분모"])}
<h3>얼굴 검출·입력 모드별 분포</h3>
<div class="stat-pair"><div><h4>입력 모드</h4>{_table(_distribution_rows(input_mode_distribution), ["구분", "인원", "비율(%)", "분모"])}</div>
<div><h4>얼굴 상태</h4>{_table(_distribution_rows(face_status_distribution), ["구분", "인원", "비율(%)", "분모"])}</div></div>
<h3>새 모델 처리·성능 요약</h3>
{_table(_summary_rows(summary), ["지표", "원본 키", "값", "단위"], 1000)}
<h3>오류</h3>{_table(errors, list(errors[0]) if errors else ["file", "error"])}
<h3>원본 통계 CSV</h3><p class="note">아래를 펼치면 보고서에 반영한 CSV 원문을 확인할 수 있습니다.</p>{_raw_csv_blocks(statistics_root)}
</section>'''
