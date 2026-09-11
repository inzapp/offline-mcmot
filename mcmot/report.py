from __future__ import annotations

import csv
import html
from collections import Counter
from pathlib import Path

from .export_html import export_standalone_html
from .report_visuals import generate_report_assets


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def table(data: list[dict], columns: list[str], limit=100) -> str:
    if not data:
        return '<div class="empty">데이터 없음</div>'
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in data[:limit]:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def image_gallery(items: list[dict], caption, css_class="") -> str:
    if not items:
        return '<div class="empty">표시할 실제 사례가 없습니다.</div>'
    return f'<div class="gallery {html.escape(css_class)}">' + ''.join(
        f'<figure><a href="{html.escape(item["src"])}"><img loading="lazy" src="{html.escape(item["src"])}"></a>'
        f'<figcaption>{caption(item)}</figcaption></figure>' for item in items) + '</div>'


def age_chart(global_people: list[dict], labels: list[str]) -> tuple[str, list[dict], int]:
    counts = Counter((row.get("age") or "unknown") for row in global_people)
    known_total = sum(counts[label] for label in labels if label != "unknown")
    bars = []
    for label in labels:
        count = counts[label]
        denominator = len(global_people) if label == "unknown" else known_total
        rate = count / denominator * 100 if denominator else 0
        qualifier = "전체 대비" if label == "unknown" else "판별 가능 인구 대비"
        bars.append(f'<div class="bar-row"><div class="bar-label"><b>{html.escape(label)}</b>'
                    f'<span>{count:,}명 · {rate:.1f}% ({qualifier})</span></div>'
                    f'<div class="bar-track"><i style="width:{rate:.2f}%"></i></div></div>')
    dates = sorted({row.get("date", "") for row in global_people})
    daily_age = []
    for date in dates:
        date_counts = Counter((row.get("age") or "unknown") for row in global_people if row.get("date") == date)
        daily_age.append({"날짜": date, **{f"{label}(명)": date_counts[label] for label in labels},
                          "판별 가능(명)": sum(date_counts[label] for label in labels if label != "unknown")})
    return '<div class="age-chart">' + ''.join(bars) + '</div>', daily_age, known_total


def generate_report(output_root: Path, cfg: dict) -> Path:
    manifest = rows(output_root / "manifest/recordings.csv")
    local = rows(output_root / "local/persons.csv")
    global_people = rows(output_root / "global/persons.csv")
    associations = rows(output_root / "global/association_events.csv")
    topology = rows(output_root / "global/topology.csv")
    attributes = rows(output_root / "attributes/inference_results.csv")
    issues = rows(output_root / "quality/issues.csv")
    accepted = [row for row in associations if row.get("accepted") == "1"]
    available_attributes = [row for row in attributes if row.get("status") == "ok"]
    bev_only = [row for row in local if "bev_only" in row.get("quality_flags", "")]
    invalid_videos = [row for row in manifest if "invalid_video" in row.get("status", "")]
    dates = sorted(set(cfg["dates"]))
    daily = []
    for date in dates:
        locals_d = [x for x in local if x.get("date") == date]
        globals_d = [x for x in global_people if x.get("date") == date]
        watched_d = sum(float(x.get("watch_seconds") or 0) > 0 for x in globals_d)
        attention_d = sum(float(x.get("attention_seconds") or 0) > 0 for x in globals_d)
        denominator = len(globals_d)
        daily.append({"날짜": date, "노출인구(local)": len(locals_d), "노출인구(global)": denominator,
                      "1초 이상 시청 인구": watched_d,
                      "시청률": f"{watched_d / denominator * 100:.2f}%" if denominator else "-",
                      "3초 이상 주목 인구": attention_d,
                      "주목률": f"{attention_d / denominator * 100:.2f}%" if denominator else "-",
                      "중복 제거": max(0, len(locals_d) - len(globals_d)),
                      "노출 시간(s)": f"{sum(float(x.get('exposure_seconds') or 0) for x in globals_d):.1f}",
                      "시청 시간(s)": f"{sum(float(x.get('watch_seconds') or 0) for x in globals_d):.1f}",
                      "주목 시간(s)": f"{sum(float(x.get('attention_seconds') or 0) for x in globals_d):.1f}"})
    status_counts = Counter(row.get("status", "unknown") for row in manifest)
    watched = sum(float(x.get("watch_seconds") or 0) > 0 for x in global_people)
    attentive = sum(float(x.get("attention_seconds") or 0) > 0 for x in global_people)
    global_count = len(global_people)
    age_labels = [label for label in cfg["attributes"]["age_labels"] if label != "unknown"] + ["unknown"]
    age_bars, daily_age, known_age_count = age_chart(global_people, age_labels)
    cards = [
        ("노출인구(local)", len(local), "카메라별 ID 합계"),
        ("노출인구(global)", global_count, "날짜별 카메라 중복 제거"),
        ("시청률", f"{watched / global_count * 100:.2f}%" if global_count else "-", f"1초 이상 {watched:,}명 / {global_count:,}명"),
        ("주목률", f"{attentive / global_count * 100:.2f}%" if global_count else "-", f"3초 이상 {attentive:,}명 / {global_count:,}명"),
        ("Accepted matches", len(accepted), "BEV·시간·ReID 통과"),
        ("Attribute coverage", len(available_attributes), f"{len(local)} IDs 중 실제 crop 보유"),
        ("BEV-only IDs", len(bev_only), "3421 raw 결손"),
        ("Invalid videos", len(invalid_videos), "메타데이터/디코딩 불가"),
        ("Data issues", len(issues), "임의 보완하지 않은 항목"),
    ]
    card_html = "".join(f'<div class="card"><small>{html.escape(k)}</small><b>{v if isinstance(v, str) else f"{v:,}"}</b><span>{html.escape(note)}</span></div>' for k, v, note in cards)
    camera_rows = []
    mappings = rows(output_root / "global/id_mapping.csv")
    visuals = generate_report_assets(output_root, cfg, topology, associations, local, mappings)
    explainer = visuals.get("global_id_explainer", {})
    for camera in sorted({row.get("camera_id", "") for row in local}):
        camera_local = [row for row in local if row.get("camera_id") == camera]
        camera_mapping = [row for row in mappings if row.get("camera_id") == camera]
        camera_attrs = [row for row in available_attributes if f":{camera}:" in row.get("local_uid", "")]
        camera_bev = [row for row in camera_local if "bev_only" in row.get("quality_flags", "")]
        camera_rows.append({"camera": camera, "local IDs": len(camera_local),
            "연결된 global IDs": len({row.get("global_id") for row in camera_mapping}),
            "속성 추론 가능": len(camera_attrs), "BEV-only": len(camera_bev),
            "노출 시간(s)": f"{sum(float(row.get('track_span_seconds') or 0) for row in camera_local):.1f}",
            "시청 시간(s)": f"{sum(float(row.get('watch_seconds') or 0) for row in camera_local):.1f}",
            "주목 시간(s)": f"{sum(float(row.get('attention_seconds') or 0) for row in camera_local):.1f}"})
    content = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Offline MCMOT 분석 보고서</title>
<style>
:root{{--ink:#132337;--muted:#607086;--line:#dce4eb;--bg:#f3f6f8;--blue:#087ea4;--green:#16845b;--red:#ba3d31}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Pretendard,"Noto Sans KR",sans-serif}}
header{{padding:48px max(5vw,28px);color:white;background:linear-gradient(125deg,#102940,#087e91)}}
header h1{{margin:0 0 10px;font-size:38px}}header p{{max-width:920px;color:#d7edf1;line-height:1.7}}
main{{max-width:1500px;margin:auto;padding:28px}}section{{background:white;border:1px solid var(--line);border-radius:16px;padding:24px;margin:20px 0;box-shadow:0 5px 18px #16324b0b}}
h2{{margin:0 0 15px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}}
.card{{border:1px solid var(--line);border-radius:13px;padding:18px;display:flex;flex-direction:column;gap:5px}}.card b{{font-size:30px;color:var(--blue)}}.card span,small{{color:var(--muted)}}
.table-wrap{{overflow:auto;max-height:560px;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:#eaf2f5;z-index:1}}tr:hover td{{background:#f6fbfc}}
.note{{border-left:4px solid var(--blue);padding:12px 16px;background:#edf8fb;line-height:1.65}}.warn{{border-color:var(--red);background:#fff3f0}}video{{width:100%;max-height:720px;background:#08131f;border-radius:12px}}pre{{padding:18px;background:#101b29;color:#cce5ec;overflow:auto;border-radius:10px}}.empty{{padding:30px;text-align:center;color:var(--muted);background:#f7f9fa;border-radius:10px}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}figure{{margin:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#f8fafb}}figure img{{display:block;width:100%;height:300px;object-fit:contain;background:#12202d}}figcaption{{padding:13px 15px;line-height:1.55;color:var(--muted)}}.wide{{width:100%;max-height:760px;object-fit:contain;background:#12202d;border-radius:12px}}.steps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;counter-reset:step}}.step{{padding:16px;border:1px solid var(--line);border-radius:12px;line-height:1.55}}.step:before{{counter-increment:step;content:counter(step);display:inline-grid;place-items:center;width:28px;height:28px;margin-right:8px;border-radius:50%;background:var(--blue);color:white;font-weight:bold}}
.age-chart{{display:grid;gap:15px;margin:22px 0}}.bar-row{{display:grid;grid-template-columns:minmax(210px,26%) 1fr;gap:16px;align-items:center}}.bar-label{{display:flex;justify-content:space-between;gap:10px}}.bar-label span{{color:var(--muted);font-size:13px}}.bar-track{{height:25px;background:#e8eef2;border-radius:7px;overflow:hidden}}.bar-track i{{display:block;height:100%;min-width:2px;background:linear-gradient(90deg,#087ea4,#20a77c);border-radius:7px}}@media(max-width:700px){{.bar-row{{grid-template-columns:1fr}}}}
.decision-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}}.decision-grid>div{{display:flex;flex-direction:column;gap:7px;padding:17px;border:1px solid var(--line);border-radius:12px;background:#f8fafb}}.decision-grid b{{color:var(--blue);font-size:17px}}.decision-grid span{{color:var(--muted);line-height:1.55}}.decision-grid .pass{{background:#eaf8f1;border-color:#91d6b7}}.decision-grid .pass b{{color:var(--green)}}.gallery.reid-grid{{grid-template-columns:repeat(4,minmax(0,1fr))}}.gallery.reid-grid figure img{{height:245px}}@media(max-width:1050px){{.gallery.reid-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:600px){{.gallery.reid-grid{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Offline MCMOT 분석 보고서</h1><p>{html.escape(cfg['site'])} · {', '.join(dates)}. Local MOT, BEV topology, 이동 가능성, ReID embedding을 함께 사용한 결과입니다. 누락된 입력에서 값은 생성하지 않았습니다.</p></header><main>
<div class="cards">{card_html}</div>
<section><h2>일별 핵심 지표</h2><p class="note">노출인구(local)는 각 카메라에서 발견한 ID의 합계이며, 노출인구(global)는 같은 사람으로 승인된 ID를 묶은 결과입니다. 시청률과 주목률의 분모는 노출인구(global)입니다.</p>{table(daily, list(daily[0]) if daily else [])}</section>
<section><h2>연령대 분포</h2><p class="note">Global ID마다 최종 선정된 대표 crop 1장을 연령 모델로 추론한 결과입니다. 판별 가능 인구는 <b>{known_age_count:,}명</b>으로 전체 노출인구(global)의 <b>{known_age_count / global_count * 100 if global_count else 0:.1f}%</b>입니다. 연령대 비율은 판별 가능한 인구를 분모로 계산하고, <code>unknown</code>만 전체 인구 대비로 표시합니다.</p>{age_bars}<h3>날짜별 연령대 인구</h3>{table(daily_age, list(daily_age[0]) if daily_age else [])}</section>
<section><h2>Global ID는 어떻게 만들어지는가</h2><p class="note">아래 흐름도는 설명용 가상 예시가 아니라, 실제 승인된 카메라 간 매칭 1건의 crop·BEV 좌표·판정값을 사용합니다. 왼쪽의 서로 다른 두 Local ID가 모든 검사를 통과하면 오른쪽의 하나의 Global ID가 됩니다.</p><div class="steps"><div class="step"><b>카메라 안에서 추적</b><br>ROI 안의 사람만 추적하고 짧은 검출 누락은 Kalman 예측으로 보완합니다.</div><div class="step"><b>공간·시간 후보 선별</b><br>BEV 위치가 겹치거나, ROI 경계에서 사라진 뒤 이동 가능한 시간 안에 나타난 경우만 비교합니다.</div><div class="step"><b>외형 비교(ReID)</b><br>두 대표 crop의 embedding L2 거리가 기준보다 가까운지 확인합니다.</div><div class="step"><b>Global ID 병합</b><br>공간·시간 gate와 ReID 기준을 모두 통과한 연결만 같은 사람으로 묶습니다.</div></div>{f'<a href="{explainer["src"]}"><img class="wide" loading="lazy" src="{explainer["src"]}"></a>' if explainer else '<div class="empty">승인 사례가 없어 흐름도를 만들 수 없습니다.</div>'}<h3>일반적인 판정 기준</h3><div class="decision-grid"><div><b>① 비교 대상인가?</b><span>서로 다른 카메라이고 같은 날짜·30분 세션이어야 합니다.</span></div><div><b>② 이동 가능한가?</b><span>동시 관측은 BEV 거리 0.055 이하, handoff는 최대 8초와 최대 속도 조건을 확인합니다.</span></div><div><b>③ 화면 경계인가?</b><span>handoff라면 출발 소실점과 도착 진입점이 각각 카메라 하위 15% 경계 범위여야 합니다.</span></div><div><b>④ 외형이 같은가?</b><span>ReID L2 거리가 {float(cfg['reid']['distance_threshold']):.4f}보다 작아야 합니다.</span></div><div><b>⑤ 모순이 없는가?</b><span>병합 결과에 같은 카메라·같은 시간의 두 사람이 생기면 거부합니다.</span></div><div class="pass"><b>✓ Global ID 병합</b><span>앞선 조건을 모두 통과한 연결을 낮은 종합점수 순으로 하나의 ID로 묶습니다.</span></div></div><p class="note warn"><b>Topology 해석:</b> 현재 topology는 승인된 handoff를 방향별로 집계한 결과입니다. 같은 방향이 최소 {cfg['global_matching']['min_topology_observations']}회 나타나면 confirmed로 표시되며, 사전에 정해진 카메라 연결표로 후보를 제한하는 방식은 아닙니다.</p></section>
<section><h2>실제 이동 밀도 히트맵</h2><p>실제 관측된 421만여 track row의 BEV 좌표를 누적한 결과입니다. 따뜻한 색일수록 사람이 자주 관측된 구간입니다.</p>{f'<a href="{visuals["heatmap"]}"><img class="wide" loading="lazy" src="{visuals["heatmap"]}"></a>' if visuals['heatmap'] else '<div class="empty">히트맵 없음</div>'}</section>
<section><h2>실제 사용자 이동 경로 사례</h2><p class="note">여러 local ID가 하나의 global ID로 병합된 사례를 우선 선택했습니다. 초록점은 경로 시작, 빨간점은 종료이며 선은 실제 관측 위치입니다.</p>{image_gallery(visuals['trajectories'], lambda x: f"{html.escape(x['global_id'])} · local ID {x['local_ids']}개 · 카메라 {html.escape(x['cameras'])}")}</section>
<section><h2>카메라별 Tracking ROI</h2><p>아래 사진은 실제 카메라 캡처입니다. 색칠된 다각형 안의 검출만 추적합니다. 흰색/색상 경계 띠는 사람이 화면에서 사라지거나 진입하는 후보 영역입니다.</p>{image_gallery(visuals['camera_rois'], lambda x: f"카메라 {html.escape(x['camera'])} · ROI 꼭짓점 {x['vertices']}개")}</section>
<section><h2>카메라별 데이터 기반 진입·소실 위치</h2><p class="note">각 local track의 시작·끝 BEV 경계거리를 카메라별로 모은 뒤, 전체 거리 분포의 하위 <b>{float(cfg['global_matching']['boundary_quantile']) * 100:.0f}%</b>를 경계 후보로 선택합니다. 아래 이미지는 그 track의 실제 카메라 좌표를 캡처 위에 다시 표시한 것입니다. 3421처럼 원본 카메라 좌표가 없는 BEV-only 데이터는 임의로 점을 만들지 않습니다.</p><div class="steps"><div class="step"><b>거리 계산</b><br>각 track의 시작점·끝점에서 BEV ROI 다각형 경계까지 최단거리를 구합니다.</div><div class="step"><b>카메라별 threshold</b><br>카메라별 시작·끝 거리 전체에서 하위 15% quantile을 계산합니다.</div><div class="step"><b>후보 시각화</b><br>파란 원은 진입점, 빨간 X는 소실점입니다. 색상 heat는 모든 후보의 밀도입니다.</div><div class="step"><b>Handoff gate</b><br>출발 카메라의 소실점과 도착 카메라의 진입점이 모두 기준을 통과해야 합니다.</div></div>{image_gallery(visuals['boundary_positions'], lambda x: f"카메라 {html.escape(x['camera'])} · BEV 정규화 거리 threshold {x['threshold']} · 진입 {x['entries']:,}건 · 소실 {x['exits']:,}건 · {html.escape(x['note'])}")}</section>
<section><h2>데이터 기반 카메라 Topology</h2><p>경계에서 종료된 track과 다른 카메라에서 시작한 track 중 BEV 속도로 이동 가능한 후보를 모으고, 최소 {cfg['global_matching']['min_topology_observations']}회 관측된 방향만 topology로 확정합니다. 화살표 숫자는 실제 승인된 전환 관측 수입니다.</p>{f'<a href="{visuals["topology"]}"><img class="wide" loading="lazy" src="{visuals["topology"]}"></a>' if visuals['topology'] else '<div class="empty">Topology 없음</div>'}{table(topology, ['from_camera','to_camera','observed_transition_count','status'], 1000)}</section>
<section><h2>실제 ReID 승인 사례</h2><p class="note">ReID는 공간·시간 조건을 먼저 통과한 후보에만 적용됩니다. 이 보고서의 L2 승인 threshold는 <b>{float(cfg['reid']['distance_threshold']):.4f}</b>이며, 거리가 작을수록 외형 embedding이 가깝습니다. 아래는 실제 서로 다른 카메라 crop이 승인된 사례 12건입니다.</p>{image_gallery(visuals['reid_matches'], lambda x: f"{html.escape(x['global_id'])} · CAM {x['left_camera']} → {x['right_camera']} · L2 {x['distance']} &lt; {x['threshold']} · BEV 거리 {x['bev_distance']} · 시간차 {x['time_gap']}초 · {html.escape(x['match_type'])}", 'reid-grid')}</section>
<section><h2>카메라별 처리 현황</h2>{table(camera_rows, list(camera_rows[0]) if camera_rows else [], 100)}</section>
</main></body></html>"""
    target = output_root / "report/index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    export_standalone_html(target, target.with_name("index_export.html"))
    return target
