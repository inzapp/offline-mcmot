from __future__ import annotations

import csv
import html
from collections import Counter
from pathlib import Path


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
        daily.append({"date": date, "local IDs": len(locals_d), "global IDs": len(globals_d),
                      "중복 제거": max(0, len(locals_d) - len(globals_d)),
                      "노출 시간(s)": f"{sum(float(x.get('exposure_seconds') or 0) for x in globals_d):.1f}",
                      "시청 시간(s)": f"{sum(float(x.get('watch_seconds') or 0) for x in globals_d):.1f}",
                      "주목 시간(s)": f"{sum(float(x.get('attention_seconds') or 0) for x in globals_d):.1f}"})
    status_counts = Counter(row.get("status", "unknown") for row in manifest)
    cards = [
        ("Local IDs", len(local), "카메라별 독립 추적"),
        ("Global IDs", len(global_people), "날짜별 중복 제거"),
        ("Accepted matches", len(accepted), "BEV·시간·ReID 통과"),
        ("Attribute coverage", len(available_attributes), f"{len(local)} IDs 중 실제 crop 보유"),
        ("BEV-only IDs", len(bev_only), "3421 raw 결손"),
        ("Invalid videos", len(invalid_videos), "메타데이터/디코딩 불가"),
        ("Data issues", len(issues), "임의 보완하지 않은 항목"),
    ]
    card_html = "".join(f'<div class="card"><small>{html.escape(k)}</small><b>{v:,}</b><span>{html.escape(note)}</span></div>' for k, v, note in cards)
    config_text = html.escape(Path(cfg["config_path"]).read_text(encoding="utf-8"))
    camera_rows = []
    mappings = rows(output_root / "global/id_mapping.csv")
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
    video = output_root / "visualization/best_matches.mp4"
    video_html = ('<video controls preload="metadata" src="../visualization/best_matches.mp4"></video>'
                  if video.exists() else '<div class="empty">매칭 사례가 생성되면 영상이 표시됩니다.</div>')
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
</style></head><body><header><h1>Offline MCMOT 분석 보고서</h1><p>{html.escape(cfg['site'])} · {', '.join(dates)}. Local MOT, BEV topology, 이동 가능성, ReID embedding을 함께 사용한 결과입니다. 누락된 입력에서 값은 생성하지 않았습니다.</p></header><main>
<div class="cards">{card_html}</div>
<section><h2>일별 핵심 지표</h2>{table(daily, list(daily[0]) if daily else [])}</section>
<section><h2>카메라별 처리 현황</h2>{table(camera_rows, list(camera_rows[0]) if camera_rows else [], 100)}</section>
<section><h2>우수 매칭 사례 영상</h2><p class="note">association score가 낮고 ReID 거리가 충분히 가까운 승인 사례를 우선 선택합니다. 좌우 crop, embedding 거리, threshold, BEV 연결을 함께 표시합니다.</p>{video_html}</section>
<section><h2>Global ID 상세</h2>{table(global_people, ['global_id','date','first_seen','last_seen','exposure_seconds','observed_seconds','predicted_gap_seconds','watch_seconds','attention_seconds','camera_ids','gender','age','quality_flags'], 1000)}</section>
<section><h2>카메라 topology</h2>{table(topology, ['from_camera','to_camera','observed_transition_count','status'], 1000)}</section>
<section><h2>Cross-camera association 감사 로그</h2><p class="note">물리 조건을 통과한 후보의 승인, ReID 거절, embedding 부재를 모두 보존하여 Global ID가 만들어진 근거를 재검토할 수 있습니다.</p>{table(sorted(associations,key=lambda x:float(x.get('association_score') or 999)), ['accepted','gate_status','match_type','left_uid','right_uid','reid_distance','reid_threshold','bev_distance','temporal_gap_seconds','association_score'], 1000)}</section>
<section><h2>속성 추론</h2>{table(attributes, ['local_uid','gender','gender_score','age','age_score','crop_path','status'], 1000)}</section>
<section><h2>입력 데이터 품질</h2><p>Manifest 상태: {html.escape(str(dict(status_counts)))}</p>{table(manifest, ['date','camera_id','nominal_start','status','raw_rows','bev_rows','video_fps','video_frames','video_duration'], 1000)}<h3>처리 이슈</h3>{table(issues, ['scope','severity','code','detail'], 1000)}</section>
<section><h2>산출 기준과 재현 설정</h2><p class="note warn">노출 시간은 짧은 Kalman 예측 구간을 포함한 ID 유지 시간입니다. 시청·주목 및 crop은 실제 관측 프레임에서만 계산합니다. 10초 녹화 term은 연결하지 않습니다.</p><pre>{config_text}</pre></section>
</main></body></html>"""
    target = output_root / "report/index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target
