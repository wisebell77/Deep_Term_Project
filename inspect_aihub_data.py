#!/usr/bin/env python3
"""AI Hub '배터리 열폭주 제어 멀티모달 데이터'(dataSetSn=71918) 구조 점검 스크립트.

사용법: python inspect_aihub_data.py <zip 파일들이 있는 최상위 폴더>
전제:   내려받은 zip 파일을 풀지 않은 상태로 둔다. 하위 폴더는 자동으로 탐색한다.
        표준 라이브러리만 사용한다(Python 3.8 이상). 데이터는 읽기만 한다.
출력:   현재 폴더에 inspect_report.txt 와 tests_summary.csv 를 만든다.
"""
import csv
import json
import re
import struct
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ZIP_RE = re.compile(
    r"^(TS|TL|VS|VL)_(센서|이미지)_(.+)_([\d.]+)_(50|100)_(가열|과충전)_(\d{8})_(\d{11})$"
)
ABN = ["abn_temprise", "abn_presrise", "abn_vdrop", "abn_v0",
       "abn_vdddrop", "abn_temprise2", "abn_maxtemp"]
REPORT = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    REPORT.append(line)


def entry_name(info):
    """zip 내부 경로의 한글이 깨져 있으면 복원한다."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:
        return info.filename
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return info.filename


def json_records(zip_path, prefix):
    """zip 안에서 파일명이 prefix로 시작하는 JSON을 (파일명, dict)로 돌려준다."""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            base = entry_name(info).rsplit("/", 1)[-1]
            if not (base.startswith(prefix) and base.endswith(".json")):
                continue
            obj = json.loads(zf.read(info).decode("utf-8-sig"))
            for rec in obj if isinstance(obj, list) else [obj]:
                yield base, rec


def time_of_day(s):
    """'105630000'(시분초밀리초) -> 초"""
    s = str(s)
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) + int(s[6:9] or 0) / 1000


def hms(s):
    """'00:14:51' 또는 '2:21:21 PM' -> 초. 해석할 수 없으면 None"""
    m = re.match(r"^\s*(\d{1,2}):(\d{1,2}):(\d{1,2})\s*([AP]M)?\s*$", str(s), re.I)
    if not m:
        return None
    h = int(m.group(1))
    if m.group(4) and m.group(4).upper() == "PM" and h < 12:
        h += 12
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def png_shape(zf, info):
    head = zf.open(info).read(26)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return "PNG 아님"
    w, h = struct.unpack(">II", head[16:24])
    return "%dx%d depth%d color%d" % (w, h, head[24], head[25])


def main(root):
    # ---------- 1. zip 목록 ----------
    tests = defaultdict(dict)   # 시험 ID -> {"TL_센서": 경로, ...}
    cond = {}                   # 시험 ID -> (형태, 용량, SOC, 방식)
    others = []
    for z in sorted(root.rglob("*.zip")):
        m = ZIP_RE.match(unicodedata.normalize("NFC", z.stem))
        if m:
            tests[m.group(8)][m.group(1) + "_" + m.group(2)] = z
            cond[m.group(8)] = m.group(3, 4, 5, 6)
        else:
            others.append(z)
    kinds = ["TS_센서", "TS_이미지", "TL_센서", "VS_센서", "VS_이미지", "VL_센서"]
    say("== 1. zip 목록 ==")
    say("시험 수:", len(tests), "/ 규칙에 맞지 않는 zip:", [z.name for z in others])
    for k in kinds:
        say("  %s: %d개" % (k, sum(k in v for v in tests.values())))
    say("조건별 시험 수:", dict(sorted(Counter("_".join(c) for c in cond.values()).items())))

    # ---------- 2. 메타데이터 ----------
    meta = {}
    for z in others:
        for base, rec in json_records(z, "meta_data"):
            m = re.search(r"\d{11}", base)
            meta[m.group(0) if m else base] = rec
    say("\n== 2. 메타데이터 ==")
    say("메타데이터 수:", len(meta))
    say("메타데이터만 있는 시험:", sorted(set(meta) - set(tests)))
    say("메타데이터가 없는 시험:", sorted(set(tests) - set(meta)))
    if meta:
        first = next(iter(meta.values()))
        say("필드:", sorted(first))
        say("벤팅 시각 예:", [m.get("bat_venting_time") for m in list(meta.values())[:8]])
        say("열폭주 시각 예:", [m.get("bat_abn_temprise") for m in list(meta.values())[:8]])

    # ---------- 3. 라벨링 JSON (시험별) ----------
    say("\n== 3. 라벨링 JSON ==")
    key_sets, nulls, dts, stage_total = Counter(), Counter(), Counter(), Counter()
    abn_y = defaultdict(list)   # 필드 -> 시험별 "Y" 레코드 수
    rows, example_paths = [], []
    n_train = n_val = n_bad_order = n_not_mono = 0
    vent_elapsed_ok = vent_clock_ok = 0
    for tid in sorted(tests):
        recs = []
        for split, kind in (("T", "TL_센서"), ("V", "VL_센서")):
            z = tests[tid].get(kind)
            if not z:
                continue
            if len(example_paths) < 2:
                with zipfile.ZipFile(z) as zf:
                    example_paths.append(entry_name(zf.infolist()[-1]))
            for _, r in json_records(z, "labeling_data"):
                key_sets[tuple(sorted(r))] += 1
                for k, v in r.items():
                    if v is None or v == "" or v != v:
                        nulls[k] += 1
                recs.append((time_of_day(r["test_time"]), str(r.get("data_id")), split, r))
        if not recs:
            continue
        recs.sort(key=lambda x: x[:2])
        n_bad_order += [r[1] for r in recs] != sorted(r[1] for r in recs)
        times = [r[0] for r in recs]
        for a, b in zip(times, times[1:]):
            dts[round(b - a, 1)] += 1
        stages = [r[3].get("tr_stage") for r in recs]
        stage_total.update(stages)
        mono = all(a <= b for a, b in zip(stages, stages[1:]))
        n_not_mono += not mono
        first_stage = {s: round(times[stages.index(s)] - times[0], 1)
                       for s in range(2, 7) if s in stages}
        first_abn = {}
        for f in ABN:
            ys = [i for i, r in enumerate(recs) if r[3].get(f) == "Y"]
            abn_y[f].append(len(ys))
            if ys:
                first_abn[f] = round(times[ys[0]] - times[0], 1)
        splits = "".join(r[2] for r in recs)
        v_runs = len(re.findall(r"V+", splits))
        nt, nv = splits.count("T"), splits.count("V")
        n_train += nt
        n_val += nv
        m = meta.get(tid, {})
        vent, tr = hms(m.get("bat_venting_time")), hms(m.get("bat_abn_temprise"))
        dur = times[-1] - times[0]
        if vent is not None:
            vent_elapsed_ok += 0 <= vent <= dur
            vent_clock_ok += times[0] <= vent <= times[-1]
        row = {"test_id": tid, "형태": cond[tid][0], "용량": cond[tid][1],
               "SOC": cond[tid][2], "방식": cond[tid][3],
               "n_train": nt, "n_val": nv, "start_tod_s": times[0],
               "duration_s": round(dur, 1),
               "dt_median_s": median(b - a for a, b in zip(times, times[1:])) if len(times) > 1 else "",
               "stage_monotonic": mono, "val_runs": v_runs,
               "val_first_pos": round(splits.find("V") / len(splits), 3) if nv else "",
               "venting_raw": m.get("bat_venting_time"), "venting_s": vent,
               "tr_raw": m.get("bat_abn_temprise"), "tr_s": tr}
        for s in range(2, 7):
            row["first_stage%d_s" % s] = first_stage.get(s, "")
        for f in ABN:
            row["first_" + f + "_s"] = first_abn.get(f, "")
        rows.append(row)

    say("zip 내부 경로 예:", example_paths)
    say("레코드 수: Training", n_train, "/ Validation", n_val, "/ 합계", n_train + n_val,
        "(전체 구축량 242,580)")
    say("필드 구성의 종류:", len(key_sets))
    for ks, n in key_sets.most_common(3):
        say("  %d건: 필드 %d개 %s" % (n, len(ks), list(ks)))
    say("비어 있는 값(필드별 건수):", dict(nulls))
    say("레코드 간격(초) 상위:", dts.most_common(6))
    say("data_id 순서와 시간 순서가 다른 시험:", n_bad_order)
    say("단계가 감소하는 구간이 있는 시험:", n_not_mono)
    say("단계 분포:", dict(sorted(stage_total.items(), key=lambda x: str(x[0]))))
    for f in ABN:
        ys = abn_y[f]
        say("  %s: Y가 있는 시험 %d개, 시험당 Y 레코드 수 최소 %d / 중앙 %s / 최대 %d"
            % (f, sum(y > 0 for y in ys), min(ys), median(ys), max(ys)))
    runs = [r["val_runs"] for r in rows if r["n_val"]]
    if runs:
        say("Validation이 시간축에서 이루는 덩어리 수: 최소 %d / 중앙 %s / 최대 %d"
            % (min(runs), median(runs), max(runs)),
            "(1이면 연속 구간, 크면 Training 사이에 흩어져 있음)")
        say("Validation이 처음 나타나는 위치(0=시작, 1=끝)의 중앙값:",
            median(r["val_first_pos"] for r in rows if r["n_val"]))
    say("벤팅 값이 시험 길이 이내(경과 시간으로 해석 가능)인 시험:", vent_elapsed_ok)
    say("벤팅 값이 시험 시각 범위 이내(시각으로 해석 가능)인 시험:", vent_clock_ok)

    # ---------- 4. 원천 센서 JSON과 라벨링 JSON 비교 (앞의 3개 시험) ----------
    say("\n== 4. 원천 센서 JSON ==")
    for tid in sorted(tests)[:3]:
        z = tests[tid].get("TS_센서")
        if not z:
            continue
        src = [r for _, r in json_records(z, "sensor_data")]
        lab = next((r for _, r in json_records(tests[tid]["TL_센서"], "labeling_data")), {}) \
            if "TL_센서" in tests[tid] else {}
        if src:
            say(tid, "원천", len(src), "건 / 라벨에만 있는 필드:",
                sorted(set(lab) - set(src[0])), "/ 원천에만 있는 필드:", sorted(set(src[0]) - set(lab)))
        else:
            with zipfile.ZipFile(z) as zf:
                say(tid, "sensor_data JSON 없음. 내부 파일 예:",
                    [entry_name(i) for i in zf.infolist()[:3]])

    # ---------- 5. 이미지 ----------
    say("\n== 5. 이미지 ==")
    shapes = defaultdict(Counter)   # 형태_용량 -> 이미지 규격
    n_png = missing = extra = 0
    for tid in sorted(tests):
        for img_kind, lab_kind in (("TS_이미지", "TL_센서"), ("VS_이미지", "VL_센서")):
            z = tests[tid].get(img_kind)
            if not z:
                continue
            with zipfile.ZipFile(z) as zf:
                pngs = [i for i in zf.infolist() if i.filename.lower().endswith(".png")]
                n_png += len(pngs)
                for i in {0, len(pngs) // 2, len(pngs) - 1} if pngs else ():
                    shapes["_".join(cond[tid][:2])][png_shape(zf, pngs[i])] += 1
                names = {entry_name(i).rsplit("/", 1)[-1] for i in pngs}
            if lab_kind in tests[tid]:
                want = {r.get("imgdata_id") for _, r in json_records(tests[tid][lab_kind], "labeling_data")}
                missing += len(want - names)
                extra += len(names - want)
    say("PNG 수:", n_png)
    say("라벨이 가리키지만 없는 이미지:", missing, "/ 라벨에 없는 이미지:", extra)
    for k in sorted(shapes):
        say("  %s: %s" % (k, dict(shapes[k])))

    Path("inspect_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    if rows:
        with open("tests_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("\n저장 완료: inspect_report.txt, tests_summary.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
