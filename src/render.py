"""video.html을 프레임 단위로 캡처해 무음 mp4로 만든다.

사용법: python3 src/render.py [--from 초] [--to 초] [--out 파일] [--no-tag] [--stills 초,초,...]
"""
import argparse, json, os, subprocess, sys
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="t0", type=float, default=0)
ap.add_argument("--to", dest="t1", type=float, default=None)
ap.add_argument("--out", default=f"{ROOT}/build/video.mp4")
ap.add_argument("--no-tag", action="store_true")
ap.add_argument("--stills", default=None, help="쉼표로 구분한 시각(초)의 정지 프레임만 저장")
ap.add_argument("--ref", default=f"{ROOT}/assets/ref")
a = ap.parse_args()

tl = json.load(open(f"{ROOT}/build/timeline.json"))
fps = tl["fps"]
html = open(f"{ROOT}/src/video.html", encoding="utf-8").read()
html = html.replace("/*TIMELINE*/null", json.dumps(tl, ensure_ascii=False))
html = html.replace("/*REF*/", "file://" + os.path.abspath(a.ref))
html = html.replace("/*SHOW_TAG*/true", "false" if a.no_tag else "true")
page_path = f"{ROOT}/build/_page_{os.getpid()}.html"
open(page_path, "w", encoding="utf-8").write(html)

with sync_playwright() as p:
    exe = "/opt/pw-browsers/chromium" if os.path.isfile("/opt/pw-browsers/chromium") else None
    b = p.chromium.launch(args=["--allow-file-access-from-files", "--force-color-profile=srgb"], **({"executable_path": exe} if exe else {}))
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    pg.on("pageerror", lambda e: print("PAGE ERROR", e, file=sys.stderr))
    pg.goto("file://" + page_path)
    pg.evaluate("window.ready")
    if a.stills:
        os.makedirs(f"{ROOT}/build/stills", exist_ok=True)
        for t in [float(x) for x in a.stills.split(",")]:
            pg.evaluate(f"window.seek({t})")
            pg.screenshot(path=f"{ROOT}/build/stills/t{t:07.2f}.jpg", type="jpeg", quality=88)
    else:
        t1 = a.t1 if a.t1 is not None else tl["total"]
        f0, f1 = round(a.t0 * fps), round(t1 * fps)
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-",
                               "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(fps), a.out], stdin=subprocess.PIPE)
        for f in range(f0, f1):
            pg.evaluate(f"window.seek({f / fps})")
            ff.stdin.write(pg.screenshot(type="jpeg", quality=95))
            if (f - f0) % 150 == 0:
                print(f"{f - f0}/{f1 - f0}", flush=True)
        ff.stdin.close(); ff.wait()
    b.close()
os.remove(page_path)
