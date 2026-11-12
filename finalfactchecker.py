from flask import Flask, request, jsonify, render_template_string
import torch
import requests
import json
import os
import re
import time
import subprocess
import threading
import datetime
import urllib.parse
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

app = Flask(__name__)

# ============ CONFIG ============
MODEL_PATH   = r"C:\Users\amd\Desktop\factcheck\saved\distilbert"
BASE_DIR     = r"C:\Users\amd\Desktop\factcheck"
LOG_FILE     = os.path.join(BASE_DIR, "check_log.txt")
JUDGE_LOG    = os.path.join(BASE_DIR, "judge_server.log")
LLAMA_SERVER = r"C:\Users\amd\Documents\jarvis\llama.cpp\build\bin\Release\llama-server.exe"
GGUF_PATH    = r"C:\Users\amd\Documents\jarvis\llama.cpp\model\mistral\Qwen3-8B-Q6_K.gguf"
MODEL_HINT   = "Qwen"
LLAMA_PORT   = 8082
MODELS_URL   = f"http://127.0.0.1:{LLAMA_PORT}/v1/models"
CHAT_URL     = f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"
CTX          = "16384"
BATCH_SIZE   = "1024"
THREADS      = "6"

UA_CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
UA_EDGE   = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0")
BROWSER_HEADERS = {
    "User-Agent": UA_CHROME,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",   # no "br" — requests can't decode brotli
}

JUDGE_READY = False
ANALYZE_LOCK = threading.Lock()

# ============ LAYER 1 : DISTILBERT ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[FactzChecker] Loading DistilBERT on", device, flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
bert_model.eval()
try:
    with open(f"{MODEL_PATH}/label_map.json") as f:
        LABELS = json.load(f)
except Exception:
    LABELS = ["FAKE", "REAL"]

def quick_scan(text):
    inputs = tokenizer(text[:1000], max_length=512, truncation=True,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(bert_model(**inputs).logits, dim=1)[0]
    return LABELS[probs.argmax().item()], probs.max().item()

# ============ LIVE PROGRESS ============
PROG_LOCK = threading.Lock()

def fresh_progress():
    return {
        "running": False,
        "steps": [
            {"name": "AI STYLE SCAN",   "status": "idle", "detail": ""},
            {"name": "WEB RESEARCH",    "status": "idle", "detail": ""},
            {"name": "DEEP FACT JUDGE", "status": "idle", "detail": ""},
        ],
    }

PROGRESS = fresh_progress()

def set_running(v):
    with PROG_LOCK:
        PROGRESS["running"] = v

def set_step(i, status, detail=""):
    with PROG_LOCK:
        PROGRESS["steps"][i]["status"] = status
        PROGRESS["steps"][i]["detail"] = detail

def progress_snapshot():
    with PROG_LOCK:
        return {
            "judge_ready": JUDGE_READY or judge_model_loaded(),
            "steps": [dict(s) for s in PROGRESS["steps"]],
            "running": PROGRESS["running"],
        }

def last_lines(path, n=3):
    try:
        with open(path, "r", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return " | ".join(lines[-n:]) or "empty log"
    except Exception:
        return "no log available"

# ============ ARTICLE EXTRACTION (3-method fallback) ============
def _html_to_article(html, url):
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    title = re.split(r"\s*[\|\-–—]\s*", title)[0].strip()
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
    if len(text) < 200:
        text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    return title, text

def _md_to_article(md, url):
    """Jina Reader returns markdown with a 'Title: ...' header line."""
    title = ""
    lines = md.splitlines()
    body_lines = []
    for line in lines:
        if not title and line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("url source:") or line.lower().startswith("warning:"):
            continue
        else:
            body_lines.append(line)
    text = " ".join(l.strip() for l in body_lines if l.strip())
    text = " ".join(text.split())
    return title, text

def fetch_direct(url, ua):
    headers = dict(BROWSER_HEADERS)
    headers["User-Agent"] = ua
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text

def fetch_via_jina(url):
    """Free reader proxy — bypasses most bot walls, returns markdown."""
    r = requests.get("https://r.jina.ai/" + url,
                     headers={"User-Agent": UA_CHROME}, timeout=40)
    r.raise_for_status()
    return r.text

def fetch_via_wayback(url):
    """Last resort: archived copy from the Internet Archive."""
    api = "https://archive.org/wayback/available?url=" + urllib.parse.quote_plus(url)
    r = requests.get(api, headers=BROWSER_HEADERS, timeout=15)
    r.raise_for_status()
    snap = (r.json().get("archived_snapshots") or {}).get("closest") or {}
    if not snap.get("url"):
        raise RuntimeError("no archived snapshot")
    r2 = requests.get(snap["url"], headers=BROWSER_HEADERS, timeout=30)
    r2.raise_for_status()
    return r2.text

def extract_article(url):
    errors = []

    # method 1: direct fetch, two browser fingerprints
    for ua in (UA_CHROME, UA_EDGE):
        try:
            set_step(0, "idle", "")  # placeholder no-op, extraction progress shown in UI hint
            html = fetch_direct(url, ua)
            title, text = _html_to_article(html, url)
            if len(text) >= 100:
                return title, text
            errors.append("direct: too little text")
        except Exception as e:
            errors.append(f"direct: {e.__class__.__name__} {e}")

    # method 2: Jina reader proxy
    try:
        md = fetch_via_jina(url)
        title, text = _md_to_article(md, url)
        if len(text) >= 100:
            return title, text
        errors.append("jina: too little text")
    except Exception as e:
        errors.append(f"jina: {e.__class__.__name__} {e}")

    # method 3: Wayback Machine snapshot
    try:
        html = fetch_via_wayback(url)
        title, text = _html_to_article(html, url)
        if len(text) >= 100:
            return title, text
        errors.append("wayback: too little text")
    except Exception as e:
        errors.append(f"wayback: {e.__class__.__name__} {e}")

    raise Exception("all fetch methods failed (" + " | ".join(errors[:3]) +
                    "). Try pasting the article text instead.")

# ============ LAYER 2 : DEEP RESEARCH ============
def search_with_ddgs(query):
    results = []
    with DDGS(timeout=8) as ddgs:
        for r in ddgs.news(query, max_results=5, safesearch="off"):
            results.append({"title": r.get("title", ""), "snippet": r.get("body", ""),
                            "url": r.get("url", ""), "source": "news search"})
        for r in ddgs.text(query, max_results=5, safesearch="off"):
            results.append({"title": r.get("title", ""), "snippet": r.get("body", ""),
                            "url": r.get("href", r.get("url", "")), "source": "web search"})
    return results

def read_article_body(link, limit=500):
    try:
        # jina first for blocked sites, direct as fallback
        try:
            md = fetch_via_jina(link)
            _, text = _md_to_article(md, link)
            return text[:limit]
        except Exception:
            r = requests.get(link, headers=BROWSER_HEADERS, timeout=12)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
            return " ".join(text.split())[:limit]
    except Exception:
        return ""

def deep_research(article_title, article_text):
    query = (article_title if len(article_title) > 10
             else " ".join(article_text.split()[:15]))[:70]

    set_step(1, "running", "Searching the web...")
    raw = []
    try:
        raw = search_with_ddgs(query)
    except Exception as e:
        set_step(1, "running", f"Primary search failed ({e}), trying news feed...")
        try:
            rss = ("https://news.google.com/rss/search?q=" + urllib.parse.quote_plus(query)
                   + "&hl=en-US&gl=US&ceid=US:en")
            r = requests.get(rss, headers=BROWSER_HEADERS, timeout=15)
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                t = (item.findtext("title") or "").strip()
                d = (item.findtext("description") or "").strip()
                if t:
                    raw.append({"title": t,
                                "snippet": BeautifulSoup(d, "html.parser").get_text(" ", strip=True)[:300],
                                "url": "", "source": "news feed"})
        except Exception:
            pass

    sources, seen_titles, seen_domains = [], set(), set()
    for item in raw:
        if not item.get("title"):
            continue
        key = item["title"][:50].lower()
        url = item.get("url", "")
        domain = re.sub(r"^www\.", "", re.sub(r"https?://", "", url)).split("/")[0] if url else ""
        if key in seen_titles:
            continue
        if domain and domain in seen_domains:
            continue
        seen_titles.add(key)
        if domain:
            seen_domains.add(domain)
        sources.append(item)
        if len(sources) >= 6:
            break

    if not sources:
        set_step(1, "done", "No coverage found on the web")
        return []

    read_count = 0
    for item in sources:
        if not item.get("url") or read_count >= 3:
            continue
        set_step(1, "running", f"Reading article {read_count + 1}/3...")
        body = read_article_body(item["url"])
        if body:
            item["body"] = body
            read_count += 1

    set_step(1, "done", f"{len(sources)} sources found, {read_count} articles read")
    return sources

# ============ LAYER 3 : JUDGE (Qwen3, /no_think) ============
SYSTEM_PROMPT = (
    "You are a strict fact-check judge. You are given one ARTICLE and a few WEB SOURCES. "
    "Judge the article ONLY using the sources. NEVER use your own knowledge or memory. "
    "Do NOT write any essay, list, guide, or explanation outside the JSON.\n"
    "Return ONE JSON object in exactly this shape:\n"
    '{"claims":[{"claim":"main claim","status":"SUPPORTED","evidence":"quote from a source","source":"name"}],'
    '"final_verdict":"REAL","confidence":80,"reasoning":"one short sentence"}\n'
    "status values: SUPPORTED | CONTRADICTED | UNVERIFIED.\n"
    "final_verdict: FAKE if a major claim is CONTRADICTED; REAL if most are SUPPORTED and none CONTRADICTED; "
    "MIXED if key claims are UNVERIFIED.\n"
    "confidence 0-100 = how well the sources cover the article.\n"
    "Your entire reply must be that single JSON object. Nothing else."
)

def _build_user_prompt(article, sources):
    lines = [f"ARTICLE:\n{article[:1000]}", "", "WEB SOURCES:"]
    for i, s in enumerate(sources[:5], 1):
        blob = s.get("body") or s.get("snippet") or ""
        lines.append(f"{i}. [{s.get('source','web')}] {s.get('title','')} — {blob[:350]}")
    lines.append("")
    lines.append("Reply with ONLY the JSON object, nothing else.")
    return "/no_think\n" + "\n".join(lines)

def _parse_judge_json(content):
    c = content.strip()
    c = re.sub(r"^```(?:json)?\s*", "", c)
    c = re.sub(r"\s*```$", "", c)
    start, end = c.find("{"), c.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge returned no JSON. Raw reply:\n---\n{content[:600]}\n---")
    return json.loads(c[start:end+1])

def ask_judge(article, sources):
    user_prompt = _build_user_prompt(article, sources)
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    payload = {
        "model": "local",
        "messages": base_messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": 1200,
        "stream": False,
    }

    last_raw = ""
    for attempt in range(3):
        if attempt == 1:
            payload["messages"] = base_messages + [
                {"role": "assistant", "content": last_raw[:300]},
                {"role": "user", "content":
                    "/no_think\nWrong. That was not the JSON I asked for. Output ONLY the single JSON object now."},
            ]
        elif attempt == 2:
            payload["messages"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    "/no_think\n" + user_prompt +
                    "\n\nAgain, output ONLY: {\"claims\":[...],\"final_verdict\":\"...\",\"confidence\":N,\"reasoning\":\"...\"}"},
            ]
        print(f"[FactzChecker] Judge attempt {attempt + 1}...", flush=True)

        r = requests.post(CHAT_URL, json=payload, timeout=300)
        if r.status_code == 503:
            raise RuntimeError("model still loading on the judge server")
        r.raise_for_status()
        last_raw = r.json()["choices"][0]["message"]["content"].strip()

        try:
            data = _parse_judge_json(last_raw)
            if "final_verdict" in data:
                return data
            raise ValueError("JSON missing final_verdict")
        except json.JSONDecodeError as e:
            print(f"[FactzChecker] Judge JSON parse failed (attempt {attempt + 1}): {e}", flush=True)
            continue
        except ValueError as e:
            print(f"[FactzChecker] Judge output invalid (attempt {attempt + 1}): {e}", flush=True)
            continue

    try:
        with open(os.path.join(BASE_DIR, "last_judge_response.txt"), "w", encoding="utf-8") as f:
            f.write(last_raw)
        print("[FactzChecker] Raw judge reply saved to last_judge_response.txt", flush=True)
    except Exception:
        pass
    raise ValueError("Judge failed to return usable JSON after 3 attempts (see last_judge_response.txt)")

# ============ JUDGE AUTO-START ============
def _get_loaded_models():
    try:
        r = requests.get(MODELS_URL, timeout=2)
        if r.status_code != 200:
            return []
        return [m.get("id", "") for m in r.json().get("data", [])]
    except Exception:
        return []

def judge_model_loaded():
    return any(MODEL_HINT in mid for mid in _get_loaded_models())

def _kill_all_llama_servers():
    try:
        out = subprocess.check_output(
            r'powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"name=\'llama-server.exe\'\").ProcessId"',
            shell=True, text=True, creationflags=0x08000000)
        killed = False
        for pid in out.split():
            pid = pid.strip()
            if pid.isdigit():
                subprocess.run(["taskkill", "/F", "/PID", pid],
                               creationflags=0x08000000)
                print(f"[FactzChecker] Killed stale llama-server PID {pid}", flush=True)
                killed = True
        if not killed:
            print("[FactzChecker] No stale llama-server processes found.", flush=True)
    except Exception as e:
        print(f"[FactzChecker] Could not scan for stale servers: {e}", flush=True)

def judge_self_test():
    try:
        r = requests.post(CHAT_URL, json={
            "model": "local",
            "messages": [{"role": "user",
                          "content": "/no_think\nReply with ONLY the JSON object {\"ok\":1}. Nothing else."}],
            "temperature": 0.0, "max_tokens": 60, "stream": False,
        }, timeout=180)
        r.raise_for_status()
        c = r.json()["choices"][0]["message"]["content"].strip()
        return json.loads(c[c.find("{"):c.rfind("}") + 1]).get("ok") == 1
    except Exception as e:
        print(f"[FactzChecker] Judge self-test error: {e}", flush=True)
        return False

def start_judge():
    global JUDGE_READY

    if not os.path.isfile(LLAMA_SERVER):
        print("[FactzChecker] ERROR: llama-server.exe not found:\n  " + LLAMA_SERVER, flush=True)
        return
    if not os.path.isfile(GGUF_PATH):
        print("[FactzChecker] ERROR: judge model not found:\n  " + GGUF_PATH, flush=True)
        return

    size_gb = os.path.getsize(GGUF_PATH) / (1024 ** 3)
    print(f"[FactzChecker] Judge model: {os.path.basename(GGUF_PATH)} ({size_gb:.1f} GB)", flush=True)
    if size_gb < 6:
        print(f"[FactzChecker] WARNING: file is only {size_gb:.2f} GB — Qwen3-8B Q8_0 should be "
              "~8.5 GB. Download is probably incomplete!", flush=True)

    if judge_model_loaded():
        print("[FactzChecker] Judge already running with the correct model. Ready.", flush=True)
        JUDGE_READY = True
        return

    _kill_all_llama_servers()
    time.sleep(4)

    cmd = [LLAMA_SERVER, "-m", GGUF_PATH,
           "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
           "--batch-size", BATCH_SIZE, "-c", CTX, "-t", THREADS,
           "--n-gpu-layers", "999", "--flash-attn", "-ngl", "999",
           "--jinja", "--cont-batching", "--mlock",
           "--parallel", "1", "-np", "1",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    print("[FactzChecker] Launching judge server (log streamed below)...", flush=True)

    with open(JUDGE_LOG, "w") as logf:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=logf,
                         creationflags=0x08000000)
        t0, printed = time.time(), 0
        while time.time() - t0 < 240:
            time.sleep(3)
            try:
                with open(JUDGE_LOG, "r", errors="ignore") as f:
                    lines = [l.rstrip() for l in f if l.strip()]
                if len(lines) > printed:
                    for l in lines[printed:]:
                        print("   [llama] " + l[:180], flush=True)
                    printed = len(lines)
            except Exception:
                pass
            if judge_model_loaded():
                break

        if judge_model_loaded():
            if judge_self_test():
                JUDGE_READY = True
                print("[FactzChecker] Judge self-test PASSED. Full 3-layer mode ON.", flush=True)
            else:
                print("[FactzChecker] Judge loaded but self-test FAILED. Log tail:", flush=True)
                print("   " + last_lines(JUDGE_LOG, 8).replace(" | ", "\n   "), flush=True)
        else:
            print("[FactzChecker] JUDGE FAILED TO LOAD in 4 minutes. Log tail:", flush=True)
            print("   " + last_lines(JUDGE_LOG, 20).replace(" | ", "\n   "), flush=True)
            print("   (full log: " + JUDGE_LOG + ")", flush=True)

threading.Thread(target=start_judge, daemon=True).start()

# ============ LOGGING ============
def count_checks():
    if not os.path.exists(LOG_FILE):
        return 0
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def log_check(content, verdict, conf, mode):
    now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{now}] | {verdict:5s} | {conf}% | {mode:5s} | {content[:70]}\n")

# ============ UI ============
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FactzChecker · 3-Layer Fake News Detector</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  :root{
    --bg:#070b14; --line:#243150; --txt:#eaf0fc; --muted:#8b99ba;
    --real:#2dd4bf; --fake:#fb5e71; --mixed:#fbbf24; --accent:#6c8cff;
  }
  body{
    font-family:'Segoe UI',system-ui,sans-serif; color:var(--txt); min-height:100vh;
    background:
      radial-gradient(1100px 500px at 85% -10%, rgba(108,140,255,.18) 0%, transparent 60%),
      radial-gradient(900px 500px at 10% 110%, rgba(45,212,191,.10) 0%, transparent 60%),
      var(--bg);
    display:flex; flex-direction:column; align-items:center; padding:44px 16px;
  }
  body::before{
    content:""; position:fixed; inset:0; pointer-events:none; opacity:.05;
    background-image:linear-gradient(var(--accent) 1px,transparent 1px),
                     linear-gradient(90deg,var(--accent) 1px,transparent 1px);
    background-size:44px 44px;
  }
  .logo{ display:flex; align-items:center; gap:14px; }
  .logo .mark{
    width:52px; height:52px; border-radius:14px; font-size:26px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,var(--accent),#a06cff);
    animation:glow 3s ease-in-out infinite;
  }
  @keyframes glow{ 0%,100%{ box-shadow:0 0 24px rgba(108,140,255,.4);} 50%{ box-shadow:0 0 46px rgba(108,140,255,.8);} }
  .logo h1{ font-size:26px; letter-spacing:.5px; }
  .logo h1 span{ background:linear-gradient(90deg,var(--accent),#a06cff);
    -webkit-background-clip:text; background-clip:text; color:transparent; }
  .sub{ color:var(--muted); font-size:13px; margin:10px 0 22px; text-align:center; max-width:540px; }
  .boot{
    width:100%; max-width:680px; margin-bottom:18px; padding:14px 18px;
    background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.4);
    border-radius:14px; font-size:13px; color:var(--mixed); display:none;
  }
  .boot.show{ display:block; }
  .boot .track{ height:6px; background:#0c1322; border-radius:20px; margin-top:10px; overflow:hidden; }
  .boot .ind{ height:100%; width:35%; border-radius:20px;
    background:linear-gradient(90deg,transparent,var(--mixed),transparent);
    animation:slide 1.4s linear infinite; }
  @keyframes slide{ from{transform:translateX(-120%);} to{transform:translateX(400%);} }
  .card{
    width:100%; max-width:680px; background:rgba(19,26,42,.85);
    border:1px solid var(--line); border-radius:22px; padding:26px;
    backdrop-filter:blur(14px); box-shadow:0 30px 70px rgba(0,0,0,.5); position:relative; overflow:hidden;
  }
  .tabs{ display:flex; gap:8px; margin-bottom:16px; }
  .tab{ flex:1; padding:12px; border-radius:12px; cursor:pointer; text-align:center;
    font-size:14px; font-weight:600; color:var(--muted);
    background:rgba(26,35,54,.7); border:1px solid var(--line); transition:.2s; }
  .tab.active{ color:#fff; border-color:var(--accent);
    background:linear-gradient(180deg,rgba(108,140,255,.22),rgba(108,140,255,.06)); }
  .urlbox, textarea{ width:100%; background:#0c1322; border:1px solid var(--line); border-radius:14px;
    color:var(--txt); padding:14px; font-size:14px; font-family:inherit; transition:.2s; }
  textarea{ min-height:150px; resize:vertical; }
  .urlbox:focus, textarea:focus{ outline:none; border-color:var(--accent);
    box-shadow:0 0 0 3px rgba(108,140,255,.15); }
  .btn{ width:100%; margin-top:16px; padding:15px; border:none; border-radius:14px;
    cursor:pointer; font-size:15px; font-weight:700; color:#fff;
    background:linear-gradient(135deg,var(--accent),#a06cff);
    box-shadow:0 12px 32px rgba(108,140,255,.35); transition:.2s; }
  .btn:hover{ transform:translateY(-2px); }
  .btn:disabled{ opacity:.55; cursor:wait; transform:none; }
  .err{ margin-top:14px; color:var(--fake); font-size:13px; text-align:center; display:none; }
  .mainbar{ display:none; margin-top:18px; }
  .mainbar.show{ display:block; }
  .mainbar .track{ height:8px; background:#0c1322; border-radius:20px; overflow:hidden; }
  .mainbar .ind{ height:100%; width:30%; border-radius:20px;
    background:linear-gradient(90deg,transparent,var(--accent),#a06cff,transparent);
    animation:slide 1.3s linear infinite; }
  .pipe{ display:none; margin-top:18px; gap:10px; }
  .pipe.show{ display:grid; grid-template-columns:1fr 1fr 1fr; }
  .step{ background:rgba(26,35,54,.7); border:1px solid var(--line); border-radius:14px;
    padding:14px 8px 12px; text-align:center; font-size:11px; color:var(--muted); transition:.3s; min-height:96px; }
  .step .ico{ font-size:22px; display:block; margin-bottom:6px; }
  .step .nm{ font-weight:700; letter-spacing:.5px; }
  .step .det{ font-size:10px; color:var(--muted); margin-top:6px; min-height:24px; line-height:1.4; }
  .step .mini{ height:4px; background:#0c1322; border-radius:10px; margin-top:8px; overflow:hidden; }
  .step .mini .f{ height:100%; width:0; border-radius:10px; background:var(--accent); transition:width .5s; }
  .step.on{ border-color:var(--accent); color:#fff;
    background:linear-gradient(180deg,rgba(108,140,255,.2),rgba(108,140,255,.05)); }
  .step.on .ico{ animation:spin 1.1s linear infinite; display:inline-block; }
  .step.on .mini .f{ width:70%; animation:pulsebar 1.2s ease-in-out infinite; }
  @keyframes pulsebar{ 0%,100%{opacity:1;} 50%{opacity:.4;} }
  .step.done{ border-color:var(--real); color:var(--real); }
  .step.done .nm{ color:var(--real); }
  .step.done .det{ color:var(--txt); }
  .step.done .mini .f{ width:100%; background:var(--real); animation:none; }
  .step.done .ico::after{ content:" ✓"; }
  @keyframes spin{ to{ transform:rotate(360deg);} }
  .result{ margin-top:24px; display:none; }
  .result.show{ display:block; animation:pop .45s ease; }
  @keyframes pop{ from{opacity:0; transform:translateY(16px) scale(.98);} to{opacity:1; transform:none;} }
  .verdict{ border-radius:18px; padding:28px; text-align:center; border:1px solid var(--line); background:rgba(26,35,54,.7); }
  .verdict.REAL{ border-color:rgba(45,212,191,.55); background:linear-gradient(180deg,rgba(45,212,191,.14),rgba(26,35,54,.7)); }
  .verdict.FAKE{ border-color:rgba(251,94,113,.55); background:linear-gradient(180deg,rgba(251,94,113,.14),rgba(26,35,54,.7)); }
  .verdict.MIXED{ border-color:rgba(251,191,36,.55); background:linear-gradient(180deg,rgba(251,191,36,.14),rgba(26,35,54,.7)); }
  .badge{ font-size:36px; font-weight:800; letter-spacing:3px; }
  .verdict.REAL .badge{ color:var(--real);} .verdict.FAKE .badge{ color:var(--fake);} .verdict.MIXED .badge{ color:var(--mixed);}
  .tag{ color:var(--muted); font-size:13px; margin-top:4px; }
  .mode{ display:inline-block; margin-top:10px; font-size:11px; padding:4px 12px;
    border-radius:20px; border:1px solid var(--line); color:var(--muted); }
  .meter{ margin-top:20px; }
  .meter .top{ display:flex; justify-content:space-between; font-size:13px; color:var(--muted); margin-bottom:8px; }
  .bar{ height:12px; background:#0c1322; border-radius:20px; overflow:hidden; }
  .fill{ height:100%; width:0; border-radius:20px; transition:width 1.2s cubic-bezier(.2,.9,.3,1); }
  .verdict.REAL .fill{ background:linear-gradient(90deg,#7ceadb,var(--real)); }
  .verdict.FAKE .fill{ background:linear-gradient(90deg,#ff8a98,var(--fake)); }
  .verdict.MIXED .fill{ background:linear-gradient(90deg,#ffe08a,var(--mixed)); }
  .reason{ margin-top:16px; font-size:13.5px; line-height:1.65; text-align:left; }
  .l1chip{ margin-top:14px; font-size:11.5px; color:var(--muted); text-align:center; }
  .section{ margin-top:22px; }
  .section h3{ font-size:12px; letter-spacing:1.5px; color:var(--muted); margin-bottom:10px; }
  .claim{ background:rgba(12,19,34,.8); border:1px solid var(--line); border-radius:12px;
    padding:12px 14px; margin-bottom:8px; font-size:13px; line-height:1.5; }
  .chip{ display:inline-block; font-size:10.5px; font-weight:700; padding:2px 10px;
    border-radius:12px; margin-bottom:6px; letter-spacing:.5px; }
  .chip.SUPPORTED{ background:rgba(45,212,191,.15); color:var(--real); border:1px solid rgba(45,212,191,.4);}
  .chip.CONTRADICTED{ background:rgba(251,94,113,.15); color:var(--fake); border:1px solid rgba(251,94,113,.4);}
  .chip.UNVERIFIED{ background:rgba(251,191,36,.15); color:var(--mixed); border:1px solid rgba(251,191,36,.4);}
  .ev{ color:var(--muted); font-size:12px; margin-top:4px; }
  .src{ display:block; font-size:12.5px; padding:8px 0; border-bottom:1px dashed var(--line);
    color:var(--txt); text-decoration:none; }
  .src:hover{ color:var(--accent); }
  .src b{ color:var(--accent); font-weight:600; }
  .footer{ color:#4b5872; font-size:11px; margin-top:26px; text-align:center; line-height:1.9; }
  .footer .count{ color:var(--accent); font-weight:700; font-size:13px; }
</style>
</head>
<body>
  <div class="logo"><div class="mark">🛡️</div><h1>Factz<span>Checker</span></h1></div>
  <div class="sub">Three-layer verification — AI style scan, live web research reading real articles, then a local 8B AI judge cross-checks every claim against the evidence.</div>

  <div class="boot" id="boot">
    ⚙️ Local 8B AI judge is loading onto your GPU (1-2 minutes). Quick Scan works now —
    full 3-layer mode unlocks automatically.
    <div class="track"><div class="ind"></div></div>
  </div>

  <div class="card">
    <div class="tabs">
      <div class="tab active" id="tabUrl" onclick="setMode('url')">🔗 Paste Link</div>
      <div class="tab" id="tabText" onclick="setMode('text')">📝 Paste Article</div>
    </div>
    <input class="urlbox" id="urlInput" placeholder="https://apnews.com/article/...">
    <textarea id="textInput" style="display:none"
      placeholder="Paste the full article text here (title + body works best)..."></textarea>

    <button class="btn" id="btn" onclick="run()">⚡ Verify This</button>
    <div class="err" id="err"></div>
    <div class="mainbar" id="mainbar"><div class="track"><div class="ind"></div></div></div>

    <div class="pipe" id="pipe">
      <div class="step" id="step0"><span class="ico">🧠</span><span class="nm">AI STYLE SCAN</span><div class="det" id="det0"></div><div class="mini"><div class="f" id="f0"></div></div></div>
      <div class="step" id="step1"><span class="ico">🌐</span><span class="nm">WEB RESEARCH</span><div class="det" id="det1"></div><div class="mini"><div class="f" id="f1"></div></div></div>
      <div class="step" id="step2"><span class="ico">⚖️</span><span class="nm">DEEP FACT JUDGE</span><div class="det" id="det2"></div><div class="mini"><div class="f" id="f2"></div></div></div>
    </div>

    <div class="result" id="result">
      <div class="verdict" id="verdict">
        <div class="badge" id="badge">—</div>
        <div class="tag" id="tag">—</div>
        <div class="mode" id="mode">—</div>
        <div class="meter">
          <div class="top"><span>Confidence</span><span id="confTxt">0%</span></div>
          <div class="bar"><div class="fill" id="fill"></div></div>
        </div>
        <div class="reason" id="reason"></div>
        <div class="l1chip" id="l1chip"></div>
      </div>
      <div class="section" id="claimsSec" style="display:none">
        <h3>CLAIM-BY-CLAIM VERDICT</h3><div id="claims"></div>
      </div>
      <div class="section" id="srcSec" style="display:none">
        <h3>RESEARCHED SOURCES</h3><div id="srcs"></div>
      </div>
    </div>
  </div>

  <div class="footer">
    <div class="count" id="counter">📊 Total Verifications: —</div>
    DistilBERT + Web Research + 8B Judge · all on local server
  </div>

<script>
let mode='url';
let pollTimer=null;
function setMode(m){
  mode=m;
  document.getElementById('tabUrl').classList.toggle('active',m==='url');
  document.getElementById('tabText').classList.toggle('active',m==='text');
  document.getElementById('urlInput').style.display = m==='url'?'block':'none';
  document.getElementById('textInput').style.display = m==='text'?'block':'none';
  document.getElementById('result').classList.remove('show');
  document.getElementById('err').style.display='none';
}
function showErr(m){const e=document.getElementById('err');e.textContent=m;e.style.display=m?'block':'none';}
function renderSteps(steps){
  steps.forEach((s,i)=>{
    const el=document.getElementById('step'+i);
    el.className='step'+(s.status==='running'?' on':(s.status==='done'?' done':''));
    document.getElementById('det'+i).textContent=s.detail||'';
  });
}
function startPolling(){
  stopPolling();
  pollTimer=setInterval(async ()=>{
    try{
      const d=await (await fetch('/status')).json();
      renderSteps(d.steps);
      if(d.judge_ready) document.getElementById('boot').classList.remove('show');
    }catch(e){}
  }, 600);
}
function stopPolling(){ if(pollTimer){clearInterval(pollTimer);pollTimer=null;} }
async function run(){
  const btn=document.getElementById('btn');
  const result=document.getElementById('result');
  const pipe=document.getElementById('pipe');
  const mainbar=document.getElementById('mainbar');
  const content = mode==='url'
    ? document.getElementById('urlInput').value.trim()
    : document.getElementById('textInput').value.trim();
  if(!content){ showErr(mode==='url'?'Paste a link first.':'Paste some article text first.'); return; }
  showErr(''); btn.disabled=true; btn.textContent='⚡ Verifying...';
  result.classList.remove('show');
  pipe.classList.add('show'); mainbar.classList.add('show');
  [0,1,2].forEach(i=>{document.getElementById('step'+i).className='step';document.getElementById('det'+i).textContent='';});
  startPolling();
  try{
    const res=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mode,content})});
    const data=await res.json();
    if(!res.ok){ throw new Error(data.error||'Server error.'); }
    stopPolling();
    renderSteps(data.steps);
    const v=document.getElementById('verdict');
    v.className='verdict '+data.verdict;
    document.getElementById('badge').textContent=data.verdict;
    const tagMap={REAL:'Verified — consistent with researched sources',
                  FAKE:'Likely Fake — contradicted by evidence',
                  MIXED:'Mixed — evidence insufficient or conflicting'};
    document.getElementById('tag').textContent=tagMap[data.verdict]||'';
    document.getElementById('mode').textContent=data.full_mode
      ? '✓ FULL 3-LAYER ANALYSIS COMPLETE' : '⚡ QUICK SCAN COMPLETE (judge offline / no sources)';
    document.getElementById('reason').textContent=data.reasoning||'';
    document.getElementById('l1chip').textContent=
      'Layer 1 scan: '+data.layer1.prediction+' at '+(data.layer1.confidence*100).toFixed(0)+'% · '+data.layer1.chars+' chars analysed';
    const cSec=document.getElementById('claimsSec'); const cBox=document.getElementById('claims');
    cBox.innerHTML='';
    if(data.claims && data.claims.length){
      cSec.style.display='block';
      data.claims.forEach(c=>{
        const d=document.createElement('div'); d.className='claim';
        d.innerHTML='<span class="chip '+(c.status||'UNVERIFIED')+'">'+(c.status||'UNVERIFIED')+
          '</span><div>'+(c.claim||'')+'</div>'+
          (c.evidence?'<div class="ev">Evidence: '+c.evidence+'</div>':'')+
          (c.source?'<div class="ev">Source: '+c.source+'</div>':'');
        cBox.appendChild(d);
      });
    } else cSec.style.display='none';
    const sSec=document.getElementById('srcSec'); const sBox=document.getElementById('srcs');
    sBox.innerHTML='';
    if(data.sources && data.sources.length){
      sSec.style.display='block';
      data.sources.forEach(s=>{
        const a=document.createElement('a');
        a.className='src'; a.href=s.link||'#'; a.target='_blank';
        a.innerHTML='<b>'+(s.source||'web result')+'</b> — '+(s.title||'');
        sBox.appendChild(a);
      });
    } else sSec.style.display='none';
    const conf=Math.round(data.confidence*100);
    document.getElementById('confTxt').textContent=conf+'%';
    document.getElementById('fill').style.width='0';
    requestAnimationFrame(()=>requestAnimationFrame(()=>{document.getElementById('fill').style.width=conf+'%';}));
    document.getElementById('counter').textContent='📊 Total Verifications: '+data.total_checks;
    result.classList.add('show');
  }catch(e){
    stopPolling();
    [0,1,2].forEach(i=>{document.getElementById('step'+i).className='step';});
    showErr(e.message);
  }finally{
    btn.disabled=false; btn.textContent='⚡ Verify This';
    setTimeout(()=>mainbar.classList.remove('show'), 500);
  }
}
function pollBoot(){
  fetch('/status').then(r=>r.json()).then(d=>{
    if(d.judge_ready) document.getElementById('boot').classList.remove('show');
    else setTimeout(pollBoot, 3000);
  }).catch(()=>setTimeout(pollBoot, 4000));
}
pollBoot();
fetch('/stats').then(r=>r.json()).then(d=>{
  document.getElementById('counter').textContent='📊 Total Verifications: '+d.total_checks;
}).catch(()=>{});
</script>
</body>
</html>
"""

# ============ ROUTES ============
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/status")
def status():
    return jsonify(progress_snapshot())

@app.route("/stats")
def stats():
    return jsonify(total_checks=count_checks())

def _analyze():
    global PROGRESS
    data = request.json or {}
    mode = data.get("mode", "text")
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify(error="No input provided."), 400

    with PROG_LOCK:
        for s in PROGRESS["steps"]:
            s["status"] = "idle"; s["detail"] = ""
        PROGRESS["running"] = True

    try:
        title = ""
        if mode == "url":
            try:
                title, text = extract_article(content)
            except Exception as e:
                return jsonify(error=f"Could not fetch that page — {e}"), 400
        else:
            text = " ".join(content.split())
            title = text[:100]

        if len(text) < 100:
            return jsonify(error="Not enough text to analyse (need at least ~100 characters)."), 400

        set_step(0, "running", "Running classifier...")
        try:
            l1_label, l1_conf = quick_scan(text)
        except Exception as e:
            return jsonify(error=f"Classifier failed: {e}"), 500
        set_step(0, "done", f"COMPLETE — {l1_label} ({l1_conf*100:.0f}%)")

        sources = []
        try:
            sources = deep_research(title, text)
        except Exception as e:
            set_step(1, "done", f"Research failed: {e}")
        if not sources:
            set_step(2, "done", "SKIPPED — nothing to verify against")

        verdict, confidence, claims, reasoning, full_mode = l1_label, l1_conf, [], "", False
        if sources:
            waited = 0
            while not judge_model_loaded() and waited < 120:
                set_step(2, "running", f"AI judge still loading on GPU... ({waited}s)")
                time.sleep(5)
                waited += 5
            if judge_model_loaded():
                set_step(2, "running", "Judge reading sources...")
                try:
                    j = ask_judge(text, sources)
                    verdict = str(j.get("final_verdict", "")).upper()
                    if verdict not in ("REAL", "FAKE", "MIXED"):
                        verdict = "MIXED"
                    try:
                        confidence = max(0.0, min(1.0, float(j.get("confidence", 50)) / 100.0))
                    except Exception:
                        confidence = 0.5
                    claims = j.get("claims", []) or []
                    reasoning = j.get("reasoning", "")
                    full_mode = True
                    set_step(2, "done", f"COMPLETE — verdict {verdict}, {len(claims)} claims checked")
                except Exception as e:
                    reasoning = (f"AI judge failed mid-analysis ({e}) — quick-scan result shown. "
                                 f"Server log: {last_lines(JUDGE_LOG, 3)}")
                    set_step(2, "done", "FALLBACK — judge error, quick-scan used")
            else:
                reasoning = ("AI judge did not finish loading — quick-scan result shown. "
                             "Check the console / judge_server.log, then retry for full 3-layer.")
                set_step(2, "done", "FALLBACK — judge not ready, quick-scan used")
        elif not sources:
            reasoning = "Web research found no coverage for this topic — showing quick-scan result only."

        try:
            log_check(content, verdict, round(confidence * 100), "FULL" if full_mode else "QUICK")
        except Exception:
            pass

        final_steps = [dict(s) for s in PROGRESS["steps"]]
        for s in final_steps:
            if s["status"] != "done":
                s["status"] = "done"
                s["detail"] = s["detail"] or "COMPLETE"

        pub_sources = [{"title": s.get("title", ""), "source": s.get("source", ""),
                        "link": s.get("url", "")} for s in sources]

        return jsonify({
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "claims": claims,
            "sources": pub_sources,
            "layer1": {"prediction": l1_label, "confidence": round(l1_conf, 4), "chars": len(text)},
            "full_mode": full_mode,
            "total_checks": count_checks(),
            "steps": final_steps,
        })
    finally:
        set_running(False)

@app.route("/analyze", methods=["POST"])
def analyze():
    if not ANALYZE_LOCK.acquire(blocking=False):
        return jsonify(error="Another verification is still running. Please wait a moment."), 409
    try:
        return _analyze()
    finally:
        ANALYZE_LOCK.release()

if __name__ == "__main__":
    print("[FactzChecker] ready on http://127.0.0.1:8000", flush=True)
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
