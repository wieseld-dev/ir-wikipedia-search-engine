import argparse
import json
import os
import shutil
import subprocess
import threading
import time

import gcsfs
import pyarrow.parquet as pq
from google.oauth2.credentials import Credentials


def load_query_ids(path):
    with open(path, "rt", encoding="utf-8") as f:
        queries = json.load(f)
    ids = set()
    for doc_ids in queries.values():
        for wid in doc_ids:
            try:
                ids.add(int(wid))
            except (TypeError, ValueError):
                continue
    return ids


def find_gcloud_cmd():
    cmd = shutil.which("gcloud")
    if cmd:
        return cmd
    candidate = os.path.expandvars(r"%LOCALAPPDATA%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError("gcloud not found in PATH or default install location")


def get_gcsfs_with_gcloud_token():
    gcloud = find_gcloud_cmd()
    access_token = subprocess.check_output([gcloud, "auth", "print-access-token"]).decode().strip()
    creds = Credentials(token=access_token)
    return gcsfs.GCSFileSystem(token=creds)


def normalize_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description="Extract articles referenced by queries_train.json from GCS parquet files.")
    parser.add_argument("--queries", default="queries_train.json", help="Path to queries JSON file")
    parser.add_argument("--bucket", default="ir-assignment3-daniel", help="GCS bucket name")
    parser.add_argument("--out", default="subset_articles.jsonl", help="Output JSONL file path")
    parser.add_argument("--state", default="subset_articles.state.json", help="State file for resume support")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output/state file")
    parser.add_argument("--heartbeat", type=int, default=10, help="Seconds between progress updates")
    args = parser.parse_args()

    query_ids = load_query_ids(args.queries)
    if not query_ids:
        raise RuntimeError("No query IDs found in queries JSON")

    fs = get_gcsfs_with_gcloud_token()
    files = sorted(fs.glob(f"{args.bucket}/*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in gs://{args.bucket}")

    found_ids = set()
    last_file = None
    if args.resume:
        if os.path.exists(args.out):
            with open(args.out, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    try:
                        found_ids.add(int(row.get("id")))
                    except (TypeError, ValueError):
                        continue
        if os.path.exists(args.state):
            try:
                with open(args.state, "rt", encoding="utf-8") as f:
                    state = json.load(f)
                last_file = state.get("last_file")
            except Exception:
                last_file = None

    remaining = set(query_ids) - found_ids
    written = 0

    start_idx = 0
    if last_file and last_file in files:
        start_idx = files.index(last_file) + 1

    progress = {
        "start_time": time.time(),
        "files_done": 0,
        "files_total": len(files) - start_idx,
        "last_file": None,
    }
    progress_lock = threading.Lock()
    stop_event = threading.Event()

    def heartbeat():
        while not stop_event.is_set():
            time.sleep(max(args.heartbeat, 1))
            with progress_lock:
                elapsed = time.time() - progress["start_time"]
                files_done = progress["files_done"]
                files_total = progress["files_total"]
                last_file_local = progress["last_file"]
                avg = elapsed / files_done if files_done else 0.0
                remaining_files = max(files_total - files_done, 0)
                eta = remaining_files * avg
            print(
                f"[heartbeat] elapsed={elapsed:.1f}s files={files_done}/{files_total} "
                f"avg_per_file={avg:.1f}s eta~{eta:.1f}s last={last_file_local}"
            )

    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()

    mode = "at" if args.resume else "wt"
    with open(args.out, mode, encoding="utf-8") as f:
        for idx, file_path in enumerate(files[start_idx:], start=start_idx + 1):
            if not remaining:
                break
            print(f"Scanning {idx}/{len(files)}: {file_path} (remaining IDs: {len(remaining)})")
            try:
                t0 = time.time()
                table = pq.read_table(
                    file_path,
                    filters=[("id", "in", sorted(remaining))],
                    filesystem=fs,
                    use_threads=True,
                )
            except Exception as exc:
                print(f"Skipping {file_path} due to error: {exc}")
                with progress_lock:
                    progress["files_done"] += 1
                    progress["last_file"] = file_path
                continue

            if table.num_rows == 0:
                with progress_lock:
                    progress["files_done"] += 1
                    progress["last_file"] = file_path
                continue

            rows = table.to_pylist()
            for row in rows:
                row = normalize_value(row)
                if "id" in row:
                    try:
                        rid = int(row["id"])
                        found_ids.add(rid)
                        if rid in remaining:
                            remaining.remove(rid)
                    except (TypeError, ValueError):
                        pass
                f.write(json.dumps(row, ensure_ascii=True))
                f.write("\n")
                written += 1
            with open(args.state, "wt", encoding="utf-8") as state_f:
                json.dump({"last_file": file_path}, state_f)
            t1 = time.time()
            with progress_lock:
                progress["files_done"] += 1
                progress["last_file"] = file_path
            print(
                f"Finished {file_path} in {t1 - t0:.1f}s; "
                f"found={len(found_ids)} remaining={len(remaining)} written={written}"
            )

    missing = sorted(query_ids - found_ids)
    print(f"Wrote {written} records to {args.out}")
    print(f"Found {len(found_ids)} / {len(query_ids)} IDs")
    if missing:
        print(f"Missing {len(missing)} IDs (showing up to 20): {missing[:20]}")
    stop_event.set()


if __name__ == "__main__":
    main()
