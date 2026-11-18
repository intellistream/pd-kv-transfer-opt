"""Fetch one pinned, public trajectory release; never execute its task content."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

REPO = "openguardrails/terminal-bench-2.1-deepseek-v4-flash-trajectories"
REVISION = "3f5b93ff1c5ea84b2c94cf479d0c942227191b4a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy")
    parser.add_argument("--spec", type=Path, help="Fetch only the source files listed in a workload spec.")
    args = parser.parse_args()
    handler = urllib.request.ProxyHandler(
        {"https": args.proxy, "http": args.proxy} if args.proxy else {}
    )
    opener = urllib.request.build_opener(handler)
    api = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
    with opener.open(api, timeout=40) as response:
        metadata_bytes = response.read()
    metadata = json.loads(metadata_bytes)
    if (
        metadata.get("sha") != REVISION
        or metadata.get("private")
        or metadata.get("gated")
    ):
        raise RuntimeError("Only the pinned public, ungated release is authorized")
    files = [
        row
        for row in metadata["siblings"]
        if row["rfilename"] == "README.md"
        or (
            row["rfilename"].startswith("trajectories/terminus2/")
            and row["rfilename"].endswith(
                ("/agent/trajectory.json", "/config.json", "/result.json")
            )
        )
    ]
    if args.spec:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        if (spec.get("source_repo"), spec.get("source_revision")) != (REPO, REVISION):
            raise ValueError("The workload spec refers to a different source revision.")
        wanted = {entry["path"] for entry in spec["source_files"]}
        files = [row for row in files if row["rfilename"] in wanted]
        if {row["rfilename"] for row in files} != wanted:
            raise ValueError("Some pinned source files are missing upstream.")
    total = sum(row["size"] for row in files)
    if not files or total > 512 * 1024**2:
        raise RuntimeError(f"Unexpected download size or empty file selection: {total}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "source-api.json").write_bytes(metadata_bytes)
    print(
        json.dumps(
            {
                "repo": REPO,
                "revision": REVISION,
                "selected_files": len(files),
                "expected_bytes": total,
            }
        ),
        flush=True,
    )

    def download(row):
        relative = PurePosixPath(row["rfilename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe source filename")
        target = args.output.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != row["size"]:
            url = (
                f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
                + urllib.parse.quote(str(relative))
            )
            for attempt in range(3):
                try:
                    with opener.open(url, timeout=40) as response:
                        payload = response.read(row["size"] + 1)
                    if len(payload) != row["size"]:
                        raise RuntimeError("Source size does not match pinned metadata")
                    target.write_bytes(payload)
                    break
                except urllib.error.HTTPError as error:
                    if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                        raise
                    time.sleep(2**attempt)
                except (TimeoutError, urllib.error.URLError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
        payload = target.read_bytes()
        # Git blob IDs validate ordinary Git files; LFS files carry their own SHA256.
        lfs_sha = row.get("lfs", {}).get("sha256")
        if lfs_sha:
            valid = hashlib.sha256(payload).hexdigest() == lfs_sha
        else:
            git_blob = b"blob " + str(len(payload)).encode() + b"\0" + payload
            valid = hashlib.sha1(git_blob).hexdigest() == row["blobId"]
        if not valid:
            raise RuntimeError(f"Source checksum failed: {relative}")
        return {
            "path": str(relative),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "upstream_blob_verified": True,
        }

    receipts = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for receipt in executor.map(download, files):
            receipts.append(receipt)
            if len(receipts) % 25 == 0:
                print(json.dumps({"verified_files": len(receipts)}), flush=True)
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "source_url": api,
        "downloaded_bytes": sum(row["bytes"] for row in receipts),
        "files": receipts,
        "purpose": "Serving workload trace replay only; not this model's agent task accuracy",
        "task_content_executed": False,
    }
    (args.output / "download-receipt.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"complete": True, "verified_files": len(receipts)}), flush=True)


if __name__ == "__main__":
    main()
