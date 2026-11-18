"""issues.py の定義から GitHub にラベル・マイルストーン・Epic・サブイシューを作成する。

usage: python scripts/github/create_issues.py OWNER/REPO [--dry-run]

一度だけ実行する想定（再実行すると Issue が重複する。ラベルは --force で上書き）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issues import EPICS, LABELS

REPO = sys.argv[1]
DRY = "--dry-run" in sys.argv


def gh(*args: str) -> str:
    if DRY:
        print("gh", *args[:4], "...")
        return "https://github.com/x/y/issues/0"
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


def issue_id(num: str) -> str:
    """サブイシュー API が要求する整数 id（Issue 番号とは別物）。"""
    if DRY:
        return "0"
    return gh("api", f"repos/{REPO}/issues/{num}", "--jq", ".id")


def main() -> None:
    for name, (color, desc) in LABELS.items():
        gh(
            "label",
            "create",
            name,
            "--repo",
            REPO,
            "--color",
            color,
            "--description",
            desc,
            "--force",
        )

    existing: set[str] = set()
    if not DRY:
        milestones = json.loads(gh("api", f"repos/{REPO}/milestones?state=all&per_page=100"))
        existing = {m["title"] for m in milestones}

    total = 0
    for epic in EPICS:
        ms = epic["milestone"]
        if ms not in existing:
            gh(
                "api",
                f"repos/{REPO}/milestones",
                "-f",
                f"title={ms}",
                "-f",
                f"description={epic['desc']}",
            )

        created: list[tuple[str, str]] = []
        for title, labels, body in epic["tasks"]:
            label_args = [arg for lb in labels for arg in ("--label", lb)]
            url = gh("issue", "create", "--repo", REPO, "--title", title, "--body", body,
                     "--milestone", ms, *label_args)  # fmt: skip
            num = url.rsplit("/", 1)[-1]
            created.append((num, title))
            total += 1
            print(f"  #{num} {title}")

        task_lines = "\n".join(f"- #{num} {title}" for num, title in created)
        ebody = f"## 目的\n{epic['desc']}\n\n## サブイシュー\n{task_lines}"
        eurl = gh("issue", "create", "--repo", REPO, "--title", epic["title"], "--body", ebody,
                  "--milestone", ms, "--label", "epic")  # fmt: skip
        enum = eurl.rsplit("/", 1)[-1]
        print(f"#{enum} {epic['title']}")
        for num, _ in created:
            gh("api", "-X", "POST", f"repos/{REPO}/issues/{enum}/sub_issues",
               "-F", f"sub_issue_id={issue_id(num)}")  # fmt: skip

    print(f"done: {len(EPICS)} epics, {total} tasks")


if __name__ == "__main__":
    main()
