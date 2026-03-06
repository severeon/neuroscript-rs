#!/usr/bin/env bash
# analyze-pr-comments.sh — Fetch PR comment data for analysis
#
# Usage:
#   ./scripts/analyze-pr-comments.sh <PR_URL_or_NUMBER> [<PR_URL_or_NUMBER> ...]
#   ./scripts/analyze-pr-comments.sh --recent 15
#
# This script fetches PR metadata and comments from GitHub, then formats them
# into readable thread files for analysis. The actual analysis is done by
# Claude Code agents (see docs/reviews/ for output).
#
# Outputs: /tmp/pr-audit/pr_<N>_thread.txt for each PR

set -euo pipefail

REPO="severeon/neuroscript-rs"

# ── Collect PR numbers ────────────────────────────────────────────────
PR_NUMBERS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recent)
      COUNT="${2:-15}"
      shift 2 || { shift; COUNT=15; }
      while IFS= read -r num; do
        PR_NUMBERS+=("$num")
      done < <(gh pr list --repo "$REPO" --limit "$COUNT" --state merged \
               --json number --jq '.[].number')
      ;;
    *)
      # Accept full URL or bare number
      num="${1##*/}"
      PR_NUMBERS+=("$num")
      shift
      ;;
  esac
done

if [[ ${#PR_NUMBERS[@]} -eq 0 ]]; then
  echo "Usage: $0 <PR_URL|NUMBER> ... | --recent N" >&2
  exit 1
fi

echo "Fetching data for ${#PR_NUMBERS[@]} PRs: ${PR_NUMBERS[*]}"

# ── Fetch and format PR data using Python for safe JSON handling ──────
python3 << PYEOF
import subprocess, json, os, sys

prs = [${PR_NUMBERS[*]/#/}]
repo = "${REPO}"
outdir = "/tmp/pr-audit"
os.makedirs(outdir, exist_ok=True)

for pr in prs:
    # Fetch comments
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/issues/{pr}/comments", "--paginate"],
        capture_output=True, text=True
    )
    comments = json.loads(result.stdout) if result.stdout.strip() else []

    # Fetch PR metadata
    result2 = subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{pr}"],
        capture_output=True, text=True
    )
    meta = json.loads(result2.stdout) if result2.stdout.strip() else {}

    # Fetch inline review comments
    result3 = subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{pr}/comments", "--paginate"],
        capture_output=True, text=True
    )
    review_comments = json.loads(result3.stdout) if result3.stdout.strip() else []

    title = meta.get("title", "unknown")
    body = (meta.get("body") or "")[:2000]

    with open(f"{outdir}/pr_{pr}_thread.txt", "w") as f:
        f.write(f"## PR #{pr}: {title}\n\n")
        f.write(f"### PR Description (excerpt)\n{body}\n\n")
        f.write("### Comments\n\n")
        for c in comments:
            user = c.get("user", {}).get("login", "unknown")
            cbody = c.get("body", "")
            created = c.get("created_at", "")
            f.write(f"---\n**{user}** ({created}):\n{cbody}\n\n")
        if review_comments:
            f.write("### Inline Review Comments\n\n")
            for c in review_comments:
                user = c.get("user", {}).get("login", "unknown")
                cbody = c.get("body", "")
                path = c.get("path", "unknown")
                line = c.get("line", "?")
                f.write(f"---\n**{user}** on \`{path}:{line}\`:\n{cbody}\n\n")

    print(f"  PR #{pr} ({len(comments)} comments, {len(review_comments)} inline): {title}")

print(f"\nDone! Thread files written to {outdir}/")
print("To analyze, use Claude Code with the prompt:")
print('  "Analyze PR comment threads in /tmp/pr-audit/ for unresolved issues"')
PYEOF
