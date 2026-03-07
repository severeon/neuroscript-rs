#!/bin/bash
# Wait for claude-review (and optionally CI) checks to complete on a PR.
# Usage: ./scripts/wait-for-review.sh <PR_NUMBER> [--all]
#   --all: wait for all checks, not just claude-review

set -uo pipefail

PR="${1:?Usage: wait-for-review.sh <PR_NUMBER> [--all]}"
WAIT_ALL="${2:-}"
POLL_INTERVAL=30

echo "Watching PR #${PR}..."

while true; do
    # gh pr checks exits non-zero when checks are pending — that's expected
    if [[ "$WAIT_ALL" == "--all" ]]; then
        CHECKS=$(gh pr checks "$PR" 2>&1) || true
    else
        CHECKS=$(gh pr checks "$PR" 2>&1 | grep -i "claude-review") || true
    fi

    if [[ -z "$CHECKS" ]]; then
        echo "  No checks found yet, waiting..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # gh pr checks uses: pass, fail, pending
    if echo "$CHECKS" | grep -qiE "\bpending\b"; then
        PENDING=$(echo "$CHECKS" | grep -ciE "\bpending\b" || true)
        echo "  ${PENDING} check(s) still running..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # All checks complete — report results
    echo ""
    echo "All checks complete for PR #${PR}:"
    echo "$CHECKS"
    echo ""

    # Check for failures
    if echo "$CHECKS" | grep -qiE "\bfail\b"; then
        RESULT="FAILED"
        echo "Some checks failed."
    else
        RESULT="PASSED"
        echo "All checks passed."
    fi

    # macOS notification
    if command -v osascript &>/dev/null; then
        osascript -e "display notification \"PR #${PR}: ${RESULT}\" with title \"Claude Review\" sound name \"Glass\""
    fi

    # Exit with appropriate code
    if [[ "$RESULT" == "FAILED" ]]; then
        exit 1
    fi
    exit 0
done
