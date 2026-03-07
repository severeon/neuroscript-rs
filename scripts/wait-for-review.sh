#!/usr/bin/env bash
# Wait for claude-review check to complete on a PR.
#
# Usage: ./scripts/wait-for-review.sh <PR_NUMBER>
#
# Polls gh pr checks every 30s until the claude-review check
# completes, then prints the result. Optionally shows a macOS
# desktop notification.
#
# Environment:
#   POLL_INTERVAL  - seconds between checks (default: 30)
#   NO_NOTIFY      - set to 1 to skip desktop notification

set -euo pipefail

PR="${1:-}"
if [ -z "$PR" ]; then
    echo "Usage: $0 <PR_NUMBER>"
    exit 1
fi

INTERVAL="${POLL_INTERVAL:-30}"

echo "Waiting for claude-review on PR #${PR}..."

while true; do
    # Get check status — look for claude-review specifically
    RESULT=$(gh pr checks "$PR" 2>/dev/null | grep -i "claude-review" || true)

    if [ -n "$RESULT" ]; then
        STATUS=$(echo "$RESULT" | awk '{print $2}')
        case "$STATUS" in
            pass)
                echo "claude-review PASSED on PR #${PR}"
                if [ "${NO_NOTIFY:-}" != "1" ] && command -v osascript &>/dev/null; then
                    osascript -e "display notification \"claude-review passed\" with title \"PR #${PR}\" sound name \"Glass\""
                fi
                exit 0
                ;;
            fail)
                echo "claude-review FAILED on PR #${PR}"
                if [ "${NO_NOTIFY:-}" != "1" ] && command -v osascript &>/dev/null; then
                    osascript -e "display notification \"claude-review failed\" with title \"PR #${PR}\" sound name \"Basso\""
                fi
                exit 1
                ;;
            *)
                # Still pending/in_progress
                ;;
        esac
    fi

    sleep "$INTERVAL"
done
