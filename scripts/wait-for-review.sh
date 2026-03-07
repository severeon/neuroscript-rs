#!/usr/bin/env bash
# Wait for CI/review checks to complete on a PR.
#
# Usage:
#   ./scripts/wait-for-review.sh <PR_NUMBER>           # wait for all checks
#   ./scripts/wait-for-review.sh <PR_NUMBER> --review   # wait for claude-review only
#   ./scripts/wait-for-review.sh <PR_NUMBER> --ci       # wait for CI test job only
#
# Wraps `gh run watch` to block until the relevant workflow run
# completes, then shows a macOS desktop notification with the result.
#
# Environment:
#   NO_NOTIFY  - set to 1 to skip desktop notification

set -euo pipefail

PR="${1:-}"
FILTER="${2:-}"

if [ -z "$PR" ]; then
    echo "Usage: $0 <PR_NUMBER> [--review|--ci]"
    exit 1
fi

notify() {
    local title="$1" message="$2" sound="$3"
    if [ "${NO_NOTIFY:-}" != "1" ] && command -v osascript &>/dev/null; then
        osascript -e "display notification \"${message}\" with title \"${title}\" sound name \"${sound}\""
    fi
}

# Find the workflow run(s) for this PR's head branch
BRANCH=$(gh pr view "$PR" --json headRefName --jq '.headRefName')
if [ -z "$BRANCH" ]; then
    echo "Error: could not determine branch for PR #${PR}"
    exit 1
fi

watch_run() {
    local workflow="$1" label="$2"
    # Get the most recent run for this branch and workflow
    local run_id
    run_id=$(gh run list --branch "$BRANCH" --workflow "$workflow" --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)

    if [ -z "$run_id" ] || [ "$run_id" = "null" ]; then
        echo "No ${label} run found for branch ${BRANCH}. Waiting for it to start..."
        # Poll until a run appears
        while true; do
            sleep 10
            run_id=$(gh run list --branch "$BRANCH" --workflow "$workflow" --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)
            if [ -n "$run_id" ] && [ "$run_id" != "null" ]; then
                break
            fi
        done
    fi

    echo "Watching ${label} run ${run_id} on branch ${BRANCH}..."
    if gh run watch "$run_id" --exit-status 2>/dev/null; then
        echo "${label} PASSED on PR #${PR}"
        notify "PR #${PR}" "${label} passed" "Glass"
        return 0
    else
        echo "${label} FAILED on PR #${PR}"
        notify "PR #${PR}" "${label} failed" "Basso"
        return 1
    fi
}

case "$FILTER" in
    --review)
        watch_run "claude-code-review.yml" "claude-review"
        ;;
    --ci)
        watch_run "ci.yml" "CI"
        ;;
    *)
        # Watch both — review first (usually slower), then CI
        echo "Waiting for all checks on PR #${PR} (branch: ${BRANCH})..."
        FAILED=0
        watch_run "claude-code-review.yml" "claude-review" || FAILED=1
        watch_run "ci.yml" "CI" || FAILED=1
        exit $FAILED
        ;;
esac
