#!/usr/bin/env bash
# Watch for CI/review notifications via ntfy.sh
#
# Usage:
#   ./scripts/watch-review.sh              # subscribe and show notifications
#   ./scripts/watch-review.sh --once       # wait for one notification then exit
#   NTFY_TOPIC=my-topic ./scripts/watch-review.sh
#
# Prerequisites:
#   - curl (for ntfy.sh subscription)
#   - terminal-notifier (macOS): brew install terminal-notifier
#   - The NTFY_TOPIC secret must be set in your GitHub repo settings
#
# Setup:
#   1. Pick a unique topic name (e.g., neuroscript-ci-<random>)
#   2. Add it as a GitHub Actions secret: NTFY_TOPIC
#   3. Export it locally: export NTFY_TOPIC=neuroscript-ci-<random>
#   4. Run this script in a terminal tab

set -euo pipefail

TOPIC="${NTFY_TOPIC:-}"
ONCE="${1:-}"

if [ -z "$TOPIC" ]; then
    echo "Error: NTFY_TOPIC environment variable not set."
    echo ""
    echo "Setup:"
    echo "  1. Pick a unique topic: export NTFY_TOPIC=neuroscript-ci-\$(openssl rand -hex 8)"
    echo "  2. Add as GitHub secret: gh secret set NTFY_TOPIC --body \"\$NTFY_TOPIC\""
    echo "  3. Run this script again"
    exit 1
fi

echo "Listening for CI/review notifications on ntfy.sh/${TOPIC}..."
echo "Press Ctrl+C to stop."
echo ""

notify() {
    local title="$1"
    local message="$2"
    local url="${3:-}"

    # macOS notification
    if command -v terminal-notifier &>/dev/null; then
        if [ -n "$url" ]; then
            terminal-notifier -title "$title" -message "$message" -open "$url" -sound default
        else
            terminal-notifier -title "$title" -message "$message" -sound default
        fi
    # Linux notification
    elif command -v notify-send &>/dev/null; then
        notify-send "$title" "$message"
    else
        echo "NOTIFICATION: $title - $message"
    fi
}

# Subscribe to ntfy.sh using server-sent events (SSE)
# The /json endpoint gives us structured data we can parse
if [ "$ONCE" = "--once" ]; then
    # Wait for a single notification then exit
    curl -s "https://ntfy.sh/${TOPIC}/json" | while read -r line; do
        if echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('title',''))" 2>/dev/null | grep -q .; then
            TITLE=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('title','Notification'))")
            MSG=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',''))")
            URL=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('click',''))")
            echo "[$TITLE] $MSG"
            notify "$TITLE" "$MSG" "$URL"
            break
        fi
    done
else
    # Continuous subscription
    curl -s "https://ntfy.sh/${TOPIC}/json" | while read -r line; do
        TITLE=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('title',''))" 2>/dev/null) || continue
        if [ -z "$TITLE" ]; then continue; fi
        MSG=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',''))")
        URL=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('click',''))")
        echo "[$(date '+%H:%M:%S')] $TITLE: $MSG"
        notify "$TITLE" "$MSG" "$URL"
    done
fi
