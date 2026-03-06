#!/usr/bin/env bash
# Hook for fix-queue agent: writes tool activity to status file
# Called by Claude's PostToolUse hook with $1=tool_name, $2=tool_input

STATUS_FILE="/tmp/fix-queue-status.txt"
TOOL_NAME="${1:-unknown}"
TOOL_INPUT="${2:-}"

# Extract a short summary based on tool type
case "$TOOL_NAME" in
  Read)
    file=$(echo "$TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path','?'))" 2>/dev/null)
    msg="Reading ${file##*/}"
    ;;
  Edit)
    file=$(echo "$TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path','?'))" 2>/dev/null)
    msg="Editing ${file##*/}"
    ;;
  Write)
    file=$(echo "$TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path','?'))" 2>/dev/null)
    msg="Writing ${file##*/}"
    ;;
  Bash)
    cmd=$(echo "$TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('command','?')[:60])" 2>/dev/null)
    msg="Running: $cmd"
    ;;
  Grep)
    pattern=$(echo "$TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pattern','?'))" 2>/dev/null)
    msg="Searching: $pattern"
    ;;
  Glob)
    pattern=$(echo "$TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pattern','?'))" 2>/dev/null)
    msg="Finding: $pattern"
    ;;
  *)
    msg="$TOOL_NAME"
    ;;
esac

# Append to status file (monitor reads last line)
echo "$(date '+%H:%M:%S') | $msg" >> "$STATUS_FILE"
