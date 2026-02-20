#!/bin/bash
# PreToolUse hook: Block dangerous commands
# Prevents destructive operations from being run by Claude Code agents.
#
# Input: JSON on stdin from Claude Code (PreToolUse event)
# Output: exit 0 = allow, exit 2 = block
#
# Blocked patterns:
#   rm -rf / rm -fr           (recursive force delete)
#   git push --force / -f     (force push)
#   curl|bash, wget|sh        (pipe-to-shell)
#   neuroscript publish w/o --dry-run
#   sudo                      (privilege escalation)
#   git reset --hard          (destructive reset)
#   git clean -f              (untracked file deletion)

set -o pipefail

INPUT=$(cat)

# Extract command from Bash tool input
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

# No command to check
[[ -z "$COMMAND" ]] && exit 0

# ── Blocklist checks ──────────────────────────────────────────────

# rm -rf / rm -fr (any flag ordering with r and f)
if echo "$COMMAND" | grep -qE '\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\b'; then
  echo "BLOCKED: recursive force delete (rm -rf) is not allowed"
  exit 2
fi

# git push --force / git push -f
if echo "$COMMAND" | grep -qE '\bgit\s+push\s+.*(-f|--force)\b'; then
  echo "BLOCKED: force push is not allowed"
  exit 2
fi

# curl|bash, curl|sh, wget|bash, wget|sh (pipe to shell)
if echo "$COMMAND" | grep -qE '\b(curl|wget)\b.*\|\s*(bash|sh)\b'; then
  echo "BLOCKED: piping downloads to shell is not allowed"
  exit 2
fi

# neuroscript publish without --dry-run
if echo "$COMMAND" | grep -qE '\bneuroscript\s+publish\b' && ! echo "$COMMAND" | grep -qE '\-\-dry-run'; then
  echo "BLOCKED: neuroscript publish requires --dry-run flag"
  exit 2
fi

# sudo
if echo "$COMMAND" | grep -qE '(^|\s|;|\|)sudo\b'; then
  echo "BLOCKED: sudo is not allowed"
  exit 2
fi

# git reset --hard
if echo "$COMMAND" | grep -qE '\bgit\s+reset\s+--hard\b'; then
  echo "BLOCKED: git reset --hard is not allowed"
  exit 2
fi

# git clean -f (any flags containing f)
if echo "$COMMAND" | grep -qE '\bgit\s+clean\s+-[a-zA-Z]*f'; then
  echo "BLOCKED: git clean -f is not allowed"
  exit 2
fi

# All checks passed
exit 0
