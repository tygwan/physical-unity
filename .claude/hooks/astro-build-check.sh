#!/bin/bash
# Hook: Pre-build lint and type check for Astro projects
# Event: PreToolUse (Bash - when running build commands)
# Purpose: Validate Astro project before build to catch errors early

TOOL_NAME="$1"
ASTRO_ROOT=""

# Find astro.config.mjs to determine project root
find_astro_root() {
  local dir="$PWD"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/astro.config.mjs" ] || [ -f "$dir/astro.config.ts" ]; then
      ASTRO_ROOT="$dir"
      return 0
    fi
    dir=$(dirname "$dir")
  done
  return 1
}

# Only run for build-related commands
case "$TOOL_NAME" in
  "build"|"deploy"|"preview")
    if find_astro_root; then
      echo "[astro-build-check] Found Astro project at: $ASTRO_ROOT"

      # Check if node_modules exists
      if [ ! -d "$ASTRO_ROOT/node_modules" ]; then
        echo "[astro-build-check] WARNING: node_modules not found. Run 'npm install' first."
        exit 1
      fi

      # Run astro check for type validation
      cd "$ASTRO_ROOT"
      if command -v npx &> /dev/null; then
        echo "[astro-build-check] Running astro check..."
        npx astro check 2>&1
        CHECK_EXIT=$?
        if [ $CHECK_EXIT -ne 0 ]; then
          echo "[astro-build-check] FAILED: astro check found errors. Fix before building."
          exit 1
        fi
        echo "[astro-build-check] PASSED: No type errors found."
      fi
    fi
    ;;
esac

exit 0
