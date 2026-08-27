#!/usr/bin/env bash
# Print a one-shot summary (always runs). Safe to run even on early failure.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

LINE="## RC Release ${REVISION} rc${RC}"
LINE+=$'\n- base-branch:       '"${BASE_BRANCH:-n/a}"
LINE+=$'\n- git tag:           '"https://github.com/apache/fesod/releases/tag/${GIT_TAG:-n/a}"
LINE+=$'\n- svn dev dist:      '"${SVN_RC_URL:-n/a}/"
STAGING="$(cat "${ARTIFACT_DIR}/staging-id.txt" 2>/dev/null || echo 'n/a (not published)')"
LINE+=$'\n- nexus staging:     '"${STAGING} (leave OPEN; close manually)"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  printf '%s\n' "${LINE}" >> "${GITHUB_STEP_SUMMARY}"
else
  printf '%s\n' "${LINE}"
fi

echo "Summary written (step summary). Remember: close the Nexus staging repo manually."