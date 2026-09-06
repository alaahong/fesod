#!/usr/bin/env bash
# Push the RC tag and release branch to apache/fesod (and best-effort to fork).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

: "${GIT_PUSH_TOKEN:?missing GIT_PUSH_TOKEN (GITHUB_TOKEN)}"

# GHES/API token as basic-auth; the token value is auto-masked by GitHub in logs.
APACHE_REPO="https://x-access-token:${GIT_PUSH_TOKEN}@github.com/apache/fesod.git"
ORIGIN_REPO="${ORIGIN_REPO:-$(git config --get remote.origin.url || echo '')}"

echo "== tag ${GIT_TAG} at ${PUSH_TARGET} =="
git tag "${GIT_TAG}" "${PUSH_TARGET}"

echo "== push tag =="
git push "${APACHE_REPO}" "${GIT_TAG}"

# create + push the RC branch from the same commit
echo "== create + push branch ${RC_BRANCH} at ${PUSH_TARGET} =="
git branch -f "${RC_BRANCH}" "${PUSH_TARGET}"
git push "${APACHE_REPO}" "${RC_BRANCH}"

# best-effort sync to the fork; must not fail the job
if [ -n "${ORIGIN_REPO}" ] && [ "${ORIGIN_REPO}" != "https://github.com/apache/fesod.git" ] \
   && [ "${ORIGIN_REPO}" != "https://github.com/apache/fesod" ] \
   && [ "${ORIGIN_REPO}" != "git@github.com:apache/fesod.git" ]; then
  git push origin "${RC_BRANCH}" 2>/dev/null || echo "  (fork sync skipped)"
fi

echo "Pushed tag ${GIT_TAG} and branch ${RC_BRANCH} to apache/fesod"