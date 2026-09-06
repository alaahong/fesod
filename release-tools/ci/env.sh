#!/usr/bin/env bash
# Shared environment & path derivation for the RC release pipeline.
# Source this from every script:  source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# Security invariants (see docs/release/rc-release-github-action.md):
#   - NO `set -x`; never echo secret values.
#   - Credentials live only in ${CREDS_DIR} (umask 077) and env vars.
set -euo pipefail

# ---- inputs (injected by the workflow, required) ----
: "${REVISION:?REVISION is required}"          # e.g. 2.1.0-incubating
: "${RC:?RC is required}"                      # e.g. 1
: "${COMMIT_SHA:?COMMIT_SHA is required}"      # exact source commit the RC is cut from
BASE_BRANCH="${BASE_BRANCH:-main}"

# ---- derived names ----
PROJECT_SHORT="fesod"
RC_BRANCH="release-${REVISION}-RC${RC}"
GIT_TAG="${REVISION}-rc${RC}"
PKG_BASE="apache-${PROJECT_SHORT}-${REVISION}-src"
PKG_TARBALL="${PKG_BASE}.tar.gz"
SVN_DIST_URL="https://dist.apache.org/repos/dist/dev/incubator/fesod"
SVN_DIR_NAME="${REVISION}-rc${RC}"
SVN_RC_URL="${SVN_DIST_URL}/${SVN_DIR_NAME}"

# ---- ephemeral dirs (persist for the whole job) ----
WORK_ROOT="${FC_WORK_ROOT:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}}/fesod-rc"
CREDS_DIR="${WORK_ROOT}/creds"
ARTIFACT_DIR="${WORK_ROOT}/artifacts"
export GNUPGHOME="${CREDS_DIR}/gnupg"
export GPG_PASSFILE="${CREDS_DIR}/gpg-passphrase"

# Naming collision guard: a fresh tag/dir with the same name must not be silently
# re-published.
PREV_TAG=""
if git cat-file -e "${GIT_TAG}^{commit}" 2>/dev/null; then
  PREV_TAG="${GIT_TAG}"
fi

# The exact commit (already checked out) from which tag/branch and source are cut.
PUSH_TARGET="${COMMIT_SHA}"

export REVISION RC BASE_BRANCH PROJECT_SHORT RC_BRANCH GIT_TAG
export PKG_TARBALL SVN_DIST_URL SVN_DIR_NAME SVN_RC_URL
export WORK_ROOT CREDS_DIR ARTIFACT_DIR GNUPGHOME GPG_PASSFILE PREV_TAG PUSH_TARGET COMMIT_SHA