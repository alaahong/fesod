#!/usr/bin/env bash
# Generate the [VOTE] email draft from the artifacts actually produced this run.
# Safe: reads files under ${ARTIFACT_DIR}, git, gpg and the clock only; emits no secrets.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

OUT="${ARTIFACT_DIR}/VOTE-${GIT_TAG}.txt"

# test summary (produced by gate-verify.sh, guaranteed on the happy path)
SUM="${ARTIFACT_DIR}/tests-summary.txt"
if [ ! -f "${SUM}" ]; then
  echo "::error::tests-summary.txt missing; run gate-verify.sh (full tests) first" >&2
  exit 1
fi
TESTS_TOTAL="$(grep -E '^Tests run:' "${SUM}" | grep -oE '[0-9]+' | sed -n '1p')"
TESTS_FAIL="$(grep -E '^Tests run:' "${SUM}" | grep -oE '[0-9]+' | sed -n '2p')"
JDK_LINE="$(grep -E '^JDK:' "${SUM}" | sed -E 's/^JDK:[[:space:]]*//')"
if [ -z "${TESTS_TOTAL:-}" ]; then
  echo "::error::could not parse test count from ${SUM}" >&2
  exit 1
fi
JDK_VER="$(printf '%s' "${JDK_LINE}" | grep -oE '[0-9]+(\.[0-9]+)*' | head -n1 || true)"
MAJOR="$(printf '%s' "${JDK_VER}" | cut -d. -f1)"
[ "${MAJOR}" = "1" ] && MAJOR="$(printf '%s' "${JDK_VER}" | cut -d. -f2)"
CLASS_MAJOR="$((MAJOR + 44))"

# git tag commit + one-line subject
GIT_MSG="$(git log -1 --format='%h %s' "${COMMIT_SHA}")"
COMMIT_SHORT="${GIT_MSG%% *}"
SUBJECT="${GIT_MSG#* }"

# SHA-512 of the source tarball (format: "<hash>  <name>" from sha512sum)
TAR_SHA="$(awk '{print $1}' "${ARTIFACT_DIR}/${PKG_TARBALL}.sha512" | head -n1)"

# optional source-release zip hash, if one was produced this run
ZIP_BLOCK=""
ZIP_SHA_FILE="$(find "${ARTIFACT_DIR}" -maxdepth 1 -name '*.zip.sha512' -print -quit || true)"
if [ -n "${ZIP_SHA_FILE:-}" ]; then
  ZIP_NAME="$(basename "${ZIP_SHA_FILE}" .sha512)"
  ZIP_SHA="$(awk '{print $1}' "${ZIP_SHA_FILE}" | head -n1)"
  ZIP_BLOCK="    ${ZIP_NAME}
      ${ZIP_SHA}
"
fi

# Nexus staging repository id (written by publish-nexus.sh)
STAGING="$(cat "${ARTIFACT_DIR}/staging-id.txt" 2>/dev/null || echo 'orgapachefesod-XXXX')"

# signing key fingerprint
FPR="$(gpg --batch --with-colons --list-secret-keys 2>/dev/null | awk -F: '/^fpr/{print $10; exit}')"
FPR="${FPR:-UNKNOWN}"

# vote close time: at least 72 hours from now (runner clock is UTC)
DEADLINE_UTC="$(date -u -d '+72 hours' '+%Y-%m-%d %H:%M UTC')"

EMAIL_LOGIN="${EMAIL_LOGIN:-$(git config user.name || echo '[Your Name]')}"

{
  printf 'Subject: [VOTE] Release Apache Fesod %s (RC%s)\n\n' "${REVISION}" "${RC}"
  printf 'Hello Fesod Community,\n\n'
  printf 'This is a call for a vote to release Apache Fesod %s as\n' "${REVISION}"
  printf 'Release Candidate %s (RC%s).\n\n' "${RC}" "${RC}"
  printf 'The source release candidate artifacts are available at:\n'
  printf '    https://dist.apache.org/repos/dist/dev/incubator/fesod/%s/\n' "${SVN_DIR_NAME}"
  printf '    %s\n' "${PKG_TARBALL}"
  printf '\nThe git tag to be voted on:\n'
  printf '    https://github.com/apache/fesod/tree/%s\n' "${GIT_TAG}"
  printf '    (commit %s "%s")\n' "${COMMIT_SHORT}" "${SUBJECT}"
  printf '\nRelease checksums (SHA-512):\n'
  printf '    %s\n      %s\n' "${PKG_TARBALL}" "${TAR_SHA}"
  [ -n "${ZIP_BLOCK}" ] && printf '%s\n' "${ZIP_BLOCK%$'\n'}"
  printf '\nThe artifacts were compiled with JDK %s (class file major version %s) and all\n' "${MAJOR}" "${CLASS_MAJOR}"
  printf '%s tests pass. The Maven staging repository (for testing only, not the\n' "${TESTS_TOTAL}"
  printf 'object of this vote) is:\n'
  printf '    https://repository.apache.org/content/repositories/%s/\n' "${STAGING}"
  printf '\nThe signing key used is:\n'
  printf '    https://downloads.apache.org/incubator/fesod/KEYS\n'
  printf '    fingerprint %s\n\n' "${FPR}"
  printf 'Please vote on releasing these source artifacts as Apache Fesod %s.\n' "${REVISION}"
  printf 'The vote will be open for at least 72 hours.\n\n'
  printf '    [ ] +1  approve the release\n'
  printf '    [ ]  0  abstain\n'
  printf '    [ ] -1  disapprove (with reason)\n\n'
  printf 'Vote members should verify the release following the checklist at\n'
  printf 'https://github.com/apache/fesod/blob/main/docs/release/verification.md\n\n'
  printf 'This vote requires at least 3 binding +1 votes from PMC members. A minimum\n'
  printf 'of 72 hours will be given.\n\n'
  printf 'This vote will be kept open until %s (at least 72 hours).\n\n' "${DEADLINE_UTC}"
  printf 'Thanks,\n%s\n' "${EMAIL_LOGIN}"
} > "${OUT}"

echo "VOTE email draft written to ${OUT}"
awk 'NR<=6{print} {c++} END{print "  [... " c-6 " lines more]"}' "${OUT}"