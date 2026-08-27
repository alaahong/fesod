#!/usr/bin/env bash
# Deploy signed artifacts to a Nexus staging repository (does NOT close/release it).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

: "${GPG_PASSFILE:?missing gpg passphrase setup}"

echo "== mvn clean deploy (signed, skip tests) =="
LOG="${ARTIFACT_DIR}/staging-deploy.log"
# Signing key/passphrase via gpg-plugin properties (path only, no secret on CLI).
set +e
./mvnw clean deploy -DskipTests -Papache-release \
  -Dgpg.passphraseFile="${GPG_PASSFILE}" \
  -Dgpg.homedir="${GNUPGHOME}" \
  > "${LOG}" 2>&1
RC_MVN=$?
set -e

if [ "${RC_MVN}" -ne 0 ]; then
  echo "::error::Nexus deploy FAILED (exit ${RC_MVN}); tail of log:" >&2
  tail -n 60 "${LOG}" >&2
  exit 1
fi

# extract the created staging repository id, e.g. orgapachefesod-1034
STAGING_ID="$(grep -oE 'orgapachefesod-[0-9]+' "${LOG}" | tail -n1 || true)"
printf '%s\n' "${STAGING_ID:-UNKNOWN}" > "${ARTIFACT_DIR}/staging-id.txt"

# no staging id => deploy produced no upload (e.g. signing/failure swallowed); do not treat as done
if [ "${STAGING_ID:-UNKNOWN}" = "UNKNOWN" ]; then
  echo "::error::no Nexus staging repository id found in deploy log" >&2
  tail -n 40 "${LOG}" >&2
  exit 1
fi

echo "== deploy succeeded =="
echo "staging repository: ${STAGING_ID:-UNKNOWN}  (leave OPEN; close manually in Nexus UI)"
tail -n 12 "${LOG}"