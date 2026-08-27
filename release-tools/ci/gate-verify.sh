#!/usr/bin/env bash
# Verification gates. Any failure aborts the job here, so nothing gets published.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

echo "== [gate 1/4] assert pom revision == ${REVISION} =="
if ! grep -q "<revision>${REVISION}</revision>" pom.xml; then
  echo "::error::pom.xml <revision> does not match REVISION=${REVISION} on base-branch ${BASE_BRANCH}" >&2
  exit 1
fi

echo "== [gate 2/4] Apache RAT license check =="
./mvnw org.apache.rat:apache-rat-plugin:check

echo "== [gate 3/5] compile from source (skip tests) =="
./mvnw clean package -DskipTests

echo "== [gate 4/5] license / notice / disclaimer present =="
for f in LICENSE NOTICE DISCLAIMER; do
  if [ ! -f "${f}" ]; then
    echo "::error::missing required file ${f}" >&2
    exit 1
  fi
  echo "  OK ${f} ($(wc -c < "${f}") bytes)"
done

echo "== [gate 5/5] JAR NOTICE copyright range & legal org name =="
NOTICE_FILE="fesod-common/target/maven-shared-archive-resources/META-INF/NOTICE"
if [ ! -f "${NOTICE_FILE}" ]; then
  echo "::error::generated NOTICE not found at ${NOTICE_FILE}" >&2
  exit 1
fi
grep -q 'The Apache Software Foundation' "${NOTICE_FILE}" \
  || { echo "::error::NOTICE legal name is not 'The Apache Software Foundation'" >&2; exit 1; }
COPY="$(grep -oE 'Copyright [0-9]{4}(-[0-9]{4})?' "${NOTICE_FILE}" | head -n1 || true)"
if [ -n "${COPY}" ]; then
  RANGE="$(printf '%s' "${COPY}" | grep -oE '[0-9]{4}-[0-9]{4}$' || true)"
  if [ -z "${RANGE}" ]; then
    echo "  OK NOTICE: ${COPY} (single year)"
  else
    START="$(printf '%s' "${RANGE}" | cut -d- -f1)"
    END="$(printf '%s' "${RANGE}" | cut -d- -f2)"
    if [ "${START}" -le "${END}" ]; then
      echo "  OK NOTICE: ${COPY}"
    else
      echo "::error::NOTICE copyright range is inverted: ${COPY}" >&2
      exit 1
    fi
  fi
else
  echo "::error::no copyright line found in ${NOTICE_FILE}" >&2
  exit 1
fi

echo "All verification gates PASSED"