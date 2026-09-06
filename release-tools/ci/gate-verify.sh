#!/usr/bin/env bash
# Verification gates. Any failure aborts the job here, so nothing gets published.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

echo "== [gate 1/5] assert pom revision == ${REVISION} =="
if ! grep -q "<revision>${REVISION}</revision>" pom.xml; then
  echo "::error::pom.xml <revision> does not match REVISION=${REVISION} on commit ${COMMIT_SHA}" >&2
  exit 1
fi

echo "== [gate 2/5] Apache RAT license check =="
./mvnw org.apache.rat:apache-rat-plugin:check

echo "== [gate 3/5] full build + run the test suite =="
./mvnw clean package -Dmaven.test.skip=false -DskipTests=false

echo "== [gate 3b/5] collect test summary =="
mkdir -p "${ARTIFACT_DIR}"
TESTS_TOTAL=0
TESTS_FAIL=0
while IFS= read -r rep; do
  last="$(grep 'Tests run:' "${rep}" | tail -n1 || true)"
  [ -n "${last}" ] || continue
  t="$(printf '%s' "${last}" | sed -E 's/.*Tests run:[[:space:]]*([0-9]+).*/\1/')"
  f="$(printf '%s' "${last}" | sed -E 's/.*Failures:[[:space:]]*([0-9]+).*/\1/')"
  TESTS_TOTAL=$((TESTS_TOTAL + ${t:-0}))
  TESTS_FAIL=$((TESTS_FAIL + ${f:-0}))
done < <(find . -path '*/surefire-reports/*.txt' -type f)

JDK_VERSION="$(java -version 2>&1 | head -n1)"
printf 'Tests run: %d, Failures: %d\nJDK: %s\n' "${TESTS_TOTAL}" "${TESTS_FAIL}" "${JDK_VERSION}" > "${ARTIFACT_DIR}/tests-summary.txt"
echo "  ${TESTS_TOTAL} tests run, ${TESTS_FAIL} failures (${JDK_VERSION})"
if [ "${TESTS_TOTAL}" -eq 0 ]; then echo "::error::no tests were executed (gate skipped tests?)" >&2; exit 1; fi
if [ "${TESTS_FAIL}" -ne 0 ]; then echo "::error::${TESTS_FAIL} test failures" >&2; exit 1; fi

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