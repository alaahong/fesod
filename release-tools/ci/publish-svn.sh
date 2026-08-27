#!/usr/bin/env bash
# Commit the signed source package to the dev distribution svn:
#   https://dist.apache.org/repos/dist/dev/incubator/fesod/<rev>-rc<rc>/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

: "${ASF_USERNAME:?missing ASF_USERNAME secret}"
: "${ASF_PASSWORD:?missing ASF_PASSWORD secret}"

# The exact secret value will be masked by GitHub as *** in logs even here.
SVN_CREDS=(--username "${ASF_USERNAME}" --password "${ASF_PASSWORD}" --non-interactive --no-auth-cache)

WC="$(mktemp -d)"
trap 'rm -rf "${WC}"' EXIT

if [ ! -f "${ARTIFACT_DIR}/${PKG_TARBALL}" ]; then
  echo "::error::source package missing; run create-src-package first" >&2
  exit 1
fi

echo "== checkout dev dist root =="
svn co "${SVN_DIST_URL}" "${WC}" "${SVN_CREDS[@]}"

echo "== stage artifacts =="
mkdir -p "${WC}/${SVN_DIR_NAME}"
cp "${ARTIFACT_DIR}/${PKG_TARBALL}"* "${WC}/${SVN_DIR_NAME}/"
svn add "${WC}/${SVN_DIR_NAME}"

echo "== commit =="
svn commit "${WC}/${SVN_DIR_NAME}" \
  -m "[Release] Apache ${PROJECT_SHORT} ${REVISION} rc${RC} source distribution" \
  "${SVN_CREDS[@]}"

echo "Published: ${SVN_RC_URL}/"