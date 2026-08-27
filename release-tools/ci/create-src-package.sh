#!/usr/bin/env bash
# Build the signed source distribution: git archive -> tar.gz -> gpg .asc -> .sha512,
# then self-verify signature/checksum and reject unexpected binaries.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

mkdir -p "${ARTIFACT_DIR}"
umask 077

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

echo "== git archive ${GIT_TAG} =="
git archive --format=tar.gz -o "${STAGE}/${PKG_TARBALL}" "${GIT_TAG}"

echo "== gpg detach-sign =="
gpg --batch --yes --armor --detach-sign \
  --pinentry-mode loopback --passphrase-file "${GPG_PASSFILE}" \
  -o "${STAGE}/${PKG_TARBALL}.asc" "${STAGE}/${PKG_TARBALL}"

echo "== sha512 =="
( cd "${STAGE}" && sha512sum "${PKG_TARBALL}" > "${PKG_TARBALL}.sha512" )

echo "== self-verify signature =="
gpg --verify "${STAGE}/${PKG_TARBALL}.asc" "${STAGE}/${PKG_TARBALL}" >/dev/null 2>&1 \
  || { echo "::error::GPG signature verification failed" >&2; exit 1; }

echo "== self-verify checksum =="
( cd "${STAGE}" && sha512sum -c "${PKG_TARBALL}.sha512" ) || { echo "::error::SHA512 mismatch" >&2; exit 1; }

echo "== binary audit (only maven-wrapper.jar allowed) =="
if tar -tzf "${STAGE}/${PKG_TARBALL}" \
   | grep -E '\.(jar|class|zip|war)$' \
   | grep -v '\.mvn/wrapper/maven-wrapper\.jar$' | grep -q .; then
  echo "::error::unexpected binary files inside source package" >&2
  tar -tzf "${STAGE}/${PKG_TARBALL}" | grep -E '\.(jar|class|zip|war)$' | grep -v '\.mvn/wrapper/maven-wrapper\.jar$' >&2
  exit 1
fi

# == inspect the actual tarball (catch CRLF / exec-bit / licensing gaps) ==
echo "== inspect source package integrity =="
XF="${STAGE}/xf"
mkdir -p "${XF}"
tar -xzf "${STAGE}/${PKG_TARBALL}" -C "${XF}"

# 1) no CRLF in release-critical LF files (mvnw, *.sh, .gitattributes); Windows *.cmd/*.bat stay CRLF on purpose
if grep -rlU $'\r' "${XF}" --include='mvnw' --include='*.sh' --include='.gitattributes' 2>/dev/null | grep -q .; then
  echo "::error::CRLF found in LF source files inside the tarball" >&2
  exit 1
fi

# 2) mvnw must carry the executable bit for hermetic Unix builds
MODE="$(stat -c '%a' "${XF}/mvnw" 2>/dev/null || echo '0')"
case "${MODE}" in
  5??|7??) : ;;
  *) echo "::error::mvnw is not executable in the source package (mode ${MODE})" >&2; exit 1 ;;
esac

# 3) RAT must pass on the packaged tree itself (catches license-header gaps, e.g. .gitattributes)
if ! ( cd "${XF}" && ./mvnw -q org.apache.rat:apache-rat-plugin:check ); then
  echo "::error::RAT check failed inside the source package" >&2
  exit 1
fi

# 4) signature must exist and be non-empty
if [ ! -s "${STAGE}/${PKG_TARBALL}.asc" ]; then
  echo "::error::missing or empty .asc signature" >&2
  exit 1
fi

# stage artifacts for the svn/nexus steps
cp "${STAGE}/${PKG_TARBALL}"* "${ARTIFACT_DIR}/"
chmod 644 "${ARTIFACT_DIR}/${PKG_TARBALL}"*
ls -l "${ARTIFACT_DIR}"
echo "Source package ready under ${ARTIFACT_DIR}"