#!/usr/bin/env bash
# Import the signing private key and configure gpg-agent for non-interactive
# batch signing. No secret is ever echoed.
#
# Secrets (env): GPG_PRIVATE_KEY (armored, or base64 of armored), GPG_PASSPHRASE.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

: "${GPG_PRIVATE_KEY:?missing GPG_PRIVATE_KEY secret}"
: "${GPG_PASSPHRASE:?missing GPG_PASSPHRASE secret}"

mkdir -p "${GNUPGHOME}" "${CREDS_DIR}" "${ARTIFACT_DIR}"
chmod 700 "${GNUPGHOME}"

# allow pinentry loopback so --passphrase-file works in batch mode
printf 'allow-loopback-pinentry\n' > "${GNUPGHOME}/gpg-agent.conf"

# write passphrase to a 0600 file (referenced by -Dgpg.passphraseFile, never printed)
umask 077
printf '%s' "${GPG_PASSPHRASE}" > "${GPG_PASSFILE}"

# import the key (auto-detect armored vs base64-of-armored)
umask 077
KEY_BODY="$(printf '%s' "${GPG_PRIVATE_KEY}")"
if printf '%s' "${KEY_BODY}" | sed -n '1p' | grep -q -- "-----BEGIN PGP"; then
  printf '%s' "${KEY_BODY}" | gpg --batch --yes --import -
else
  printf '%s' "${KEY_BODY}" | base64 --decode | gpg --batch --yes --import -
fi

# fail if no usable secret key got imported
SECRETS="$(gpg --batch --list-secret-keys --with-colons 2>/dev/null | grep '^sec:' | head -n1 || true)"
if [ -z "${SECRETS}" ]; then
  echo "::error::No secret signing key imported from GPG_PRIVATE_KEY" >&2
  exit 1
fi

echo "GPG key imported into ephemeral GNUPGHOME=${GNUPGHOME}"