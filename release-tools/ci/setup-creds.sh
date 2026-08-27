#!/usr/bin/env bash
# Generate, at runtime only, the credential-bearing config files:
#   ~/.m2/settings.xml   (ASF staging server username/password from env secret)
# Files are 0600, never catted to logs, and live only in the ephemeral job home.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

: "${ASF_USERNAME:?missing ASF_USERNAME secret}"
: "${ASF_PASSWORD:?missing ASF_PASSWORD secret}"

umask 077

# ---- ~/.m2/settings.xml : ASF Nexus staging server ----
M2_DIR="${HOME}/.m2"
mkdir -p "${M2_DIR}"
cat > "${M2_DIR}/settings.xml" <<'EOF_SETTINGS'
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0">
  <servers>
    <server>
      <id>apache.releases.https</id>
      <username>${env.ASF_USERNAME}</username>
      <password>${env.ASF_PASSWORD}</password>
    </server>
    <server>
      <id>apache.snapshots.https</id>
      <username>${env.ASF_USERNAME}</username>
      <password>${env.ASF_PASSWORD}</password>
    </server>
  </servers>
</settings>
EOF_SETTINGS
chmod 600 "${M2_DIR}/settings.xml"

# ---- ~/.subversion : store ASF auth so svn needs no interactive prompt ----
SVN_HOME="${HOME}/.subversion"
mkdir -p "${SVN_HOME}/auth" "${SVN_HOME}/servers.d" "${SVN_HOME}/config.d"
# non-interactive; credentials passed per-invocation by publish-svn.sh

echo "Runtime credentials configured under M2_DIR=${M2_DIR} (not echoed)"