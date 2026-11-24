#!/bin/sh
# Start nova-notes as the unprivileged `nova` user (uid/gid 10001).
#
# Docker seeds fresh volumes from the image's prepared /app/data directory.
# Existing mounts must already be accessible to nova; startup never changes
# their ownership. Starting with `--user 10001:10001` skips the user switch.
#
# NOVA_RUN_AS_ROOT=true is a temporary break-glass override (not recommended).
set -eu

NOVA_UID=10001
NOVA_GID=10001

# Check access and launch the app under the same identity.
startup='
    data="${DATA_FOLDER:-/app/data}"
    if ! mkdir -p -- "$data" || ! [ -r "$data" ] || ! [ -w "$data" ] || ! [ -x "$data" ]; then
        printf "nova-notes: data directory %s is not accessible to the runtime user.\n" "$data" >&2
        printf "Create a backup and prepare the mounted data directory offline for the runtime user (default UID:GID 10001:10001), then restart. Startup does not change ownership.\n" >&2
        exit 1
    fi
    exec "$@"
'

if [ "$(id -u)" = "0" ]; then
    if [ "${NOVA_RUN_AS_ROOT:-false}" = "true" ]; then
        exec "$@"
    fi
    exec setpriv --reuid="$NOVA_UID" --regid="$NOVA_GID" --clear-groups -- \
        /bin/sh -eu -c "$startup" nova-entrypoint "$@"
fi
exec /bin/sh -eu -c "$startup" nova-entrypoint "$@"
