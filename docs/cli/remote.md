# Remote inference

`--remote NAME` (config `remote`/`remote_hosts`) runs the whole thing on
another machine: inference, model management (load/unload, `/api/ps`) and
`/pull` happen there, while sessions, files, images and the tool harness stay
local — there is no separate remote code path. Names resolve through
`tailscale status --json` (falling back to the MagicDNS name), and any
`host[:port]` works. Short names come from the config's `remote_hosts` map —
the shipped entries are just my own fleet, so replace them with yours. A
short `/api/version` preflight fails fast with the Tailscale node state when
the machine is down, and resuming a session reconnects to the machine it last
ran on.
