# bootstrap/ — "lost everything" runbook

These scripts rebuild a user's whole fleet environment on a cluster from zero:
the user toolchain, the project checkout and its lane worktrees, the shared
langywrap checkout (venv, pi, rtk), the fleet's runtime skeleton, and the
squashfs images. Use them when `/work` and every image are gone, or to
rehearse that into a scratch prefix.

Nothing here copies a credential. Logins are manual steps (below).

## Files

| File | Role |
|---|---|
| `rebuild-all.sh` | the ordered pipeline: toolchain, checkouts, state, toolchain image, workset image, version check |
| `toolchain-install.sh` | uv, uv pythons, uv tools, node, npm globals, rustup/cargo, `config/` shims, `bin/` helper shims, MKL linker script |
| `checkouts.sh` | project clone (+ `upstream`, `gc.auto=0`), lane worktrees, langywrap clone + submodules, venv, pi build, rtk build; `snapshot` records the live worktree list |
| `state-init.sh` | `agents/{logs,state/runtime/<project>/store}`, fleet homes, empty ephemeral store; prints the manual steps |
| `lock-versions.sh` | writes `versions.lock` from the live install; `--check` compares an install with it |
| `setup-caches.sh` | the `$HOME` to `/work` redirect tool (formerly in a project checkout); calls `toolchain-install.sh` for the env files and node |
| `versions.lock` | every pinned version, generated from the live install on 2026-10-12 |
| `_lib.sh` | shared option parsing, config and prefix handling |
| `../examples/adgfem.bootstrap.json` | the ADGFEM disaster-recovery config: `site` + `bootstrap` sections, usable before the checkout exists |
| `../examples/adgfem.worktrees` | `git worktree list --porcelain` snapshot of the live checkout (`checkouts.sh snapshot`) |

Configuration is the project's slurmhelps config (`docs/config.md`): the
`site` section (work root, homes, partitions) and the `bootstrap` section
(repo URLs, branch, langywrap URL/branch/submodules, worktree snapshot,
`env.local.sh` lines, `bin` shims, lock file). After a total loss the project
checkout is gone too, so pass a standalone copy with `--config FILE`.

All scripts take `--config FILE`, `--prefix DIR` (instead of `site.work_root`;
every site path follows) and `--dry-run`. They write only below the prefix and
are idempotent: run them again after a failure.

`toolchain-install.sh` (step `config`) writes `$WORK_ROOT/config/env.sh` as a
shim (`WORK_ROOT=...`, `SH_TOOLCHAIN_PROFILE`, `. slurmhelps/env/env.sh`) and
`config/toolchain.sh` as a shim to the site toolchain profile.
`maint/install-shims.sh` writes the same shims on a live host.

## What is pinned (versions.lock)

| Component | Pin | Notes |
|---|---|---|
| uv | `UV_VERSION` 0.12.24 | GitHub release tarball, sha256 checked, into `$PREFIX/bin` |
| uv pythons | `UV_PYTHONS` 3.11.17, 3.13.16 | 3.11 backs `venv-ocr`; 3.13 is installed but the fleet does not use it |
| uv tools | `UV_TOOLS` `rust-just==1.58.0` | `just` for justfiles |
| node | `NODE_VERSION` v26.11.1 (npm 11.20.0) | nodejs.org tarball, SHASUMS256 checked; pi needs >= 22.19 |
| npm globals | `NPM_GLOBALS` | none today (`npm-global` is empty) |
| rust | `RUSTUP_VERSION` 1.29.1, `RUST_TOOLCHAIN` 1.99.0, profile minimal | used only to build langywrap's `rtk` submodule. langywrap COMMITS the binary (`.exec/rtk`), so `checkouts.sh` skips the build when it is present; cargo is still installed (`--with-rust auto`, 12 s) so rtk can be rebuilt (168 s; the rebuilt binary is not byte-identical). Nothing else uses cargo. |
| fleet python | `FLEET_PYTHON` `/usr/bin/python3.12` (3.12.14) | a SYSTEM package of the cluster image; the langywrap venv is built from it, so it is in no image |
| langywrap venv | `LANGYWRAP_VENV_REQS` | base dependencies only, used as `uv pip` constraints. The `knowledge-graph` extra (textify/graphify) is not in the live venv and the fleet does not need it. |
| langywrap, pi, rtk | `*_COMMIT`, `PI_RELEASE_TAG` | a rebuild takes the branch tip; `--check` shows the drift |

Regenerate the lock from the live install after an intentional upgrade:

    bootstrap/lock-versions.sh --config examples/adgfem.bootstrap.json

## Submodules

`LANGYWRAP_SUBMODULES="pi rtk"`. `pi` is the langypi harness (node build,
`scripts/build_pi.sh`); it needs the upstream tag `v0.85.1`, which
`checkouts.sh` fetches from `PI_UPSTREAM`. `rtk` is built with cargo. The
other submodules stay uninitialized, as on the live checkout: `textify` and
`graphify` belong to the optional `knowledge-graph` extra; `headroom`,
`openwolf` and `tmuix` are optional tools.

## Runbook: everything is gone

Run each heavy step inside Slurm, never on the login node. On MFF Troja:

    srun -p ffa-short -c 8 --mem=32G --time=110 env -u BASH_ENV bash <cmd>

0. **Manual: GitHub SSH key** (manual step 1). `holimart/adgfem` and
   `holimart/langywrap` are private, so the clones need it.
1. **Get slurmhelps back:** `git clone git@github.com:holimart/slurmhelps.git /work/ffa/$USER/slurmhelps`.
2. **Rehearse** (optional; nothing live is touched):

       bootstrap/rebuild-all.sh --config examples/adgfem.bootstrap.json \
           --prefix /tmp/rehearsal --dry-run

3. **Rebuild** into the real work root:

       bootstrap/rebuild-all.sh --config examples/adgfem.bootstrap.json

   It prints a step table. `wsimage` is SKIPPED until the project's
   `lanes.json` has a `fleet.workset` block. With `--prefix`, the image steps
   read a `<clone>/slurmhelps.json` (git-excluded) whose `site.work_root` is
   the prefix, so a rehearsal never builds from or into the live tree. `check` reports `DIFF` for
   commits that moved since the lock (expected) and for any version that did
   not reproduce (investigate).
4. **Manual: logins** (steps 2-5 below).
5. **Shims for other projects:** run `maint/install-shims.sh` so the old
   `$WORK_ROOT/config/*.sh` and `bin/*.sh` paths keep working.
6. **Verify:** `ly ensure`, then `ly doctor` (`langypi doctor --strict`
   inside the allocation), `ly status`.

Partial reruns: `rebuild-all.sh --from checkouts`, `--skip tcimage,wsimage`;
`checkouts.sh --skip pi,rtk`; `toolchain-install.sh --only node,config`.

## Manual steps (cannot be scripted)

1. **GitHub SSH key.** `ssh-keygen -t ed25519`, add the public key to GitHub
   (and to GitLab MFF for the `upstream` remote), check with
   `ssh -T git@github.com`. `checkouts.sh --https` rewrites GitHub URLs to
   https, which only works for public repos (`holimart/pi` is public).
2. **Pi / openai-codex OAuth**, interactive, on the login node, once:

       source $WORK_ROOT/config/env.sh
       export HOME=$WORK_ROOT/home/compute PI_CODING_AGENT_DIR=$WORK_ROOT/home/pi/.pi/agent
       langypi            # then inside Pi: /login openai-codex
       langypi auth check --provider openai-codex --json --no-refresh

   This writes `home/pi/.pi/agent/auth.json` (0600). The workset image
   excludes it; stage-in copies it from `/work` into the node-local tree.
3. **Codex CLI** (only if a lane uses it): `CODEX_HOME=$WORK_ROOT/home/codex codex login`.
4. **API keys** come from the environment (`OLLAMA_API_KEY` for Ollama Cloud).
   Keep them in a private 0600 file sourced from `config/env.local.sh`, never
   in a repo or an image.
5. **Claude Code steward:** `CLAUDE_CONFIG_DIR=$WORK_ROOT/home/claude claude`, then log in.

## Rehearsal (2026-10-12, compute node c14t32, prefix in node-local /tmp)

`rebuild-all.sh` ran end to end in 131 s: toolchain 22 s, checkouts 109 s
(including the pi build, about 50 s). The tree was 2.3 GB. Every pinned
version reproduced (17 of 18 keys MATCH). The one DIFF was `LANGYWRAP_COMMIT`,
because the langywrap master tip moved after the lock was written. A second
run of `toolchain-install.sh` found everything present. A forced rtk rebuild
took 168 s and printed `rtk 0.42.3`.

Network from compute nodes: https to github.com, nodejs.org and the npm
registry works, and `ssh git@github.com` authenticates with the login key in
`~/.ssh`.

**Lane branches are not on origin.** Of the 9 lane worktrees in the snapshot,
only `count-only` (a detached commit already on origin) could be recreated.
`lane/harden`, `lane/selector`, `lane/features` and the other lane branches
exist only in the `/work` repo. After a total loss, all uncommitted or
unpushed lane work is gone. To make it recoverable, push the lane branches or
keep the workset image somewhere other than `/work`.
