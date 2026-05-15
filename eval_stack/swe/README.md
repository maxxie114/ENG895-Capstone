# SWE-bench Verified — GLM Coding Plan

Inference runs on the `eval-minimax` DigitalOcean droplet against the GLM Coding
Plan endpoint. Patches are graded by **sb-cli** (no local Docker needed for
grading; the droplet has Docker only for the agent's per-instance sandboxes).

## Files

| File | Where it runs | What it does |
|---|---|---|
| `config.yaml` | droplet | Overrides on top of mini-swe-agent's default `swebench.yaml` (model, retries, timeouts) |
| `run_inference.sh` | droplet | **Full pipeline**: runs `mini-extra swebench` then auto-grades `preds.json` with the local `swebench.harness.run_evaluation`. Refuses to start if `OPENAI_BASE_URL` isn't the coding endpoint |
| `setup_droplet.sh` | local | scp's `config.yaml` + `run_inference.sh` to the droplet. Idempotent |
| `sync_loop.sh` | local | Continuous `rsync droplet:~/swebench/runs/ → ./results/` while the run is live |
| `submit.sh` | local | sb-cli escape hatch — kept for when [SWE-bench/sb-cli#27](https://github.com/SWE-bench/sb-cli/issues/27) is fixed. Currently unreliable (always returns "failed") |

> **Why not sb-cli?** As of 2026-05-13 the sb-cli evaluation backend has a known
> server-side bug ([#25](https://github.com/SWE-bench/sb-cli/issues/25),
> [#26](https://github.com/SWE-bench/sb-cli/issues/26),
> [#27](https://github.com/SWE-bench/sb-cli/issues/27)) that returns
> `failed=N, resolved=0` for every submission, including known-good gold patches.
> We confirmed empirically. The pipeline runs the same canonical harness locally
> on the droplet and gets truthful verdicts.

## End-to-end flow

```bash
# 0. (One-time) sb-cli account already verified, key in ../.env
# 0. (One-time) ssh key already on droplet, GLM_API_KEY pushed to droplet's ~/swebench/.env

# 1. Push latest scripts/config to droplet
./setup_droplet.sh

# 2. Pilot — 3 instances, 4 workers — to validate the pipeline
ssh root@64.23.189.5 \
  "cd ~/swebench && RUN_ID=glm51-pilot SLICE=0:3 WORKERS=4 nohup ./run_inference.sh > /tmp/pilot.out 2>&1 &"

# 3. Watch progress
ssh root@64.23.189.5 'tail -f ~/swebench/runs/glm51-pilot/run.log'

# 4. Pull results down (start in another terminal during the run, leave running)
./sync_loop.sh

# 5. (auto) sb-cli submission runs at the end of run_inference.sh.
#    Report lands at ~/swebench/runs/$RUN_ID/sb-report/

# 6. Full run (after pilot validates) — same script, no slice
ssh root@64.23.189.5 \
  "cd ~/swebench && RUN_ID=glm51-full WORKERS=24 nohup ./run_inference.sh > /tmp/full.out 2>&1 &"
```

## Resumability

mini-swe-agent skips already-completed instances when re-run with the same `-o`
output dir. Just re-launch with the same `RUN_ID` to resume after any crash.

## Endpoint guardrail

`run_inference.sh` aborts unless `OPENAI_BASE_URL` contains `/coding/` —
prevents accidental pay-as-you-go billing if the env file is ever wrong.
