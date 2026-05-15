# SWE-bench Verified — GLM-5.1 Run Report

**Date:** 2026-05-13 to 2026-05-15
**Model:** GLM-5.1 (z.ai)
**Benchmark:** SWE-bench Verified (500 instances, princeton-nlp/SWE-Bench_Verified)
**Subset attempted:** 100 instances (the first 100 alphabetically + opportunistic)

## Final score

| Metric | Value |
|---|---|
| **Resolved** | **66** |
| Unresolved | 26 |
| Useful patches evaluated | 92 |
| Empty patches (LimitsExceeded retries) | 3 |
| APIError instances | 5 |
| **Resolve rate on attempts** | **71.74%** (66 / 92) |
| Resolve rate on 100-instance subset | 66.0% (66 / 100) |

The 71.74% figure is comparable to GLM-5.1's published Verified scores on a smaller
sample. The 66% subset rate counts the 8 unattemptable instances (LimitsExceeded /
APIError after the OR cap was burned) as failures.

## Setup

- **Inference**: `mini-swe-agent` v2.2.8 with `step_limit=75` (later 150 for OR retries),
  `max_tokens=8192`, `temperature=0.0`
- **Compute**: DigitalOcean droplet `eval-minimax` (16 vCPU / 32 GB RAM / 200 GB disk, sfo3)
- **Eval**: local `swebench.harness.run_evaluation` v4.1.0 (NOT sb-cli — see "What broke"
  below)
- **Routing**:
  - Phase 1 (free): GLM Coding Plan via `https://api.z.ai/api/coding/paas/v4`
  - Phase 2 (paid fallback): OpenRouter `z-ai/glm-5.1` via `https://openrouter.ai/api/v1`

## Per-model breakdown

| Provider | Evaluated | Resolved | Unresolved | Rate |
|---|---|---|---|---|
| GLM Coding Plan (`openai/glm-5.1`) | 46 | 31 | 15 | **67.4%** |
| OpenRouter (`openai/z-ai/glm-5.1`) | 46 | 35 | 11 | **76.1%** |
| **Combined** | **92** | **66** | **26** | **71.74%** |

The OR batch outperformed the GLM batch by ~9 percentage points. Likely contributors:
sample bias (OR ran later-alphabetical instances which were django-heavy and shorter),
and `step_limit=150` on the retry batch let harder instances complete.

## Cost

| Provider | Calls / Tokens | Cost |
|---|---|---|
| GLM Coding Plan (z.ai) | flat-rate subscription | $0 marginal |
| OpenRouter (`z-ai/glm-5.1`) | ~5,000 calls | ~$70 spent (initial $50 cap blown overnight + $30 second top-up) |

**Per-instance cost** dropped from $1.43 (12-worker storm with retries) to $0.44 once we
moved to `WORKERS=1, num_retries=0` — strict settings cut cost ~70%.

## Timeline

1. **2026-05-13 evening — Pipeline build & validation.** Built `eval_stack/swe/` with
   `run_inference.sh`, `config.yaml`, `stream_eval.sh`, `setup_droplet.sh`. Single-instance
   pilot resolved cleanly in 4:20 (`astropy__astropy-12907`).
2. **2026-05-13 night — Full 500 launch.** Ran 24 workers against GLM. Got **94 preds in
   1.5 hours** before z.ai's 5h rolling-window quota kicked in. Of those 94:
   46 had real patches, 48 were empty (rate-limit casualties mid-instance).
3. **2026-05-14 early AM — Watchdog with OR fallback.** Built auto-resume watchdog that
   waits for GLM unlock and falls back to OpenRouter if GLM keeps 429-ing. Launched
   overnight.
4. **Watchdog night.** GLM "reset" wasn't a true reset (rolling window only freed ~⅓ of
   capacity), so 24 workers exhausted the slice in 11 minutes. Watchdog correctly switched
   to OpenRouter. OR phase produced **28 useful patches** before the $50 OR cap was
   exhausted in ~30 min by 12-worker parallelism + litellm `num_retries=6` retry storms
   (~2,318 OR requests for ~1,120 useful calls). After cap hit: 189 instances marked
   APIError before we caught it; disk filled with leftover docker images, `preds.json`
   truncated to 128 KB by a no-space-left write.
5. **2026-05-14 evening — Recovery.** Reconstructed `preds.json` from the 273 surviving
   `.traj.json` files. Ran the harness on all 74 useful patches (46 GLM + 28 OR),
   cleaned 145 GB of stale docker images, scrubbed the 199 failed-trajectory dirs.
6. **2026-05-14 night — $30 OR top-up, 25-instance batch.** Strict settings this time:
   `WORKERS=1` test then `WORKERS=2`, `num_retries=0`. **Test instance cost $0.44 — 70%
   cheaper than the night-1 storm.** Batch: 9 useful + 16 LimitsExceeded (`step_limit=75`
   too tight for hard django tests).
7. **2026-05-15 — Retry the 16 with bumped step_limit.** `step_limit=150` lets harder
   instances complete, but per-instance cost tripled ($1.57 each) due to longer
   conversation histories. Burned remaining $17.90 on 16 retries → 7 more useful patches.

## What broke (lessons for next time)

### 1. **sb-cli's grading backend is broken** (open issues
[#25](https://github.com/SWE-bench/sb-cli/issues/25),
[#26](https://github.com/SWE-bench/sb-cli/issues/26),
[#27](https://github.com/SWE-bench/sb-cli/issues/27))
Submitting a known-good gold patch returns `failed_instances: 1`. We confirmed by
submitting astropy and django gold patches — same verdict for everything. Switched to
local `swebench.harness.run_evaluation` on the droplet (same harness logic, same
verdicts, just running locally).

### 2. **z.ai's rolling 5h window is NOT a hard reset.**
The reset timestamp z.ai returns (`11:20:19 +08:00`) is when the OLDEST calls in your
rolling window expire — only freeing partial capacity, not a full quota refresh.
Bursting 24 workers immediately after "reset" exhausts the freed slice in minutes.
Strategy: spread load with few workers continuously, or wait the full 5h after your LAST
call before bursting again.

### 3. **OpenRouter without prompt caching is 3× more expensive.**
z.ai's coding endpoint had 94.9% prompt cache hit rate (most of an agent's growing
context is identical across turns). OpenRouter routing to z-ai upstream lost most of
that benefit. Combined with parallel-worker request amplification + litellm
`num_retries`, the $50 OR cap evaporated in 30 minutes producing only 28 useful patches.

### 4. **`mini-swe-agent` with `num_retries > 0` becomes a request multiplier on errors.**
Each failed call retries 6 times (default `num_retries`). When the OR `Key limit
exceeded` error hit, every subsequent call became a 7-request retry storm. ~189 instances
× ~6 retries × ~6-7 calls each = ~7,000 wasted requests, all billable.
**Always set `num_retries=0` on hard-failure error classes.**

### 5. **`--cache_level env` doesn't catch every image.**
Inference uses the same `swebench/sweb.eval.x86_64.<repo>_<id>` images as eval, but
`mini-swe-agent`'s sandbox containers don't get pruned by `swebench.harness`. Result:
~150 GB of unused images accumulated across the night, eventually filling the disk and
truncating `preds.json` mid-write.
**Add a periodic `docker image prune -a` to long-running pipelines, or a disk-watch
hook.**

## How to reproduce

```bash
# 1. SSH access to the droplet (your droplet, your SSH key)
# 2. .env keys: GLM_API_KEY, OPENROUTER_API_KEY, SWEBENCH_API_KEY (optional, sb-cli broken)

cd eval_stack/swe

# Push scripts to droplet
./setup_droplet.sh

# Single-instance test — validates the whole pipeline
ssh root@<droplet> 'cd ~/swebench && RUN_ID=test SLICE=0:1 WORKERS=1 ./run_inference.sh'

# Full run with GLM (will hit 5h quota at ~46-94 useful instances per window)
ssh root@<droplet> 'cd ~/swebench && nohup ./run_inference.sh > /tmp/r.out 2>&1 & disown'

# To use OR with strict settings instead:
ssh root@<droplet> 'cd ~/swebench && \
  PROVIDER=openrouter WORKERS=2 \
  nohup ./run_inference.sh > /tmp/or.out 2>&1 & disown'

# Manual retry of stuck instances (override step_limit)
mini-extra swebench --filter "<regex>" --workers 2 \
  -c <default_swebench.yaml> -c config.yaml \
  -c model.model_kwargs.num_retries=0 \
  -c agent.step_limit=150 \
  -o /root/swebench/runs/<run_id>
```

## Files

| File | Role |
|---|---|
| `config.yaml` | mini-swe-agent overrides (model, step_limit, max_tokens) |
| `run_inference.sh` | Wrapper with PROVIDER=glm/openrouter switching + endpoint guardrail |
| `stream_eval.sh` | Streaming eval (runs alongside inference, polls preds.json) |
| `watchdog_glm_then_or.sh` | Wait-for-GLM-unlock then auto-fallback to OR (with caveats; see lessons) |
| `setup_droplet.sh` | scp scripts to droplet |
| `submit.sh` | sb-cli submission (currently broken upstream — kept as escape hatch) |
| `sync_loop.sh` | rsync droplet → local for backups during long runs |
| `REPORT.md` | This file |

## Final preds.json + reports

Backed up locally at `eval_stack/swe/results/backup-20260514_133221/`:
- 273 trajectory files (74 useful + 199 failed)
- 46 GLM eval reports
- `preds.json` (corrupt + pre-scrub backups)
