# SD-301 — technical document (More Than A Hack 2027, Phase 1)

Team **SD-301**. Output: `outputs/SD-301.csv` (199,319 rows, `frame,label`). Regenerate with
`python run_sd301.py`. Everything is tracking-based; **no video decoding, no external dataset, no
training on the ground truth.**

## 1. Approach

Possession is inferred from the tactical Automatic Tracking Data (ATD) with a small, fully
explainable model: per-frame evidence, then a sequence decoder that encodes football rules.

1. **Ball clean-up.** Ball gaps up to 0.6 s are bridged linearly; longer gaps stay unknown
   (the ball is missing on 41 % of graded frames, so unknown must be a first-class value).
2. **Team evidence (ball visible).** Each player gets a weight `exp(-d²/2σ²)` (σ = 2.5 m, ignored
   beyond 6 m); the evidence is the signed, normalised weight of home minus away, scaled by how
   close the closest player is. This is a soft nearest-player rule that does not flip in crowds.
3. **Dead-ball evidence (log-odds per frame).** Ball outside the lines (+), ball stationary over
   2 s (+), ball travelling > 5 m in 2 s (−), unseen for > 4 s (+, weak), gap that starts or ends
   within 3 m of a touchline (+), players walking (+, weak).
4. **Evidence when the ball is not tracked (self-supervised).**
   * A *dead-ball* logistic model on collective player movement (mean speed, share running /
     standing, centroid drift, block length / width over ±1 s).
   * A *possession* logistic model per half on the shape of the two teams (who is further up the
     pitch, line heights, block length / width, mean speed difference).
   Features are computed over ±3 s windows (chosen over ±1 s and ±6 s on the *held-out
   pseudo-label* AUC, never on the possession ground truth).
   Targets are **pseudo-labels computed from the tracking itself** on frames where the ball *is*
     tracked (outside lines or stationary ⇒ dead; fast inside the pitch ⇒ alive; clear
     single-side control ⇒ that side). The models are refitted from scratch in every run, in
     < 1 s. Held-out AUC on the pseudo-labels: ≈ 0.89 (dead ball, cross-half), ≈ 0.85 (possession).
5. **Restart prior.** Throw-ins, goal kicks, corners and free kicks are awarded against the last
   side to touch the ball, so after a dead-ball stretch resuming with the *same* side costs
   ln(0.85/0.15) ≈ 1.7 nats. Implemented with two dead states (dead-after-H, dead-after-A).
6. **Smart Tagging.**
   * `SET PIECES` whose `Team` tag names a side add a soft team bias (0.3 per frame) over the
     12 s after the instance starts. `N/A` instances (all throw-ins, all kick-offs) are ignored.
   * Phases of play (build-up, progression, final third, ..., and the defending/transition
     counterparts) name who holds the ball (attacking codes: the tagged side; defending codes:
     the other side). They last tens of seconds and span several possessions (`DATA.md`), so they
     are **fallback evidence only**: weight 0.25 on frames where the ball is not tracked, never
     on frames where measured control exists.
7. **Decoding.** One Viterbi pass per half over the four states with switching costs expressing
   football: an H↔A hand-over needs ≈ 1.5 s of sustained opposing control, entering or leaving a
   stoppage ≈ 1 s. A missing ball gives no evidence, so the previous state persists (this is the
   behaviour the scorer's abstention fill would impose anyway). We never output `X` on graded
   frames; `X` is written outside the two published halves.

## 2. Assets used

| asset | used | how |
|---|---|---|
| tactical ATD (players + ball) | yes | everything above |
| tactical Smart Tagging | yes | set-piece team bias only |
| tactical homography | no | tracking is already in pitch coordinates |
| tactical video | no | never decoded; not needed for this approach |
| broadcast ATD | tried, **dropped** | aligned to the tactical timeline on player positions (`sd301/sync.py`; offsets ≈ 4013 frames in the 1st half, ≈ 26,920 in the 2nd; match cost 0.2-0.5 m vs > 1.7 m elsewhere). Its ball agrees with the tactical ball (median 0.8 m) but it only recovers 5,035 of the 64,643 missing graded ball frames (missing falls 40.7 % → 37.5 %: both trackers lose the ball at the same moments), and it did not improve the full-match Smart-Tagging agreement (0.739 → 0.721), so `use_broadcast_ball` is off |
| broadcast video / homography | no | not needed |
| public 5-minute ground truth | read-only sanity check | see §3 |

Also used: the published half boundaries (video frames 11185–90651 and 110808–190197, from
`DATA.md`) to decide where to write `X`.

## 3. Training data and rule compliance

* No labelled dataset of any kind was used. The only fitted models are the two logistic models
  of §1.4; their targets are derived from the provided *automatic* tracking, never from the
  possession ground truth, the video, or hand labels.
* The public ground truth was **read to understand the definition** of possession and dead
  ball (what a dead stretch looks like, that restarts follow stoppages), as the rules allow.
  It was not used to fit, tune or select any parameter. All constants are in `Params`
  (`sd301/pipeline.py`) and were fixed from football reasoning before looking at any score:
  distances in metres, durations in seconds, e.g. a hand-over needs ~1.5 s of control.
* Honest log of what the public window influenced: (a) the first run scored 0.00 spellF1
  because of a sign bug in the decoder (dead and in-play costs inverted); the fix made no
  parameter change; (b) later changes (player-only models, restart prior, Smart Tagging) were
  designed from the observations in the data documentation and checked for stability with
  ground-truth-free tools (`tools/sensitivity.py`, `tools/diagnostics.py`), not chosen by
  public score. The public window has 17 possessions and cannot rank versions anyway.
* The ±3 s feature window (v4) moved the public-window spellF1 from 0.533 to 0.477 (one
  possession in 17: inside the noise) while leaving the full-match Smart-Tagging agreement
  unchanged at 0.739. We kept it because the rule we set beforehand (held-out pseudo-label AUC)
  favoured it, and report the drop rather than reverse the decision on 17 possessions.
* We did not compare the output against the video.

## 4. Possession rules implemented

* Three states only: H, A, D; D is a stoppage, never a possession.
* **Controlled possession** is approximated by *sustained* proximity: a side takes over only
  after about 1.5 s of net control, so touches, deflections and aerial duels do not split a
  possession. A loose ball stays with the last controller (state persists).
* **Restarts**: stoppage → other side most likely; set-piece Smart Tagging overrides softly.
* **Dead ball**: out of play, stationary for seconds, or players in a stoppage pattern when the
  ball is untracked. We do not infer dead-ball ownership.

## 5. Limitations

* Dead ball is over-predicted when the tracker loses the ball for long stretches without other
  cues (the dead share is ≈ 48 % of graded frames; expected order of magnitude for effective
  playing time is 55–60 min in 90 min, i.e. ~35–45 % dead). The player-only model reduces but does
  not remove this.
* When the ball is visible but the stoppage has begun (tracker follows a ball boy or a
  second ball) we call it alive.
* Independent check, ground-truth-free: the fraction of alive frames inside Smart Tagging
  phases that name the side we label as holder is 0.739 for this pipeline against 0.646 for
  `baseline_smooth50` and 0.619 for `baseline_raw` (`tools/tag_agreement.py`; the same ordering
  as the organisers' published full-match scores). It is a consistency indicator, not a score:
  the tags are automatic and phase tags are also (weak) inputs.
* Sanity of the long stoppages: the longest dead stretches we output (74-216 s) are at players'
  mean speed 0.8 m/s with the ball untracked, around the four goals of the match.
* Smart Tagging team tags are automatic and sometimes wrong; their weight is deliberately small.
* No video: aerial duels and tackles that need appearance cues are invisible to us.
* Next steps with more time: use the homography with the video to see the ball when the tracker
  cannot; learn the dead-ball and possession models on an *external* tracked match.

## 6. Reproduction

```bash
pip install -r requirements.txt            # numpy, pandas
# put the data bundle in data/ (see DATA.md); data/ is git-ignored and never published
python run_sd301.py                         # -> outputs/SD-301.csv (≈ 15 s)
python evaluation/validate_submission.py outputs/SD-301.csv
python tools/diagnostics.py outputs/SD-301.csv
python tools/sensitivity.py                 # ground-truth-free robustness table
```

Deterministic: no random numbers anywhere. Python 3.8–3.12, numpy ≥ 1.21.

## 7. Robustness (ground-truth-free)

Changing each hand-set constant by ±30 % keeps ≥ 97 % of graded frames unchanged and moves the
possession count (277 in the reference run) between 240 and 327 (the hand-over time,
`switch_s`, being by far the most influential constant). See `outputs/sensitivity.md`.
