# Architecture

**Status:** design, not yet implemented. Written 2026-09-03.
**Working package name:** `chronoseq` — *provisional*, see [Open decisions](#open-decisions).

---

## 1. What this project is

Today the repo is a single 206-line script that unshuffles one specific video. This document
describes its intended form: a **CPU-first toolkit for recovering and verifying the temporal
structure of video.**

The underlying primitive is **seriation** — recovering a latent 1-D ordering from pairwise
similarity. Video frame reconstruction is one application of it. Once the primitive is built
properly, three capabilities fall out of the same engine:

| capability | question it answers |
| --- | --- |
| **reconstruct** | These frames are unordered. What was the true order? |
| **verify** | This video is *mostly* ordered. Where is that order broken? |
| **segment** | Where are the shot boundaries? |

`verify` is the one with real-world use beyond a puzzle: archival footage QA, dataset
validation, tamper detection.

## 2. Design principles

1. **CPU-only is the contract, not a fallback.** Every default path must run acceptably on a
   low-end laptop with no GPU. GPU is an optional accelerator selected by config, never a
   requirement. CI runs on free-tier CPU runners, which makes the green badge direct evidence
   for this claim.
2. **Video length must never be a hard limit.** Bounded memory via chunking; long videos trade
   time, not feasibility. "It will eventually get there" beats "it cannot."
3. **Measure before changing.** The evaluation harness lands first. Every subsequent claim is a
   delta against a recorded baseline, never an assertion.
4. **Linear where possible, quadratic only in bounded windows.** The current design is
   quadratic in both time and memory and dies at ~3,000 frames. See §7.
5. **Degrade tiers, don't degrade correctness.** Quality/cost trade-offs are explicit user
   choices (embedding tiers), not silent heuristics.

## 3. The baseline being replaced

Current pipeline in [`reconstruct.py`](../reconstruct.py): extract frames → pHash coarse sort →
dense 300×300 SSIM matrix → greedy nearest-neighbour path → 2-opt → orientation check → write.

**Measured on the bundled 300-frame clip:** SSIM matrix 49.6 s; full run ~104 s.
Adjacent-frame MAD improves 0.01818 → 0.00194, a **9.4× reduction in discontinuity**. 14 frames
still exceed 2× median jump.

Note the README's "95–99% accuracy" is **unverifiable** — the repo ships no ground-truth
ordering. Replacing that claim with measured numbers is the point of §6.6.

### Three defects carried by the baseline

1. **pHash is a no-op.** With `chunk_size=None` the SSIM matrix is *complete*, which makes the
   greedy path invariant to input ordering — the coarse sort provably cannot change the output.
   It costs ~10 s and does nothing. Separately, `phash_order` sorts by the hash's **raw integer
   value**, not Hamming distance as the README claims; pHash bits are DCT sign bits with no
   ordinal meaning, so integer sort is not a similarity sort at all.
2. **`choose_start` inverts its own spec.** The README correctly wants the most *unique* frame
   (a path endpoint). The code uses `argmin(row_sums)` on the distance matrix, selecting the most
   *central* frame. Should be `argmax`.
3. **Orientation is a bolt-on.** SSIM is symmetric — `S[i][j] == S[j][i]` — so the algorithm is
   structurally incapable of distinguishing forward from backward. `enforce_correct_direction`
   patches this after the fact with a global heuristic. §6.4 removes the need for it.

Also: shipped videos are `mpeg4` (MPEG-4 Part 2) from OpenCV's `mp4v` fourcc, which **no browser
can play**, at 87 MB / 62 MB. Any web delivery requires H.264 re-encoding.

## 4. Module map

```
                    ┌──────────┐
   video ─────────► │  decode  │ ──► frames (streamed, never all in RAM)
                    └────┬─────┘
                         ▼
                    ┌──────────┐
                    │  embed   │ ──► vectors        O(n)   tiered, §6.1
                    └────┬─────┘
                         ▼
                    ┌──────────┐
                    │  graph   │ ──► k-NN graph     O(n log n)  §6.2
                    └────┬─────┘
                         ▼
              ┌──────────┴───────────┐
              ▼                      ▼
        ┌──────────┐           ┌──────────┐
        │ segment  │           │  order   │  spectral seriation  §6.3
        └────┬─────┘           └────┬─────┘
             │                      ▼
             │                 ┌──────────┐
             │                 │   flow   │  adjacent pairs only  §6.4
             │                 └────┬─────┘
             ▼                      ▼
        shot bounds           ordering + orientation
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                   reconstruct              verify  §6.5
```

`chunk` (§6.7) wraps this whole graph in a windowed scheduler so total video length never
determines memory use.

## 5. Repository layout

```
chronoseq/
  __init__.py
  cli.py               # argparse/typer entry point, subcommands per §8
  decode.py            # streaming frame reader (ffmpeg pipe / cv2)
  embed/
    __init__.py        # tier dispatch
    pixels.py          # tier 0 — downscaled gray / DCT
    onnx_cnn.py        # tier 1 — MobileNetV3-Small via ONNX Runtime
    clip.py            # tier 2 — CLIP ViT-B/32, opt-in
  graph.py             # dense or hnswlib k-NN, cosine similarity
  order.py             # spectral seriation + local refinement
  flow.py              # Lucas-Kanade sparse flow, adjacent pairs
  segment.py           # shot boundary detection
  verify.py            # forensic signature detection
  chunk.py             # adaptive windowed scheduler
  evaluate.py          # ground-truth harness + metrics
  report.py            # JSON / human-readable output
docs/
  ARCHITECTURE.md      # this file
tests/
benchmarks/
pyproject.toml
```

## 6. Module contracts

### 6.1 `embed` — frames → vectors

The keystone change. Instead of comparing images pairwise (quadratic, expensive), map each
frame to a vector once (linear), then compare vectors. Dot products are BLAS-vectorized and
roughly **1000× cheaper per pair** than windowed SSIM. Even keeping a dense matrix, the bundled
300-frame case drops from ~50 s to milliseconds.

```python
def embed(frames: Iterable[np.ndarray], tier: int = 1, dim: int | None = None) -> np.ndarray:
    """Returns (n, d) float32, L2-normalised so cosine similarity == dot product."""
```

| tier | method | cost/frame (CPU) | 18k frames | use when |
| --- | --- | --- | --- | --- |
| **0** | downscaled gray / DCT coefficients | ~free | seconds | locked-off camera; the bundled clip |
| **1** | **MobileNetV3-Small, ONNX Runtime** | 2–10 ms | 36–180 s | **general footage — default** |
| **2** | CLIP ViT-B/32 | 50–100 ms | 15–30 min | opt-in high-quality |

All tiers are **linear** in frame count. Tier 1 buys robustness to camera motion and lighting
that SSIM structurally lacks. Tier selection is `--tier`, defaulting to 1, with tier 0 auto-
suggested (not auto-selected) when a static-camera probe succeeds.

*SSIM is retained only as a verification metric on small candidate sets, never as the matrix.*

### 6.2 `graph` — vectors → sparse similarity graph

```python
def build_graph(vectors: np.ndarray, k: int = 32, exact: bool = False) -> sparse.csr_matrix
```

Exact k-NN (cheap dot products) is fine to ~20k frames. Beyond that, `hnswlib` — CPU-only, no
GPU dependency, ~O(n log n). Memory goes from O(n²) to O(n·k): at n=18,000, k=32 that is
**4.6 MB instead of 1.3 GB**.

### 6.3 `order` — graph → ordering

**Spectral seriation.** Sort by the Fiedler vector (second-smallest eigenvector of the graph
Laplacian). Atkins, Boman & Hendrickson (1998), *A spectral algorithm for seriation and the
consecutive ones problem*, proved this recovers the exact ordering for Robinson matrices — and
the bundled clip's similarity matrix is visibly Robinson (clean diagonal band).

```python
def seriate(graph: sparse.csr_matrix, refine: bool = True) -> np.ndarray
```

**This is the architectural keystone, and the reason it is not merely a quality upgrade:**
`scipy.sparse.linalg.eigsh` computes the Fiedler vector on a *sparse* graph without ever
densifying. Greedy NN + 2-opt fundamentally requires a dense distance matrix; spectral does not.
**Choosing spectral seriation is what makes §6.2 and §6.7 possible at all.** The two items are
dependent, not independent.

Optional local refinement (windowed 2-opt over a small neighbourhood) may follow, but only if
§6.6 shows it earns its cost.

### 6.4 `flow` — optical flow refinement and orientation

Motivated by a specific gap: SSIM is a *symmetric scalar* and cannot distinguish forward from
backward. Optical flow is **vectorial and antisymmetric** (flow A→B ≈ −flow B→A), so it supplies
exactly what the appearance metric cannot.

It is also **strongest where appearance is weakest.** The bundled clip's SSIM values span only
**0.84–1.0** — brutally compressed — so in low-motion stretches candidate frames are nearly
indistinguishable by appearance. Flow magnitude and direction still separate them.

```python
def refine_with_flow(order: np.ndarray, frames: FrameSource) -> tuple[np.ndarray, Orientation]
```

**Cost discipline — the critical design point.** Flow is *never* computed pairwise across the
matrix; 45,000 flow computations is a non-starter. It runs **only on the ~n adjacent pairs of an
already-established candidate ordering** — 300 computations, not 45,000. Sparse Lucas-Kanade on
Shi-Tomasi corners, roughly an order of magnitude cheaper than dense Farnebäck and sufficient for
a global motion estimate.

Outputs: a principled orientation decision (replacing `enforce_correct_direction`), local
reordering in ambiguous low-motion runs, and per-adjacency continuity scores feeding §6.5.

### 6.5 `verify` — forensic integrity

Same engine, inverted question. Each corruption class has a **distinct, visually legible
signature in the similarity matrix** — confirmed empirically by rendering the bundled clip's
matrix under four permutations:

| corruption | matrix signature | flow signature |
| --- | --- | --- |
| intact | smooth diagonal band (Robinson) | coherent, continuous |
| shuffled | structureless noise | incoherent |
| reversed segment | band breaks and inverts in-block | flow sign flips |
| splice / moved block | bright rectangles stranded off-diagonal | discontinuity at seam |
| **dropped frame** | local band narrowing | **flow magnitude doubles** |
| **loop / duplicate** | **bright stripe parallel to main diagonal** | — |

Loop detection needs no hashing scheme — it is near-duplicate structure, found by scanning for
off-diagonal bands. **This is where pHash finally earns its place:** as an O(n) cheap pre-filter
for near-duplicate candidates (small Hamming radius), verified afterwards with SSIM or flow on
the handful of survivors. The dead code from §3 gets a real job here.

Output is a structured report (JSON + human-readable) listing detected anomalies with frame
ranges and confidence.

### 6.6 `evaluate` — the ground-truth harness

**Lands first. Everything else is measured against it.**

Take any video, shuffle with a recorded permutation, reconstruct, compare to truth:

- **Kendall tau** — rank correlation, the headline number
- **Spearman rho** — rank correlation, outlier-sensitive
- **exact-position accuracy** — fraction placed exactly right
- **adjacency accuracy** — fraction of true-adjacent pairs still adjacent (the metric that
  actually matters for perceived smoothness)
- **adjacent-frame MAD** — the baseline continuity measure, comparable to §3

This converts an unverifiable claim into a reproducible measurement, and doubles as the test
suite (§10).

### 6.7 `chunk` — the adaptive scheduler

Makes total video length irrelevant to memory.

- Overlapping windows of `W` frames, overlap `O` (defaults W=600, O=60).
- Each window seriated independently — **embarrassingly parallel**, `multiprocessing.Pool`.
- Stitch via the overlap: shared frames give the correspondence to align consecutive windows'
  orderings *and* resolve their relative orientation.
- Memory becomes **O(W·k), independent of total length**.

**Adaptive sizing.** Probe `psutil.virtual_memory().available` and `os.cpu_count()`; choose `W`
to fit a memory budget and workers to fit cores. A 2-core/8 GB laptop gets small windows and 2
workers; a 16-core box gets large windows and 16. Explicit `--window` / `--workers` override.

**Two consequences that bite if ignored:**

1. **Orientation is global but chunking is local.** Each window may independently come out
   reversed. The overlap resolves *relative* orientation between neighbours; one global flip
   decision follows at the end. §6.4 makes this far more reliable than the current heuristic.
2. **A video with a genuine scene cut has no single valid ordering across it.** Shot boundaries
   must be detected *first*, then seriation runs within each shot. This is why `segment` is a
   first-class module and CLI verb (§8) rather than an internal detail — it is a correctness
   prerequisite, not a nicety. It also falls out nearly free from the same similarity structure
   (block-diagonal breaks) and is independently useful and demoable.

## 7. Scaling analysis

Extrapolated from the measured 49.6 s / 45k pairs (~1.1 ms/pair) baseline:

| frames | @30 fps | pairs | **baseline** SSIM time | **baseline** dense RAM | **target** (tier 1 + k-NN) |
| --- | --- | --- | --- | --- | --- |
| 300 | 10 s | 45 K | 49.6 s *(measured)* | 0.4 MB | ~3 s |
| 1,000 | 33 s | 500 K | ~9 min | 4 MB | ~10 s |
| 3,000 | 100 s | 4.5 M | ~82 min | 36 MB | ~30 s |
| 18,000 | 10 min | 162 M | ~50 hours | 1.3 GB | ~3–5 min |
| 50,000 | 28 min | 1.25 B | weeks | 10 GB | ~10–15 min |

Two independent quadratic walls — time *and* memory — both removed by §6.1 + §6.2 + §6.7.

### GPU as optional accelerator

Two hot spots, both with drop-in CPU→GPU backends: the embedding pass (ONNX Runtime CUDA /
DirectML execution provider — a provider-selection flag, not a rewrite) and k-NN search
(FAISS-GPU). Sparse eigendecomposition is not worth GPU-ing. **CPU stays the default and the
tested path**; GPU is a config value.

## 8. CLI surface

```
chronoseq reconstruct <video> -o out.mp4 [--tier N] [--window W] [--workers N]
chronoseq verify      <video> [--report json|text]
chronoseq segment     <video> [--report json|text]
chronoseq evaluate    <video> [--seed N] [--tier N]
chronoseq embed       <video> -o data/        # dumps vectors + matrix for the demo site build
```

`embed` exists so the demo site's build step can precompute artifacts without reimplementing
anything (§11).

## 9. Rejected approaches — recorded so they are not revisited

**VLM / micro-LLM to "lead" optical flow.** Optical flow is a dense geometric correspondence
problem (per pixel, where did it go?), solved by variational methods or purpose-built CNNs.
Language models have no role in it. A VLM also operates at the wrong granularity: it produces
semantic captions at 1–10 s/frame on CPU, and two frames 33 ms apart receive **identical**
captions. It would cost minutes of CPU to learn nothing about sub-frame ordering, while breaking
the CPU-only contract. *The instinct underneath — learned representation beats hand-crafted — is
correct, and is honoured by the tiered CNN embeddings in §6.1.*

**Cryptographic hash chaining ("blockchain-style") for loop/tamper detection.** Fails twice.
(a) *No trusted reference:* for an arbitrary video, a chain computed from a tampered file is a
perfectly valid chain of that tampered file; without a signed original it detects nothing.
(b) *Avalanche is backwards:* crypto hashes change completely on a single bit flip, so any
transcode, recompression or re-mux shatters the chain. Loop detection needs *perceptual*
near-duplicate matching instead (§6.5).

*The legitimate relative of this idea is signed perceptual-hash provenance (C2PA / Content
Credentials), which requires controlling the video at creation time — a different product from
"analyse this file someone handed me." Noted as a possible future direction, out of scope.*

## 10. Testing and CI

- **The evaluation harness is the test suite.** Property-based tests shuffle with a fixed seed
  and assert recovery quality above a threshold — correctness testing and the accuracy claim are
  the same machinery.
- Unit tests per module; golden-file tests for report output.
- Deterministic seeds throughout; no test may depend on wall-clock or thread ordering.
- Committed benchmark numbers, regenerated by `benchmarks/`.
- **CI on GitHub Actions free-tier CPU runners.** Because the project is CPU-only by design, the
  full suite runs there — making the green badge direct evidence for the central design claim.

## 11. Relationship to the demo site

The demo site lives in a **separate private repo**, deployed publicly on Vercel; this repo stays
public as the thing the site promotes. The site consumes build-time artifacts from
`chronoseq embed`:

- 300×300 similarity matrix (~90 KB as uint8, ~40 KB as PNG)
- sprite sheet of all frames (1.0 MB WebP at 160×90)
- H.264 720p re-encodes (3.9 MB + 2.9 MB)

The expensive step (matrix construction) is precomputed and static; **greedy NN, 2-opt and
spectral ordering all run live in JS** — trivial on 300 nodes, and they are the visually
interesting steps. Under 10 MB total, fully static, no backend. `segment` gives the site a third
demoable visual alongside reconstruction and forensics.

## 12. Implementation phases

| phase | scope | gate |
| --- | --- | --- |
| **1** | Package skeleton, CLI, `evaluate` harness, baseline recorded | baseline numbers committed |
| **2** | `embed` tiers 0/1 + `graph` + `order` (spectral) | must beat phase-1 baseline, measured |
| **3** | `flow` refinement + orientation; delete `enforce_correct_direction` | orientation correct without the hack |
| **4** | `verify` + `segment` + pHash near-duplicate pre-filter | detects all six §6.5 signatures |
| **5** | `chunk` adaptive scheduler | bounded memory on a 30-min video |
| **6** | *later* — GPU providers, CLIP tier 2, C2PA provenance | — |

**Phase 1 is non-negotiable as the entry point.** If tiers + spectral do not measurably beat
greedy + 2-opt on this footage, that should cost an afternoon, not a rewrite. An honest
possibility: on locked-off tripod footage the baseline may already be near ceiling, in which
case the real wins are in **scaling and forensics**, not ordering accuracy — and the harness is
what will tell us that.

## Open decisions

1. **Package name.** `chronoseq` is a working placeholder used throughout this document. Needs
   confirming or replacing before phase 1 creates import paths.
2. **Repo strategy.** Whether this lands as commits on `video-reconstruction` (preserving the
   history and the "grew from a script" story) or as a fresh repo with the old one archived and
   linked. Leaning toward the former.
