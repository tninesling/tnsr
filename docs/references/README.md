# Reference Papers

## Kernel performance reading order

The six additions below were selected against the current
[GEMM architecture proposal](../research/ada-gemm-replacement.md),
[register/layout findings](../research/ada-register-tiles.md), and
[training dataflow proposal](../research/ppo-training-dataflow.md).
The order is a recommendation for this repository, not a ranking of the papers.

| Order | Paper | Research focus for tnsr |
| --- | --- | --- |
| 1 | [ROLLER](roller-osdi-2022.pdf) | Construct resource-aware register/shared tiles and prune poor schedules before benchmarking. |
| 2 | [Triton](triton-mapl-2019.pdf) | Lower tile IR with coordinated thread ownership, coalescing, shared allocation, and prefetching. |
| 3 | [Welder](welder-osdi-2023.pdf) | Choose fusion boundaries and tile layouts together using intermediate traffic and reuse. |
| 4 | [ThunderKittens](thunderkittens-arxiv-2024.pdf) | Study register/shared layouts, swizzles, and producer/consumer pipelines. |
| 5 | [Ansor](ansor-osdi-2020.pdf) | Explore schedule families and allocate tuning effort to expensive subgraphs. |
| 6 | [FlashAttention-2](flashattention-2-arxiv-2023.pdf) | Improve fused attention work partitioning and reduce shared-memory communication. |

Start with ROLLER and Triton for the proposed hierarchical SIMT GEMM family.
Use Welder for training-region fusion and ThunderKittens for layout/pipeline
design. Ansor becomes useful once the candidate schedule families exist.
FlashAttention-2 supports the packed encoder work; existing profiling gives
standalone attention a smaller coverage envelope than the FFNs.

Treat the application notes below as research hypotheses. Preserve the selected
arithmetic policy: reduction reassociation, split-K, online normalization, and
lower-precision tensor-core operands can change results. Hopper-specific TMA
and WGMMA designs need alternatives on Ada. Validate actual JIT instructions,
registers, spills, tail handling, and complete training updates before claiming
a gain. Paper benchmark speedups are not predictions for tnsr.

## DNNFusion

- File: [`dnnfusion-pldi-2021.pdf`](dnnfusion-pldi-2021.pdf)
- Title: *DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion*
- Authors: Wei Niu, Jiexiong Guan, Yanzhi Wang, Gagan Agrawal, and Bin Ren
- Venue: PLDI 2021
- DOI: <https://doi.org/10.1145/3453483.3454083>
- Source: <https://arxiv.org/abs/2108.13342v2>
- Retrieved: 2026-08-08
- SHA-256: `69b55864c9ca9072e43bacdf09dc176ef506a47714bf84243bce788d48042459`

The arXiv record identifies the deposited version as CC BY 4.0. The paper is
stored here to make the fusion design reproducible and reviewable without
depending on an external URL.

## ROLLER

- File: [`roller-osdi-2022.pdf`](roller-osdi-2022.pdf)
- Title: *ROLLER: Fast and Efficient Tensor Compilation for Deep Learning*
- Authors: Hongyu Zhu et al.
- Venue: OSDI 2022
- Source: <https://www.usenix.org/conference/osdi22/presentation/zhu>
- PDF: <https://www.usenix.org/system/files/osdi22-zhu.pdf>
- Retrieved: 2026-10-31
- Pages: 17, including proceedings cover
- SHA-256: `231f5567c4da7dedc641364d885e2163f2ece11a290d6689c6b1ea5dc11b1443`

Study its rTile abstraction, bottom-up tile construction, and micro-performance
model. It narrows schedule choices using hardware properties rather than
relying entirely on a large empirical search.

**Question for tnsr:** can we rank the proposed 64x64 and 64x128 SIMT tiles
using reuse, register/shared capacity, alignment, tail waste, and output-grid
parallelism before measuring a small finalist set? Connect this to
`src/tile/matmul_schedule.rs`; calibrate against the actual generated kernel.

## Triton

- File: [`triton-mapl-2019.pdf`](triton-mapl-2019.pdf)
- Title: *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*
- Authors: Philippe Tillet, H. T. Kung, and David Cox
- Venue: MAPL 2019
- DOI: <https://doi.org/10.1145/3315508.3329973>
- Source/PDF: <https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf>
- Retrieved: 2026-10-31
- Pages: 10
- SHA-256: `7c4d200693c223c4d4c53c98b22c2693bdcf6adf3c902436b070368be097a7c1`

Start with Section 5, especially machine-dependent passes: hierarchical
tiling, memory coalescing, shared-memory allocation, and synchronization. This is
the original compiler paper, whose tile representation is particularly close
to our own; it is not documentation of the current Triton implementation.

**Question for tnsr:** can thread ownership and register-fragment prefetching
be designed together so larger outer products retain wide shared loads?
Compare with `src/ptx/graph/matmul.rs` and the documented stride-33 scalarization
failure. Preserve increasing-K accumulation under StrictF32.

## Welder

- File: [`welder-osdi-2023.pdf`](welder-osdi-2023.pdf)
- Title: *Welder: Scheduling Deep Learning Memory Access via Tile-graph*
- Authors: Yining Shi, Zhi Yang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Ziming Miao, Yuxiao Guo, Fan Yang, and Lidong Zhou
- Venue: OSDI 2023
- Source: <https://www.usenix.org/conference/osdi23/presentation/shi>
- PDF: <https://www.usenix.org/system/files/osdi23-shi.pdf>
- Retrieved: 2026-10-31
- Pages: 19, including proceedings cover
- SHA-256: `7105a13f0a737aa189dec8a13b13b78aa7ef6adedf1217fc43f46863a2462b27`

Study tile-graph construction, scheduling across memory levels, and its traffic
cost model. The useful extension to DNNFusion is jointly considering reuse
within operators and across operator boundaries.

**Question for tnsr:** can fusion planning estimate saved intermediate bytes
alongside shared/register capacity and recomputation, especially for
SwiGLU forward/backward regions and GEMM epilogues? Start at
`src/ptx/plan/fusion.rs` and `src/tile/graph.rs`. Check whether a fused region
reduces complete-update time after any loss in GEMM efficiency.

## ThunderKittens

- File: [`thunderkittens-arxiv-2024.pdf`](thunderkittens-arxiv-2024.pdf)
- Title: *ThunderKittens: Simple, Fast, and Adorable AI Kernels*
- Authors: Benjamin F. Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, and Christopher Re
- Version: arXiv:2410.20399v1 (2024)
- Source: <https://arxiv.org/abs/2410.20399v1>
- PDF: <https://arxiv.org/pdf/2410.20399v1>
- Retrieved: 2026-10-31
- Pages: 30
- SHA-256: `262204198eea890622265616d68eca4d95bdeb1173d6108041f594f7ddc56268`

Study register/shared tile representations, shared-memory swizzles, and
block/grid scheduling. Appendix C gives concrete layout examples useful for
checking how bank distribution interacts with hardware-supported transfers.

**Question for tnsr:** which layout invariants should our tile IR encode to
retain vector alignment while reducing bank conflicts? Separate reusable
ownership/layout principles from Hopper TMA/WGMMA mechanisms; our Ada path
cannot directly adopt those instructions. Evaluate both staging stores and
fragment loads, not only the nominal scalar bank mapping.

## Ansor

- File: [`ansor-osdi-2020.pdf`](ansor-osdi-2020.pdf)
- Title: *Ansor: Generating High-Performance Tensor Programs for Deep Learning*
- Authors: Lianmin Zheng et al.
- Venue: OSDI 2020
- Source: <https://www.usenix.org/conference/osdi20/presentation/zheng>
- PDF: <https://www.usenix.org/system/files/osdi20-zheng.pdf>
- Retrieved: 2026-10-31
- Pages: 18, including proceedings cover
- SHA-256: `73ef4169ed3856f016195bc28a7819e7040274e0af8fb90df2e92a964ea04232`

Read Sections 4-6 on hierarchical sketch generation, evolutionary search with
a learned cost model, and task scheduling. Its search covers program
structures as well as low-level tile annotations.

**Question for tnsr:** can a bounded tuner compare distinct SIMT/tensor-core
schedule families and spend its budget according to measured training cost?
Keep numerical-policy legality as a hard filter. Compare with the existing
rewrite-cost work in [learned rewrite cost](../learned-rewrite-cost.md), while
keeping graph rewrite selection and kernel scheduling as distinct decisions.

## FlashAttention-2

- File: [`flashattention-2-arxiv-2023.pdf`](flashattention-2-arxiv-2023.pdf)
- Title: *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
- Author: Tri Dao
- Version: arXiv:2307.08691v1 (2023; preprint of the ICLR 2024 work)
- Source: <https://arxiv.org/abs/2307.08691v1>
- PDF: <https://arxiv.org/pdf/2307.08691v1>
- Retrieved: 2026-10-31
- Pages: 14
- SHA-256: `4aa8935dfacaf6ae8c68f772ca92f730154a0dd0e1bceeb59c7bc56c512d5868`

Read Section 3 on reducing non-matmul work and partitioning work across blocks
and warps. It explains why avoiding score/probability materialization is only
part of making an attention kernel efficient.

**Question for tnsr:** how should a tiled replacement for the existing scalar
online loops divide query rows and K/V tiles without extra cross-warp
reductions? Connect to `src/ptx/plan/online.rs` and the
[packed attention training proposal](../research/ppo-fused-attention-training.md).
Include backward, padding masks, saved state, and precision checks; mathematical
exactness does not imply the same floating-point accumulation order.
