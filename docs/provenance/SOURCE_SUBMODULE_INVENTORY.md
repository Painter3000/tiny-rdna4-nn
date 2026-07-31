# Public submodule inventory

The independent Model-B public repository materializes and pins the following
Git submodules. The top-level Gitlinks are part of the first public commit.

## Top-level submodules

| Path | URL | Pinned commit |
|---|---|---|
| `dependencies/cutlass` | `https://github.com/NVIDIA/cutlass` | `82f5075946e2569589439d500733b700a3141374` |
| `dependencies/fmt` | `https://github.com/fmtlib/fmt` | `fa2eb2d2e3ec5c21629f8ccd88ae05ec40b963fa` |
| `dependencies/cmrc` | `https://github.com/vector-of-bool/cmrc` | `952ffddba731fc110bd50409e8d2b8a06abbd237` |
| `dependencies/amd-gsplat` | `https://github.com/Painter3000/amd-gsplat-rocm72-gfx1201` | `2c62b22552c0ad4ed120aae304ce66ae27bc5d08` |

## Recursive amd-gsplat dependency

| Path | URL | Pinned commit |
|---|---|---|
| `dependencies/amd-gsplat/gsplat/cuda/csrc/third_party/glm` | `https://github.com/g-truc/glm.git` | `33b4a621a697a305bc3a7610d290677b96beb181` |

The nested GLM Gitlink is owned by the pinned amd-gsplat commit. It is
materialized with `git submodule update --init --recursive`.

## Validation contract

The public Phase-4A2 build checks:

- every required top-level dependency is recorded as a Gitlink;
- each initialized submodule HEAD matches its pinned Gitlink;
- nested GLM is initialized and matches its amd-gsplat Gitlink;
- the superproject and all required submodules are clean before building;
- no dependency clone or network operation occurs during the build.
