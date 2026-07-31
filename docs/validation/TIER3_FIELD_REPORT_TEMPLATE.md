# Tier-3 external field report

Incomplete reports are `INSUFFICIENT_ENVIRONMENT_METADATA`.

| Field | Value |
|---|---|
| Operating system | |
| Kernel | |
| GPU model | |
| GPU / PCI revision | |
| gcnArchName | |
| ROCm version | |
| HIP version | |
| PyTorch version | |
| rocWMMA version and commit | |
| Compiler and complete version | |
| Compiler flags | |
| Linker flags | |
| Build command | |
| tiny-rdna4-nn commit | |
| amd-gsplat commit | |
| Local binary SHA-256 | |
| Tier-1 result | |
| Tier-2 status | |
| Runtime | |
| Peak VRAM | |
| Failure log (required on FAIL) | |

Recommended attachments: `rocminfo`, `hipconfig --full`, `uname -a`, and
`torch.cuda.get_device_properties(0)`.
