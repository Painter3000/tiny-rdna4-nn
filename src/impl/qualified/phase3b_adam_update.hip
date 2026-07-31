// P3B-TRAIN-NATIVE-001: separate deterministic Adam V1 update for gfx1201.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

extern "C" __global__ void phase3b_adam_fp32_master_update_kernel(
    float* master, __half* compute, float* exp_avg, float* exp_avg_sq,
    const float* gradient, uint32_t count, float beta1, float beta2,
    float beta1_power, float beta2_power, float epsilon, float learning_rate) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float g = gradient[i];
    // P3B-ADAM-MOMENT1-BEGIN
    float m = beta1 * exp_avg[i] + (1.0f - beta1) * g;
    // P3B-ADAM-MOMENT1-END
    // P3B-ADAM-MOMENT2-BEGIN
    float v = beta2 * exp_avg_sq[i] + (1.0f - beta2) * (g * g);
    // P3B-ADAM-MOMENT2-END
    // P3B-ADAM-BIAS-CORRECTION-BEGIN
    float m_hat = m / (1.0f - beta1_power);
    float v_hat = v / (1.0f - beta2_power);
    float denom = sqrtf(v_hat) + epsilon;
    // P3B-ADAM-BIAS-CORRECTION-END
    // P3B-ADAM-WEIGHT-UPDATE-BEGIN
    float update = learning_rate * (m_hat / denom);
    float next = master[i] - update;
    master[i] = next;
    exp_avg[i] = m;
    exp_avg_sq[i] = v;
    // P3B-ADAM-WEIGHT-UPDATE-END
    // P3B-ADAM-FP16-CAST-BEGIN
    compute[i] = __float2half(next);
    // P3B-ADAM-FP16-CAST-END
}

#define HIP_CHECK(call) do { \
    hipError_t e_ = (call); \
    if (e_ != hipSuccess) { \
        std::cerr << "P3B-PROBE-HIP-ERROR: " << hipGetErrorString(e_) << "\n"; \
        std::exit(2); \
    } \
} while (false)

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: phase3b_adam INPUT OUTPUT BETA1_POWER BETA2_POWER\n";
        return 2;
    }
    constexpr uint32_t count = 3 * 64 * 64;
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<float> host(4 * count);
    in.read(reinterpret_cast<char*>(host.data()), host.size() * sizeof(float));
    if (!in || in.peek() != std::ifstream::traits_type::eof()) return 3;
    float *master = nullptr, *m = nullptr, *v = nullptr, *g = nullptr;
    __half* compute = nullptr;
    size_t bytes = count * sizeof(float);
    HIP_CHECK(hipMalloc(&master, bytes)); HIP_CHECK(hipMalloc(&m, bytes));
    HIP_CHECK(hipMalloc(&v, bytes)); HIP_CHECK(hipMalloc(&g, bytes));
    HIP_CHECK(hipMalloc(&compute, count * sizeof(__half)));
    HIP_CHECK(hipMemcpy(master, host.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(m, host.data() + count, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v, host.data() + 2 * count, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g, host.data() + 3 * count, bytes, hipMemcpyHostToDevice));
    float b1p = std::strtof(argv[3], nullptr), b2p = std::strtof(argv[4], nullptr);
    hipLaunchKernelGGL(phase3b_adam_fp32_master_update_kernel,
        dim3((count + 255) / 256), dim3(256), 0, nullptr,
        master, compute, m, v, g, count, 0.9f, 0.999f, b1p, b2p, 1.0e-8f, 1.0e-3f);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint16_t> half_bits(count);
    HIP_CHECK(hipMemcpy(host.data(), master, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host.data() + count, m, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host.data() + 2 * count, v, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(half_bits.data(), compute, count * sizeof(uint16_t),
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(master)); HIP_CHECK(hipFree(compute)); HIP_CHECK(hipFree(m));
    HIP_CHECK(hipFree(v)); HIP_CHECK(hipFree(g));
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(host.data()), 3 * bytes);
    out.write(reinterpret_cast<const char*>(half_bits.data()),
              half_bits.size() * sizeof(uint16_t));
    if (!out) return 4;
    std::cout << "P3B-ADAM-HARDWARE: PASS\n";
    return 0;
}
