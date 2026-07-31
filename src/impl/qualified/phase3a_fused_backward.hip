// P3A-BWD-NATIVE-001: gfx1201 rocWMMA backward and fixed-order dW reduction.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t WIDTH = 64, TILE = 16, WAVE = 32, WAVES = 4;
constexpr uint32_t THREADS = WAVE * WAVES, SLOTS = 8;
using Half = rocwmma::float16_t;
using F32 = rocwmma::float32_t;
using FragA = rocwmma::fragment<
    rocwmma::matrix_a, TILE, TILE, TILE, Half, rocwmma::row_major>;
using FragB = rocwmma::fragment<
    rocwmma::matrix_b, TILE, TILE, TILE, Half, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<
    rocwmma::accumulator, TILE, TILE, TILE, F32>;
using RegA = rocwmma::apply_register_file_t<FragA>;
using RegB = rocwmma::apply_register_file_t<FragB>;

union HalfBits { Half value; uint16_t bits; };
__host__ __device__ uint16_t half_bits(Half value) {
    HalfBits u{}; u.value = value; return u.bits;
}

#define HIP_CHECK(call) do { \
    hipError_t e_ = (call); \
    if (e_ != hipSuccess) { \
        std::cerr << "P3A-PROBE-HIP-ERROR: " << #call << ": " \
                  << hipGetErrorString(e_) << "\n"; std::exit(2); \
    } \
} while (false)

void lt_check(hipblasStatus_t status, const char* what) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "P3A-PROBE-HIPBLASLT-ERROR: " << what << ": "
                  << static_cast<int>(status) << "\n";
        std::exit(3);
    }
}

void lt_run(
    hipblasLtHandle_t handle, const Half* a, uint64_t ar, uint64_t ac,
    hipblasOperation_t trans_a, const Half* b, uint64_t br, uint64_t bc,
    hipblasOperation_t trans_b, void* output, uint64_t cr, uint64_t cc,
    hipDataType output_type) {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    lt_check(hipblasLtMatmulDescCreate(
        &desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), "desc");
    lt_check(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)), "transa");
    lt_check(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)), "transb");
    lt_check(hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, ar, ac, ar), "layout a");
    lt_check(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, br, bc, br), "layout b");
    lt_check(hipblasLtMatrixLayoutCreate(&lc, output_type, cr, cc, cr), "layout c");
    lt_check(hipblasLtMatrixLayoutCreate(&ld, output_type, cr, cc, cr), "layout d");
    lt_check(hipblasLtMatmulPreferenceCreate(&pref), "preference");
    uint64_t workspace = 0;
    lt_check(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace, sizeof(workspace)), "workspace");
    hipblasLtMatmulHeuristicResult_t candidates[32]{};
    int count = 0;
    lt_check(hipblasLtMatmulAlgoGetHeuristic(
        handle, desc, la, lb, lc, ld, pref, 32, candidates, &count), "heuristic");
    int selected = -1;
    for (int i = 0; i < count; ++i)
        if (candidates[i].state == HIPBLAS_STATUS_SUCCESS
            && candidates[i].workspaceSize == 0) { selected = i; break; }
    if (selected < 0) {
        std::cerr << "P3A-GATE no zero-workspace hipBLASLt plan\n"; std::exit(4);
    }
    const float alpha = 1.0f, beta = 0.0f;
    lt_check(hipblasLtMatmul(
        handle, desc, &alpha, a, la, b, lb, &beta, output, lc, output, ld,
        &candidates[selected].algo, nullptr, 0, nullptr), "matmul");
    HIP_CHECK(hipDeviceSynchronize());
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(ld); hipblasLtMatrixLayoutDestroy(lc);
    hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(la);
    hipblasLtMatmulDescDestroy(desc);
}

struct Header {
    uint32_t magic, rows, tile_count, witness_entries;
};

__global__ void phase3a_forward_saved_kernel(
    const Half* x, const Half* w0, const Half* w1, const Half* w2,
    uint32_t rows, float* z0, Half* h0, float* z1, Half* h1, Half* y) {
    __shared__ Half saved0[WIDTH];
    __shared__ Half saved1[WIDTH];
    uint32_t r = blockIdx.x, n = threadIdx.x;
    if (r >= rows || n >= WIDTH) return;
    uint32_t i = r * WIDTH + n;
    float a0 = 0.0f;
    for (uint32_t k = 0; k < WIDTH; ++k)
        a0 += static_cast<float>(x[r * WIDTH + k])
            * static_cast<float>(w0[k * WIDTH + n]);
    z0[i] = a0;
    h0[i] = static_cast<Half>(a0 > 0.0f ? a0 : 0.0f);
    saved0[n] = h0[i];
    __syncthreads();
    float a1 = 0.0f;
    for (uint32_t k = 0; k < WIDTH; ++k)
        a1 += static_cast<float>(saved0[k])
            * static_cast<float>(w1[k * WIDTH + n]);
    z1[i] = a1;
    h1[i] = static_cast<Half>(a1 > 0.0f ? a1 : 0.0f);
    saved1[n] = h1[i];
    __syncthreads();
    float a2 = 0.0f;
    for (uint32_t k = 0; k < WIDTH; ++k)
        a2 += static_cast<float>(saved1[k])
            * static_cast<float>(w2[k * WIDTH + n]);
    y[i] = static_cast<Half>(a2);
}

__device__ void store_half_acc(
    FragAcc const& acc, Half* output, uint32_t n_begin, const float* gate) {
    uint32_t lane = threadIdx.x % WAVE;
    auto const& reg = rocwmma::to_register_file(acc);
    for (uint32_t slot = 0; slot < SLOTS; ++slot) {
        uint32_t row = slot + (lane >= 16 ? 8 : 0);
        uint32_t col = lane & 15;
        float v = reg[slot];
        if (gate && gate[row * WIDTH + n_begin + col] <= 0.0f) v = 0.0f;
        output[row * WIDTH + n_begin + col] = static_cast<Half>(v);
    }
}

__device__ void store_float_acc(
    FragAcc const& acc, float* output, uint32_t n_begin) {
    uint32_t lane = threadIdx.x % WAVE;
    auto const& reg = rocwmma::to_register_file(acc);
    for (uint32_t slot = 0; slot < SLOTS; ++slot) {
        uint32_t row = slot + (lane >= 16 ? 8 : 0);
        uint32_t col = lane & 15;
        output[row * WIDTH + n_begin + col] = reg[slot];
    }
}

__device__ void capture_a(FragA const& f, uint16_t* output) {
    uint32_t lane = threadIdx.x % WAVE;
    if (threadIdx.x / WAVE != 0 || blockIdx.x != 0) return;
    auto const& reg = rocwmma::to_register_file(f);
    for (uint32_t slot = 0; slot < SLOTS; ++slot)
        output[lane * SLOTS + slot] = half_bits(reg[slot]);
}

__device__ void capture_b(FragB const& f, uint16_t* output) {
    uint32_t lane = threadIdx.x % WAVE;
    if (threadIdx.x / WAVE != 0 || blockIdx.x != 0) return;
    auto const& reg = rocwmma::to_register_file(f);
    for (uint32_t slot = 0; slot < SLOTS; ++slot)
        output[lane * SLOTS + slot] = half_bits(reg[slot]);
}

__device__ void prepare_dw_operands(
    const Half* left, const Half* right, uint32_t rows, uint32_t row0,
    Half* left_t, Half* right_t) {
    for (uint32_t i = threadIdx.x; i < WIDTH * TILE; i += THREADS) {
        uint32_t feature = i / TILE, local_row = i % TILE;
        uint32_t global_row = row0 + local_row;
        left_t[i] = global_row < rows
            ? left[global_row * WIDTH + feature] : static_cast<Half>(0.0f);
        right_t[i] = global_row < rows
            ? right[global_row * WIDTH + feature] : static_cast<Half>(0.0f);
    }
    __syncthreads();
}

__device__ void compute_dw_layer(
    const Half* left, const Half* right, uint32_t rows, uint32_t row0,
    Half* left_t, Half* right_t, float* partial,
    uint16_t* witness_a, uint16_t* witness_b) {
    uint32_t wave = threadIdx.x / WAVE;
    uint32_t n_begin = wave * TILE;
    prepare_dw_operands(left, right, rows, row0, left_t, right_t);
    for (uint32_t mt = 0; mt < 4; ++mt) {
        FragA a; FragB b; FragAcc acc;
        rocwmma::fill_fragment(acc, 0.0f);
        rocwmma::load_matrix_sync(a, left_t + mt * TILE * TILE, TILE);
        rocwmma::load_matrix_sync(b, right_t + n_begin * TILE, TILE);
        rocwmma::mma_sync(acc, a, b, acc);
        if (mt == 0) {
            if (witness_a) capture_a(a, witness_a);
            if (witness_b) capture_b(b, witness_b);
        }
        rocwmma::store_matrix_sync(
            partial + mt * TILE * WIDTH + n_begin, acc, WIDTH,
            rocwmma::mem_row_major);
    }
    __syncthreads();
}

__global__ void phase3a_backward_tile_kernel(
    const Half* x, const Half* w0, const Half* w1, const Half* w2,
    const float* z0_all, const Half* h0_all,
    const float* z1_all, const Half* h1_all, const Half* dy_all,
    uint32_t rows, Half* dz1_all, Half* dz0_all, float* dx_all,
    float* partials, uint16_t* witnesses) {
    __shared__ __align__(16) Half gradient[TILE * WIDTH];
    __shared__ __align__(16) Half left_t[WIDTH * TILE];
    __shared__ __align__(16) Half right_t[WIDTH * TILE];
    uint32_t row0 = blockIdx.x * TILE;
    uint32_t wave = threadIdx.x / WAVE;
    uint32_t n_begin = wave * TILE;
    uint32_t valid = row0 < rows ? min(TILE, rows - row0) : 0;
    const float* z1 = z1_all + row0 * WIDTH;
    const float* z0 = z0_all + row0 * WIDTH;
    Half* dz1 = dz1_all + row0 * WIDTH;
    Half* dz0 = dz0_all + row0 * WIDTH;
    float* dx = dx_all + row0 * WIDTH;
    const Half* dy = dy_all + row0 * WIDTH;
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(RegA::size() == SLOTS);
    static_assert(RegB::size() == SLOTS);
#endif
    for (uint32_t i = threadIdx.x; i < TILE * WIDTH; i += THREADS)
        gradient[i] = i / WIDTH < valid ? dy[i] : static_cast<Half>(0.0f);
    __syncthreads();

    FragAcc dx2; rocwmma::fill_fragment(dx2, 0.0f);
    FragA dx2_a_first; FragB dx2_b_first;
    // P3A-BWD-LAYER2-DX-MMA-BEGIN
    for (uint32_t kt = 0; kt < 4; ++kt) {
        FragA a; FragB b;
        rocwmma::load_matrix_sync(a, gradient + kt * TILE, WIDTH);
        rocwmma::load_matrix_sync(b, w2 + n_begin * WIDTH + kt * TILE, WIDTH);
        rocwmma::mma_sync(dx2, a, b, dx2);
        if (kt == 0) { dx2_a_first = a; dx2_b_first = b; }
    }
    // P3A-BWD-LAYER2-DX-MMA-END
    // P3A-BWD-LAYER2-RELU-GATE-BEGIN
    store_half_acc(dx2, dz1, n_begin, z1);
    // P3A-BWD-LAYER2-RELU-GATE-END
    if (wave == 0 && blockIdx.x == 0) {
        capture_a(dx2_a_first, witnesses + 0 * 256);
        capture_b(dx2_b_first, witnesses + 1 * 256);
    }
    __syncthreads();
    // P3A-BWD-LAYER2-DW-MMA-BEGIN
    compute_dw_layer(
        h1_all, dy_all, rows, row0, left_t, right_t,
        partials + (2 * gridDim.x + blockIdx.x) * WIDTH * WIDTH,
        witnesses + 2 * 256, witnesses + 3 * 256);
    // P3A-BWD-LAYER2-DW-MMA-END
    // P3A-BWD-BRIDGE21-LDS-STORE-BEGIN
    for (uint32_t i = threadIdx.x; i < TILE * WIDTH; i += THREADS)
        gradient[i] = i / WIDTH < valid ? dz1[i] : static_cast<Half>(0.0f);
    // P3A-BWD-BRIDGE21-LDS-STORE-END
    // P3A-BWD-BRIDGE21-BARRIER
    __syncthreads();
    FragA bridge21[4];
    // P3A-BWD-BRIDGE21-LDS-LOAD-BEGIN
    for (uint32_t kt = 0; kt < 4; ++kt)
        rocwmma::load_matrix_sync(bridge21[kt], gradient + kt * TILE, WIDTH);
    // P3A-BWD-BRIDGE21-LDS-LOAD-END

    FragAcc dx1; rocwmma::fill_fragment(dx1, 0.0f);
    // P3A-BWD-LAYER1-DX-MMA-BEGIN
    for (uint32_t kt = 0; kt < 4; ++kt) {
        FragB b;
        rocwmma::load_matrix_sync(b, w1 + n_begin * WIDTH + kt * TILE, WIDTH);
        rocwmma::mma_sync(dx1, bridge21[kt], b, dx1);
    }
    // P3A-BWD-LAYER1-DX-MMA-END
    // P3A-BWD-LAYER1-RELU-GATE-BEGIN
    store_half_acc(dx1, dz0, n_begin, z0);
    // P3A-BWD-LAYER1-RELU-GATE-END
    __syncthreads();
    // P3A-BWD-LAYER1-DW-MMA-BEGIN
    compute_dw_layer(
        h0_all, dz1_all, rows, row0, left_t, right_t,
        partials + (1 * gridDim.x + blockIdx.x) * WIDTH * WIDTH,
        nullptr, nullptr);
    // P3A-BWD-LAYER1-DW-MMA-END
    // P3A-BWD-BRIDGE10-LDS-STORE-BEGIN
    for (uint32_t i = threadIdx.x; i < TILE * WIDTH; i += THREADS)
        gradient[i] = i / WIDTH < valid ? dz0[i] : static_cast<Half>(0.0f);
    // P3A-BWD-BRIDGE10-LDS-STORE-END
    // P3A-BWD-BRIDGE10-BARRIER
    __syncthreads();
    FragA bridge10[4];
    // P3A-BWD-BRIDGE10-LDS-LOAD-BEGIN
    for (uint32_t kt = 0; kt < 4; ++kt)
        rocwmma::load_matrix_sync(bridge10[kt], gradient + kt * TILE, WIDTH);
    // P3A-BWD-BRIDGE10-LDS-LOAD-END

    FragAcc dx0; rocwmma::fill_fragment(dx0, 0.0f);
    // P3A-BWD-LAYER0-DX-MMA-BEGIN
    for (uint32_t kt = 0; kt < 4; ++kt) {
        FragB b;
        rocwmma::load_matrix_sync(b, w0 + n_begin * WIDTH + kt * TILE, WIDTH);
        rocwmma::mma_sync(dx0, bridge10[kt], b, dx0);
    }
    // P3A-BWD-LAYER0-DX-MMA-END
    store_float_acc(dx0, dx, n_begin);
    __syncthreads();
    // P3A-BWD-LAYER0-DW-MMA-BEGIN
    compute_dw_layer(
        x, dz0_all, rows, row0, left_t, right_t,
        partials + (0 * gridDim.x + blockIdx.x) * WIDTH * WIDTH,
        nullptr, nullptr);
    // P3A-BWD-LAYER0-DW-MMA-END
    // P3A-BWD-PARTIAL-STORE-BEGIN
    // rocWMMA stores above are the sole partial-gradient publication path.
    // P3A-BWD-PARTIAL-STORE-END
}

__global__ void phase3a_dw_fixed_order_reduce_kernel(
    const float* partials, uint32_t tile_count, float* final_dw) {
    uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= 3 * WIDTH * WIDTH) return;
    uint32_t layer = element / (WIDTH * WIDTH);
    uint32_t offset = element % (WIDTH * WIDTH);
    float acc = 0.0f;
    // P3A-DW-REDUCE-BEGIN
    // P3A-DW-REDUCE-FIXED-ORDER
    for (uint32_t batch_tile = 0; batch_tile < tile_count; ++batch_tile)
        acc += partials[(layer * tile_count + batch_tile) * WIDTH * WIDTH + offset];
    // P3A-DW-REDUCE-END
    final_dw[element] = acc;
}

__global__ void phase3a_relu_gate_kernel(
    const Half* input, const float* preactivation, uint32_t count, Half* output) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = preactivation[i] > 0.0f ? input[i] : static_cast<Half>(0.0f);
}

float weight_value(uint32_t set, uint32_t layer, uint32_t k, uint32_t n) {
    if (k == n) return 0.5f + layer / 16.0f + set / 32.0f;
    uint32_t q = 1 + ((k * (3 + set) + n * (5 + layer)
        + layer * 7 + set * 11) % 4);
    return static_cast<float>(q) / 1024.0f;
}

float input_value(const std::string& id, uint32_t r, uint32_t c) {
    if (id.find("sparse") != std::string::npos)
        return c == ((r * 5 + 3) & 63) ? 0.75f : 0.0f;
    if (id.find("partial") != std::string::npos)
        return ((r * 3 + c) & 1) ? -0.375f : 0.5f;
    return ((r + c) & 1) ? -0.25f : 0.375f;
}

float gradient_value(const std::string& id, uint32_t r, uint32_t c) {
    if (id.find("sparse") != std::string::npos)
        return ((r * 7 + c * 3) % 11 == 0) ? 0.5f : 0.0f;
    if (id.find("partial") != std::string::npos)
        return ((r + c * 3) & 1) ? -0.3125f : 0.4375f;
    if (id.find("set_b") != std::string::npos)
        return ((r * 3 + c) & 1) ? -0.1875f : 0.3125f;
    return ((r + c) & 1) ? -0.25f : 0.375f;
}

template <class T>
void write_vector(std::ofstream& out, const std::vector<T>& v) {
    out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: phase3a_fused_backward CASE OUTPUT\n"; return 2;
    }
    std::string id = argv[1];
    uint32_t rows = id.find("_m32") != std::string::npos ? 32
        : id.find("_m48") != std::string::npos ? 48
        : id.find("_m45") != std::string::npos ? 45 : 64;
    uint32_t tiles = (rows + TILE - 1) / TILE;
    uint32_t set = id.find("set_b") != std::string::npos ? 1 : 0;
    std::vector<Half> x(rows * WIDTH), dy(rows * WIDTH);
    std::array<std::vector<Half>, 3> w, wp;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < WIDTH; ++c) {
            x[r * WIDTH + c] = static_cast<Half>(input_value(id, r, c));
            dy[r * WIDTH + c] = static_cast<Half>(gradient_value(id, r, c));
        }
    for (uint32_t l = 0; l < 3; ++l) {
        w[l].resize(WIDTH * WIDTH);
        wp[l].resize(WIDTH * WIDTH);
        for (uint32_t k = 0; k < WIDTH; ++k)
            for (uint32_t n = 0; n < WIDTH; ++n)
                w[l][k * WIDTH + n] = wp[l][n * WIDTH + k] =
                    static_cast<Half>(weight_value(set, l, k, n));
    }
    Half *d_x = nullptr, *d_dy = nullptr, *d_w[3]{}, *d_wp[3]{};
    Half *d_h0 = nullptr, *d_h1 = nullptr, *d_y = nullptr;
    Half *d_dz1 = nullptr, *d_dz0 = nullptr;
    float *d_z0 = nullptr, *d_z1 = nullptr, *d_dx = nullptr;
    float *d_parts = nullptr, *d_final = nullptr;
    Half *d_hb_dh1 = nullptr, *d_hb_dz1 = nullptr;
    Half *d_hb_dh0 = nullptr, *d_hb_dz0 = nullptr;
    float *d_hb_dx = nullptr, *d_hb_dw[3]{};
    uint16_t* d_witness = nullptr;
    size_t half_matrix = rows * WIDTH * sizeof(Half);
    size_t float_matrix = rows * WIDTH * sizeof(float);
    HIP_CHECK(hipMalloc(&d_x, half_matrix)); HIP_CHECK(hipMalloc(&d_dy, half_matrix));
    HIP_CHECK(hipMemcpy(d_x, x.data(), half_matrix, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dy, dy.data(), half_matrix, hipMemcpyHostToDevice));
    for (uint32_t l = 0; l < 3; ++l) {
        HIP_CHECK(hipMalloc(&d_w[l], WIDTH * WIDTH * sizeof(Half)));
        HIP_CHECK(hipMalloc(&d_wp[l], WIDTH * WIDTH * sizeof(Half)));
        HIP_CHECK(hipMemcpy(d_w[l], w[l].data(), WIDTH * WIDTH * sizeof(Half),
            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_wp[l], wp[l].data(), WIDTH * WIDTH * sizeof(Half),
            hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMalloc(&d_h0, half_matrix)); HIP_CHECK(hipMalloc(&d_h1, half_matrix));
    HIP_CHECK(hipMalloc(&d_y, half_matrix)); HIP_CHECK(hipMalloc(&d_dz1, half_matrix));
    HIP_CHECK(hipMalloc(&d_dz0, half_matrix)); HIP_CHECK(hipMalloc(&d_z0, float_matrix));
    HIP_CHECK(hipMalloc(&d_z1, float_matrix)); HIP_CHECK(hipMalloc(&d_dx, float_matrix));
    HIP_CHECK(hipMalloc(&d_parts, 3 * tiles * WIDTH * WIDTH * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_final, 3 * WIDTH * WIDTH * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_witness, 4 * 256 * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_hb_dh1, half_matrix));
    HIP_CHECK(hipMalloc(&d_hb_dz1, half_matrix));
    HIP_CHECK(hipMalloc(&d_hb_dh0, half_matrix));
    HIP_CHECK(hipMalloc(&d_hb_dz0, half_matrix));
    HIP_CHECK(hipMalloc(&d_hb_dx, float_matrix));
    for (uint32_t l = 0; l < 3; ++l)
        HIP_CHECK(hipMalloc(&d_hb_dw[l], WIDTH * WIDTH * sizeof(float)));
    HIP_CHECK(hipMemset(d_parts, 0, 3 * tiles * WIDTH * WIDTH * sizeof(float)));
    HIP_CHECK(hipMemset(d_witness, 0, 4 * 256 * sizeof(uint16_t)));
    hipLaunchKernelGGL(phase3a_forward_saved_kernel,
        dim3(rows), dim3(WIDTH), 0, nullptr,
        d_x, d_w[0], d_w[1], d_w[2], rows, d_z0, d_h0, d_z1, d_h1, d_y);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());

    hipblasLtHandle_t handle = nullptr;
    lt_check(hipblasLtCreate(&handle), "create handle");
    // Column-major views: transposed physical W times row-major tensor view.
    lt_run(handle, d_wp[2], WIDTH, WIDTH, HIPBLAS_OP_N,
           d_dy, WIDTH, rows, HIPBLAS_OP_N,
           d_hb_dh1, WIDTH, rows, HIP_R_16F);
    hipLaunchKernelGGL(phase3a_relu_gate_kernel,
        dim3((rows * WIDTH + 255) / 256), dim3(256), 0, nullptr,
        d_hb_dh1, d_z1, rows * WIDTH, d_hb_dz1);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    lt_run(handle, d_wp[1], WIDTH, WIDTH, HIPBLAS_OP_N,
           d_hb_dz1, WIDTH, rows, HIPBLAS_OP_N,
           d_hb_dh0, WIDTH, rows, HIP_R_16F);
    hipLaunchKernelGGL(phase3a_relu_gate_kernel,
        dim3((rows * WIDTH + 255) / 256), dim3(256), 0, nullptr,
        d_hb_dh0, d_z0, rows * WIDTH, d_hb_dz0);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    lt_run(handle, d_wp[0], WIDTH, WIDTH, HIPBLAS_OP_N,
           d_hb_dz0, WIDTH, rows, HIPBLAS_OP_N,
           d_hb_dx, WIDTH, rows, HIP_R_32F);
    // C memory is row-major dW because the column-major result is dW^T.
    lt_run(handle, d_hb_dz0, WIDTH, rows, HIPBLAS_OP_N,
           d_x, WIDTH, rows, HIPBLAS_OP_T,
           d_hb_dw[0], WIDTH, WIDTH, HIP_R_32F);
    lt_run(handle, d_hb_dz1, WIDTH, rows, HIPBLAS_OP_N,
           d_h0, WIDTH, rows, HIPBLAS_OP_T,
           d_hb_dw[1], WIDTH, WIDTH, HIP_R_32F);
    lt_run(handle, d_dy, WIDTH, rows, HIPBLAS_OP_N,
           d_h1, WIDTH, rows, HIPBLAS_OP_T,
           d_hb_dw[2], WIDTH, WIDTH, HIP_R_32F);
    hipblasLtDestroy(handle);
    hipLaunchKernelGGL(phase3a_backward_tile_kernel,
        dim3(tiles), dim3(THREADS), 0, nullptr,
        d_x, d_w[0], d_w[1], d_w[2], d_z0, d_h0, d_z1, d_h1, d_dy,
        rows, d_dz1, d_dz0, d_dx, d_parts, d_witness);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    hipLaunchKernelGGL(phase3a_dw_fixed_order_reduce_kernel,
        dim3((3 * WIDTH * WIDTH + 255) / 256), dim3(256), 0, nullptr,
        d_parts, tiles, d_final);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> z0(rows * WIDTH), z1(rows * WIDTH), dx(rows * WIDTH);
    std::vector<float> parts(3 * tiles * WIDTH * WIDTH), final_dw(3 * WIDTH * WIDTH);
    std::vector<Half> h0(rows * WIDTH), h1(rows * WIDTH), y(rows * WIDTH);
    std::vector<Half> dz1(rows * WIDTH), dz0(rows * WIDTH);
    std::vector<uint16_t> witness(4 * 256);
    std::vector<float> hb_dx(rows * WIDTH), hb_dw(3 * WIDTH * WIDTH);
    HIP_CHECK(hipMemcpy(z0.data(), d_z0, float_matrix, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(z1.data(), d_z1, float_matrix, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dx.data(), d_dx, float_matrix, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h0.data(), d_h0, half_matrix, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h1.data(), d_h1, half_matrix, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(y.data(), d_y, half_matrix, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dz1.data(), d_dz1, half_matrix, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dz0.data(), d_dz0, half_matrix, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(parts.data(), d_parts, parts.size() * sizeof(float),
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(final_dw.data(), d_final, final_dw.size() * sizeof(float),
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(witness.data(), d_witness, witness.size() * sizeof(uint16_t),
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hb_dx.data(), d_hb_dx, float_matrix, hipMemcpyDeviceToHost));
    for (uint32_t l = 0; l < 3; ++l)
        HIP_CHECK(hipMemcpy(hb_dw.data() + l * WIDTH * WIDTH, d_hb_dw[l],
            WIDTH * WIDTH * sizeof(float), hipMemcpyDeviceToHost));
    for (auto p : d_w) HIP_CHECK(hipFree(p));
    for (auto p : d_wp) HIP_CHECK(hipFree(p));
    HIP_CHECK(hipFree(d_x)); HIP_CHECK(hipFree(d_dy)); HIP_CHECK(hipFree(d_h0));
    HIP_CHECK(hipFree(d_h1)); HIP_CHECK(hipFree(d_y)); HIP_CHECK(hipFree(d_dz1));
    HIP_CHECK(hipFree(d_dz0)); HIP_CHECK(hipFree(d_z0)); HIP_CHECK(hipFree(d_z1));
    HIP_CHECK(hipFree(d_dx)); HIP_CHECK(hipFree(d_parts)); HIP_CHECK(hipFree(d_final));
    HIP_CHECK(hipFree(d_witness));
    HIP_CHECK(hipFree(d_hb_dh1)); HIP_CHECK(hipFree(d_hb_dz1));
    HIP_CHECK(hipFree(d_hb_dh0)); HIP_CHECK(hipFree(d_hb_dz0));
    HIP_CHECK(hipFree(d_hb_dx));
    for (auto p : d_hb_dw) HIP_CHECK(hipFree(p));

    std::ofstream out(argv[2], std::ios::binary);
    Header header{0x50334131u, rows, tiles, 256};
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    write_vector(out, z0); write_vector(out, z1);
    write_vector(out, h0); write_vector(out, h1); write_vector(out, y);
    write_vector(out, dz1); write_vector(out, dz0); write_vector(out, dx);
    write_vector(out, parts); write_vector(out, final_dw); write_vector(out, witness);
    write_vector(out, hb_dx); write_vector(out, hb_dw);
    if (!out) return 6;
    std::cout << "P3A-PROBE-HARDWARE: PASS\n";
    return 0;
}
