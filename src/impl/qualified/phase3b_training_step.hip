// P3B-TRAIN-STEP-001: dynamic-weight wrapper around the frozen Phase-3A path.
#define main phase3a_frozen_standalone_main
#include "../../../repos/tiny-cuda-nn-rocwmma-probe/phase3a_fused_backward/impl/native/phase3a_fused_backward.hip"
#undef main

namespace {
struct StepHeader {
    uint32_t magic;
    uint32_t rows;
};

template<class T>
void read_exact(std::ifstream& in, std::vector<T>& value) {
    in.read(reinterpret_cast<char*>(value.data()), value.size() * sizeof(T));
    if (!in) { std::cerr << "P3B-TRAIN input truncated\n"; std::exit(7); }
}

template<class T>
void write_exact(std::ofstream& out, const std::vector<T>& value) {
    out.write(reinterpret_cast<const char*>(value.data()), value.size() * sizeof(T));
}
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: phase3b_training_step INPUT OUTPUT\n";
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    StepHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || header.magic != 0x50334231u || header.rows == 0 || header.rows > 64)
        return 3;
    uint32_t rows = header.rows, tiles = (rows + TILE - 1) / TILE;
    std::vector<Half> x(rows * WIDTH), target(rows * WIDTH);
    std::array<std::vector<Half>, 3> w, wp;
    read_exact(in, x); read_exact(in, target);
    for (uint32_t l = 0; l < 3; ++l) {
        w[l].resize(WIDTH * WIDTH); wp[l].resize(WIDTH * WIDTH);
        read_exact(in, w[l]);
        for (uint32_t k = 0; k < WIDTH; ++k)
            for (uint32_t n = 0; n < WIDTH; ++n)
                wp[l][n * WIDTH + k] = w[l][k * WIDTH + n];
    }
    if (in.peek() != std::ifstream::traits_type::eof()) return 4;

    Half *d_x = nullptr, *d_target = nullptr, *d_dy = nullptr;
    Half *d_w[3]{}, *d_wp[3]{}, *d_h0 = nullptr, *d_h1 = nullptr, *d_y = nullptr;
    Half *d_dz1 = nullptr, *d_dz0 = nullptr;
    float *d_z0 = nullptr, *d_z1 = nullptr, *d_dx = nullptr;
    float *d_parts = nullptr, *d_final = nullptr;
    uint16_t* d_witness = nullptr;
    size_t hb = rows * WIDTH * sizeof(Half), fb = rows * WIDTH * sizeof(float);
    HIP_CHECK(hipMalloc(&d_x, hb)); HIP_CHECK(hipMalloc(&d_target, hb));
    HIP_CHECK(hipMalloc(&d_dy, hb));
    HIP_CHECK(hipMemcpy(d_x, x.data(), hb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_target, target.data(), hb, hipMemcpyHostToDevice));
    for (uint32_t l = 0; l < 3; ++l) {
        HIP_CHECK(hipMalloc(&d_w[l], WIDTH * WIDTH * sizeof(Half)));
        HIP_CHECK(hipMalloc(&d_wp[l], WIDTH * WIDTH * sizeof(Half)));
        HIP_CHECK(hipMemcpy(d_w[l], w[l].data(), WIDTH * WIDTH * sizeof(Half),
            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_wp[l], wp[l].data(), WIDTH * WIDTH * sizeof(Half),
            hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMalloc(&d_h0, hb)); HIP_CHECK(hipMalloc(&d_h1, hb));
    HIP_CHECK(hipMalloc(&d_y, hb)); HIP_CHECK(hipMalloc(&d_dz1, hb));
    HIP_CHECK(hipMalloc(&d_dz0, hb)); HIP_CHECK(hipMalloc(&d_z0, fb));
    HIP_CHECK(hipMalloc(&d_z1, fb)); HIP_CHECK(hipMalloc(&d_dx, fb));
    HIP_CHECK(hipMalloc(&d_parts, 3 * tiles * WIDTH * WIDTH * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_final, 3 * WIDTH * WIDTH * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_witness, 4 * 256 * sizeof(uint16_t)));
    HIP_CHECK(hipMemset(d_parts, 0, 3 * tiles * WIDTH * WIDTH * sizeof(float)));
    HIP_CHECK(hipMemset(d_witness, 0, 4 * 256 * sizeof(uint16_t)));

    hipLaunchKernelGGL(phase3a_forward_saved_kernel, dim3(rows), dim3(WIDTH), 0,
        nullptr, d_x, d_w[0], d_w[1], d_w[2], rows,
        d_z0, d_h0, d_z1, d_h1, d_y);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    std::vector<Half> y(rows * WIDTH), dy(rows * WIDTH);
    HIP_CHECK(hipMemcpy(y.data(), d_y, hb, hipMemcpyDeviceToHost));
    float loss = 0.0f;
    for (uint32_t i = 0; i < rows * WIDTH; ++i) {
        float difference = static_cast<float>(y[i]) - static_cast<float>(target[i]);
        loss += 0.5f * 0.03125f * difference * difference;
        dy[i] = static_cast<Half>(0.03125f * difference);
    }
    HIP_CHECK(hipMemcpy(d_dy, dy.data(), hb, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(phase3a_backward_tile_kernel, dim3(tiles), dim3(THREADS),
        0, nullptr, d_x, d_w[0], d_w[1], d_w[2], d_z0, d_h0, d_z1, d_h1,
        d_dy, rows, d_dz1, d_dz0, d_dx, d_parts, d_witness);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    hipLaunchKernelGGL(phase3a_dw_fixed_order_reduce_kernel,
        dim3((3 * WIDTH * WIDTH + 255) / 256), dim3(256), 0, nullptr,
        d_parts, tiles, d_final);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> z0(rows * WIDTH), z1(rows * WIDTH), dx(rows * WIDTH);
    std::vector<float> final_dw(3 * WIDTH * WIDTH);
    std::vector<Half> h0(rows * WIDTH), h1(rows * WIDTH);
    std::vector<Half> dz1(rows * WIDTH), dz0(rows * WIDTH);
    HIP_CHECK(hipMemcpy(z0.data(), d_z0, fb, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(z1.data(), d_z1, fb, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dx.data(), d_dx, fb, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h0.data(), d_h0, hb, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h1.data(), d_h1, hb, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dz1.data(), d_dz1, hb, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dz0.data(), d_dz0, hb, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(final_dw.data(), d_final, final_dw.size() * sizeof(float),
        hipMemcpyDeviceToHost));

    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<char*>(&header), sizeof(header));
    out.write(reinterpret_cast<char*>(&loss), sizeof(loss));
    write_exact(out, z0); write_exact(out, z1); write_exact(out, h0);
    write_exact(out, h1); write_exact(out, y); write_exact(out, dy);
    write_exact(out, dz1); write_exact(out, dz0); write_exact(out, dx);
    write_exact(out, final_dw);
    for (auto p : d_w) HIP_CHECK(hipFree(p));
    for (auto p : d_wp) HIP_CHECK(hipFree(p));
    HIP_CHECK(hipFree(d_x)); HIP_CHECK(hipFree(d_target)); HIP_CHECK(hipFree(d_dy));
    HIP_CHECK(hipFree(d_h0)); HIP_CHECK(hipFree(d_h1)); HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_dz1)); HIP_CHECK(hipFree(d_dz0)); HIP_CHECK(hipFree(d_z0));
    HIP_CHECK(hipFree(d_z1)); HIP_CHECK(hipFree(d_dx)); HIP_CHECK(hipFree(d_parts));
    HIP_CHECK(hipFree(d_final)); HIP_CHECK(hipFree(d_witness));
    if (!out) return 5;
    std::cout << "P3B-TRAIN-STEP-HARDWARE: PASS\n";
    return 0;
}
