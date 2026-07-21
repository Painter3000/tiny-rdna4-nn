/* TCNN_RDNA4_P3A1_HIPBLASLT_001: unfused FP32 hipBLASLt MLP. */

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/networks/hipblaslt_mlp.h>

#include <hipblaslt/hipblaslt.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <unordered_map>

namespace tcnn {

void launch_fused_relu_backward(uint32_t width, hipStream_t stream, const float* upstream, const float* mask,
	float* dz, float* partial, float* db, uint32_t batch, GradientMode mode);

namespace {

// TCNN_RDNA4_P3A2_EPILOGUE_001: immutable epilogue plans and per-instance
// launch descriptors keep dynamic pointers out of the global cache.
std::atomic<uint64_t> g_cache_hits{0}, g_cache_misses{0};
std::atomic<uint64_t> g_bias_launches{0}, g_relu_bias_launches{0}, g_relu_aux_bias_launches{0};
std::atomic<uint64_t> g_epilogue_fallbacks{0}, g_post_kernel_launches{0};
// TCNN_RDNA4_P3A2A_STREAM_HANDLE_001: planning and per-stream execution
// handles are disjoint; counters are diagnostic and never affect dispatch.
std::atomic<uint64_t> g_execution_handle_creations{0}, g_execution_handle_reuses{0}, g_execution_handle_overflows{0};
std::atomic<uint64_t> g_epilogue_descriptor_count{0};
// TCNN_RDNA4_P3A4_FUSED_RELU_BGRAD_001: deterministic two-stage hidden
// ReLU derivative and bias-gradient reduction without final atomics.
std::atomic<uint64_t> g_fused_stage1_launches{0}, g_fused_relu_only_launches{0}, g_biasgrad_finalize_launches{0};
std::atomic<uint64_t> g_fused_backward_fallbacks{0}, g_legacy_activation_grad_launches{0}, g_legacy_bias_grad_launches{0};
std::atomic<uint64_t> g_fused_partial_bytes_live{0}, g_fused_partial_bytes_peak{0};

void check_lt(hipblasStatus_t status, const char* operation) {
	if (status != HIPBLAS_STATUS_SUCCESS) {
		throw std::runtime_error{fmt::format("HipBLASLtMLP {} failed with hipBLASLt status {}", operation, (int)status)};
	}
}

enum class GemmRole : uint8_t { Forward, InputGradient, WeightGradient };
enum class EpilogueKind : uint8_t { Default, Bias, ReluBias, ReluAuxBias };
struct MatmulKey {
	int device; uint32_t ar, ac, br, bc, cr, cc; bool ta, tb; GemmRole role; EpilogueKind epilogue; bool aux;
	bool operator==(const MatmulKey& o) const {
		return device==o.device && ar==o.ar && ac==o.ac && br==o.br && bc==o.bc && cr==o.cr && cc==o.cc && ta==o.ta && tb==o.tb && role==o.role && epilogue==o.epilogue && aux==o.aux;
	}
};
struct KeyHash {
	size_t operator()(const MatmulKey& k) const {
		size_t h = (size_t)k.device;
		for (uint32_t v : {k.ar,k.ac,k.br,k.bc,k.cr,k.cc}) h = h * 1099511628211ull ^ v;
		return h * 131u ^ ((size_t)k.ta << 5) ^ ((size_t)k.tb << 4) ^ ((size_t)k.role << 2) ^ (size_t)k.epilogue ^ ((size_t)k.aux << 7);
	}
};

hipblasLtEpilogue_t to_lt_epilogue(EpilogueKind kind) {
	if (kind == EpilogueKind::Bias) return HIPBLASLT_EPILOGUE_BIAS;
	if (kind == EpilogueKind::ReluBias) return HIPBLASLT_EPILOGUE_RELU_BIAS;
	if (kind == EpilogueKind::ReluAuxBias) return HIPBLASLT_EPILOGUE_RELU_AUX_BIAS;
	return HIPBLASLT_EPILOGUE_DEFAULT;
}

void configure_desc(hipblasLtMatmulDesc_t desc, const MatmulKey& key) {
	const hipblasOperation_t ta = key.ta ? HIPBLAS_OP_T : HIPBLAS_OP_N;
	const hipblasOperation_t tb = key.tb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
	check_lt(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "set TRANSA");
	check_lt(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "set TRANSB");
	if (key.epilogue != EpilogueKind::Default) {
		const auto epilogue = to_lt_epilogue(key.epilogue);
		check_lt(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)), "set EPILOGUE");
	}
}

struct MatmulPlan {
	hipblasLtMatmulDesc_t desc = nullptr;
	hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
	hipblasLtMatmulAlgo_t algorithm{};
	MatmulPlan(const MatmulKey& key) {
		check_lt(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), "MatmulDescCreate");
		configure_desc(desc, key);
		check_lt(hipblasLtMatrixLayoutCreate(&a, HIP_R_32F, key.ar, key.ac, key.ar), "A layout");
		check_lt(hipblasLtMatrixLayoutCreate(&b, HIP_R_32F, key.br, key.bc, key.br), "B layout");
		check_lt(hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, key.cr, key.cc, key.cr), "C layout");
		check_lt(hipblasLtMatrixLayoutCreate(&d, HIP_R_32F, key.cr, key.cc, key.cr), "D layout");
	}
	~MatmulPlan() {
		if (d) hipblasLtMatrixLayoutDestroy(d); if (c) hipblasLtMatrixLayoutDestroy(c);
		if (b) hipblasLtMatrixLayoutDestroy(b); if (a) hipblasLtMatrixLayoutDestroy(a);
		if (desc) hipblasLtMatmulDescDestroy(desc);
	}
};

struct ExecutionHandleEntry {
	hipStream_t stream = nullptr;
	hipblasLtHandle_t handle = nullptr;
	std::mutex submit_mutex;
};

struct DeviceContext {
	static constexpr size_t kExecutionHandleCapacity = 64;
	int device = -1;
	hipblasLtHandle_t planning_handle = nullptr;
	std::mutex mutex;
	std::unordered_map<MatmulKey, std::shared_ptr<MatmulPlan>, KeyHash> plans;
	std::unordered_map<uintptr_t, std::shared_ptr<ExecutionHandleEntry>> execution_handles;
	explicit DeviceContext(int device_) : device{device_} { check_lt(hipblasLtCreate(&planning_handle), "create planning handle"); }
	~DeviceContext() noexcept {
		int previous = -1; const bool restore = hipGetDevice(&previous) == hipSuccess;
		if (hipSetDevice(device) == hipSuccess) {
			plans.clear();
			for (auto& item : execution_handles) if (item.second->handle) { hipblasLtDestroy(item.second->handle); item.second->handle = nullptr; }
			execution_handles.clear();
			if (planning_handle) { hipblasLtDestroy(planning_handle); planning_handle = nullptr; }
		}
		if (restore && previous != device) hipSetDevice(previous);
	}
};

DeviceContext& device_context(int device) {
	struct Registry { std::mutex mutex; std::unordered_map<int, std::unique_ptr<DeviceContext>> devices; };
	static Registry registry;
	std::lock_guard<std::mutex> guard{registry.mutex};
	auto& result = registry.devices[device];
	if (!result) result = std::make_unique<DeviceContext>(device);
	return *result;
}

std::shared_ptr<ExecutionHandleEntry> execution_handle_for(int device, hipStream_t stream) {
	auto& context = device_context(device);
	std::lock_guard<std::mutex> guard{context.mutex};
	const uintptr_t key = reinterpret_cast<uintptr_t>(stream);
	auto found = context.execution_handles.find(key);
	if (found != context.execution_handles.end() && found->second->stream == stream) { ++g_execution_handle_reuses; return found->second; }
	if (context.execution_handles.size() >= DeviceContext::kExecutionHandleCapacity) {
		++g_execution_handle_overflows;
		throw std::runtime_error{fmt::format("HipBLASLtMLP execution-handle capacity exceeded: device={}, capacity={}, requested_stream={}", device, DeviceContext::kExecutionHandleCapacity, (const void*)stream)};
	}
	auto entry = std::make_shared<ExecutionHandleEntry>(); entry->stream = stream;
	check_lt(hipblasLtCreate(&entry->handle), "create execution handle");
	context.execution_handles.emplace(key, entry); ++g_execution_handle_creations; return entry;
}

std::shared_ptr<MatmulPlan> plan_for(const MatmulKey& key, hipblasLtMatmulDesc_t heuristic_desc = nullptr) {
	auto& context = device_context(key.device);
	std::lock_guard<std::mutex> guard{context.mutex};
	auto found = context.plans.find(key);
	if (found != context.plans.end()) { ++g_cache_hits; return found->second; }
	++g_cache_misses;
	auto plan = std::make_shared<MatmulPlan>(key);
	hipblasLtMatmulPreference_t preference = nullptr;
	check_lt(hipblasLtMatmulPreferenceCreate(&preference), "PreferenceCreate");
	const uint64_t workspace = 0;
	check_lt(hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)), "set workspace");
	hipblasLtMatmulHeuristicResult_t candidates[32]{}; int count = 0;
	const auto status = hipblasLtMatmulAlgoGetHeuristic(context.planning_handle, heuristic_desc ? heuristic_desc : plan->desc, plan->a, plan->b, plan->c, plan->d, preference, 32, candidates, &count);
	hipblasLtMatmulPreferenceDestroy(preference);
	check_lt(status, "AlgoGetHeuristic");
	bool selected = false;
	for (int i=0; i<count; ++i) if (candidates[i].state == HIPBLAS_STATUS_SUCCESS && candidates[i].workspaceSize == 0) {
		plan->algorithm = candidates[i].algo; selected = true; break;
	}
	if (!selected) throw std::runtime_error{"HipBLASLtMLP found no zero-workspace algorithm for the exact GEMM signature."};
	context.plans.emplace(key, plan);
	return plan;
}

void gemm(hipStream_t stream, GemmRole role, const float* a, uint32_t ar, uint32_t ac, bool ta,
	const float* b, uint32_t br, uint32_t bc, bool tb, float* d, uint32_t cr, uint32_t cc, float beta,
	EpilogueKind epilogue = EpilogueKind::Default, hipblasLtMatmulDesc_t launch_desc = nullptr) {
	int device = 0; CUDA_CHECK_THROW(hipGetDevice(&device));
	MatmulKey key{device,ar,ac,br,bc,cr,cc,ta,tb,role,epilogue,epilogue==EpilogueKind::ReluAuxBias};
	auto plan = plan_for(key, launch_desc); auto execution = execution_handle_for(device, stream);
	const float alpha = 1.0f;
	std::lock_guard<std::mutex> submit_guard{execution->submit_mutex};
	check_lt(hipblasLtMatmul(execution->handle, launch_desc ? launch_desc : plan->desc, &alpha, a, plan->a, b, plan->b,
		&beta, d, plan->c, d, plan->d, &plan->algorithm, nullptr, 0, stream), "Matmul");
}

TCNN_DEVICE float activate(float x, Activation a) {
	if (a == Activation::ReLU) return x > 0.f ? x : 0.f;
	if (a == Activation::Sigmoid) return 1.f / (1.f + expf(-x));
	return x;
}
__global__ void bias_activation(uint32_t n, float* values, const float* bias, float* pre, uint32_t width, uint32_t batch, MatrixLayout layout, Activation activation) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; if (i >= n) return;
	uint32_t row = layout == MatrixLayout::ColumnMajor ? i % width : i / batch;
	float v = values[i] + bias[row]; if (pre) pre[i] = v; values[i] = activate(v, activation);
}
__global__ void activation_only(uint32_t n, float* values, Activation activation) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; if (i >= n) return;
	values[i] = activate(values[i], activation);
}
__global__ void activation_gradient(uint32_t n, const float* pre, const float* output, const float* upstream, float* delta, Activation activation) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; if (i >= n) return;
	float derivative = 1.f;
	// ReLU's post-activation value carries the same derivative predicate as
	// its preactivation, allowing immutable RELU_BIAS descriptors in training.
	if (activation == Activation::ReLU) derivative = output[i] > 0.f ? 1.f : 0.f;
	else if (activation == Activation::Sigmoid) derivative = output[i] * (1.f-output[i]);
	delta[i] = upstream[i] * derivative;
}
__global__ void bias_gradient(uint32_t width, const float* delta, float* gradient, uint32_t batch, MatrixLayout layout, GradientMode mode) {
	uint32_t row = threadIdx.x + blockIdx.x * blockDim.x; if (row >= width) return;
	float sum = 0.f; for (uint32_t sample=0; sample<batch; ++sample) sum += delta[layout == MatrixLayout::ColumnMajor ? row + sample*width : sample + row*batch];
	if (mode == GradientMode::Overwrite) gradient[row] = sum; else gradient[row] += sum;
}

void update_partial_peak(uint64_t live) {
	auto peak = g_fused_partial_bytes_peak.load();
	while (peak < live && !g_fused_partial_bytes_peak.compare_exchange_weak(peak, live)) {}
}

void require_contiguous(const GPUMatrixDynamic<float>& matrix) {
	const uint32_t expected = matrix.layout() == MatrixLayout::ColumnMajor ? matrix.m() : matrix.n();
	if (matrix.stride() != expected) throw std::runtime_error{"HipBLASLtMLP requires contiguous tiny-cuda-nn matrix views."};
}
void linear_unfused(hipStream_t stream, const GPUMatrixDynamic<float>& input, const float* weights, const float* bias,
	GPUMatrixDynamic<float>& output, GPUMatrixDynamic<float>* pre, uint32_t in, uint32_t out, Activation activation) {
	require_contiguous(input); require_contiguous(output);
	if (input.layout() == MatrixLayout::ColumnMajor && output.layout() == MatrixLayout::ColumnMajor) {
		gemm(stream, GemmRole::Forward, weights, in, out, true, input.data(), in, input.n(), false, output.data(), out, input.n(), 0.f);
	} else if (input.layout() == MatrixLayout::RowMajor && output.layout() == MatrixLayout::RowMajor) {
		gemm(stream, GemmRole::Forward, input.data(), input.n(), in, false, weights, in, out, false, output.data(), input.n(), out, 0.f);
	} else if (input.layout() == MatrixLayout::RowMajor) {
		gemm(stream, GemmRole::Forward, weights, in, out, true, input.data(), input.n(), in, true, output.data(), out, input.n(), 0.f);
	} else {
		gemm(stream, GemmRole::Forward, input.data(), in, input.n(), true, weights, in, out, false, output.data(), input.n(), out, 0.f);
	}
	linear_kernel(bias_activation, 0, stream, input.n()*out, output.data(), bias, pre ? pre->data() : nullptr, out, input.n(), output.layout(), activation);
	++g_post_kernel_launches;
	CUDA_CHECK_THROW(hipGetLastError());
}

} // namespace

template <typename T>
struct HipBLASLtMLP<T>::EpilogueState {
	struct Entry {
		MatmulKey key;
		const float* bias;
		hipblasLtMatmulDesc_t desc = nullptr;
		~Entry() { if (desc) hipblasLtMatmulDescDestroy(desc); --g_epilogue_descriptor_count; }
	};
	std::mutex mutex;
	std::vector<std::shared_ptr<Entry>> entries;
	hipblasLtMatmulDesc_t descriptor(const MatmulKey& key, const float* bias) {
		std::lock_guard<std::mutex> guard{mutex};
		for (const auto& entry : entries) if (entry->key.ta == key.ta && entry->key.tb == key.tb && entry->key.epilogue == key.epilogue && entry->key.aux == key.aux && entry->bias == bias) return entry->desc;
		auto entry = std::make_shared<Entry>(); entry->key = key; entry->bias = bias;
		check_lt(hipblasLtMatmulDescCreate(&entry->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), "launch MatmulDescCreate");
		configure_desc(entry->desc, key);
		const void* pointer = bias;
		check_lt(hipblasLtMatmulDescSetAttribute(entry->desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &pointer, sizeof(pointer)), "set BIAS_POINTER");
		const hipDataType bias_type = HIP_R_32F;
		check_lt(hipblasLtMatmulDescSetAttribute(entry->desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)), "set BIAS_DATA_TYPE");
		entries.emplace_back(entry); ++g_epilogue_descriptor_count;
		return entry->desc;
	}
};

uint64_t hipblaslt_mlp_cache_hits() { return g_cache_hits.load(); }
uint64_t hipblaslt_mlp_cache_misses() { return g_cache_misses.load(); }
uint64_t hipblaslt_mlp_cache_size() {
	int device=0; CUDA_CHECK_THROW(hipGetDevice(&device)); auto& context=device_context(device);
	std::lock_guard<std::mutex> guard{context.mutex}; return context.plans.size();
}
uint64_t hipblaslt_epilogue_bias_launches() { return g_bias_launches.load(); }
uint64_t hipblaslt_epilogue_relu_bias_launches() { return g_relu_bias_launches.load(); }
uint64_t hipblaslt_epilogue_relu_aux_bias_launches() { return g_relu_aux_bias_launches.load(); }
uint64_t hipblaslt_epilogue_fallbacks() { return g_epilogue_fallbacks.load(); }
uint64_t hipblaslt_post_kernel_launches() { return g_post_kernel_launches.load(); }
uint64_t hipblaslt_planning_handle_count() { int device=0; CUDA_CHECK_THROW(hipGetDevice(&device)); return device_context(device).planning_handle ? 1 : 0; }
uint64_t hipblaslt_execution_handle_count() { int device=0; CUDA_CHECK_THROW(hipGetDevice(&device)); auto& context=device_context(device); std::lock_guard<std::mutex> guard{context.mutex}; return context.execution_handles.size(); }
uint64_t hipblaslt_execution_handle_capacity() { return DeviceContext::kExecutionHandleCapacity; }
uint64_t hipblaslt_execution_handle_creations() { return g_execution_handle_creations.load(); }
uint64_t hipblaslt_execution_handle_reuses() { return g_execution_handle_reuses.load(); }
uint64_t hipblaslt_execution_handle_overflows() { return g_execution_handle_overflows.load(); }
uint64_t hipblaslt_epilogue_descriptor_count() { return g_epilogue_descriptor_count.load(); }
uint64_t hipblaslt_fused_relu_biasgrad_stage1_launches() { return g_fused_stage1_launches.load(); }
uint64_t hipblaslt_fused_relu_only_launches() { return g_fused_relu_only_launches.load(); }
uint64_t hipblaslt_biasgrad_finalize_launches() { return g_biasgrad_finalize_launches.load(); }
uint64_t hipblaslt_fused_relu_biasgrad_fallbacks() { return g_fused_backward_fallbacks.load(); }
uint64_t hipblaslt_legacy_activation_grad_launches() { return g_legacy_activation_grad_launches.load(); }
uint64_t hipblaslt_legacy_bias_grad_launches() { return g_legacy_bias_grad_launches.load(); }
uint64_t hipblaslt_fused_partial_bytes_live() { return g_fused_partial_bytes_live.load(); }
uint64_t hipblaslt_fused_partial_bytes_peak() { return g_fused_partial_bytes_peak.load(); }

template <typename T> HipBLASLtMLP<T>::HipBLASLtMLP(uint32_t input_width, uint32_t hidden_width, uint32_t output_width,
	uint32_t n_hidden_layers, Activation activation, Activation output_activation)
: m_input_width{input_width},m_hidden_width{hidden_width},m_output_width{output_width},m_n_hidden_layers{n_hidden_layers},m_activation{activation},m_output_activation{output_activation} {
	static_assert(std::is_same<T,float>::value, "HipBLASLtMLP is FP32-only.");
	int device=0; CUDA_CHECK_THROW(hipGetDevice(&device)); hipDeviceProp_t props{}; CUDA_CHECK_THROW(hipGetDeviceProperties(&props,device));
	if (std::strcmp(props.gcnArchName,"gfx1201") != 0) throw std::runtime_error{"HipBLASLtMLP supports gfx1201 only."};
	if (!input_width || !output_width) throw std::runtime_error{"HipBLASLtMLP dimensions must be positive."};
	if (hidden_width!=16 && hidden_width!=32 && hidden_width!=64 && hidden_width!=128) throw std::runtime_error{"HipBLASLtMLP supports n_neurons 16, 32, 64, or 128."};
	if (n_hidden_layers!=1 && n_hidden_layers!=2 && n_hidden_layers!=4) throw std::runtime_error{"HipBLASLtMLP supports 1, 2, or 4 hidden layers."};
	if (activation!=Activation::None && activation!=Activation::ReLU) throw std::runtime_error{"HipBLASLtMLP hidden activation must be None or ReLU."};
	if (output_activation!=Activation::None && output_activation!=Activation::Sigmoid) throw std::runtime_error{"HipBLASLtMLP output activation must be None or Sigmoid."};
	for (uint32_t i=0;i<=n_hidden_layers;++i) { uint32_t in=i?hidden_width:input_width, out=i==n_hidden_layers?output_width:hidden_width;
		size_t wo=m_total_n_params, bo=wo+(size_t)in*out; m_layers.push_back({in,out,wo,bo}); m_total_n_params=bo+out; }
	m_epilogue_state = std::make_unique<EpilogueState>();
}

template <typename T> HipBLASLtMLP<T>::~HipBLASLtMLP() = default;

template <typename T> void HipBLASLtMLP<T>::linear(hipStream_t stream, const GPUMatrixDynamic<T>& input, const T* weights, const T* bias,
	GPUMatrixDynamic<T>& output, GPUMatrixDynamic<T>* pre, uint32_t in, uint32_t out, Activation activation, bool inference) {
	// hipBLASLt's column-major bias axis is D's row dimension. This matches
	// tiny-cuda-nn ColumnMajor outputs, but not the transposed RowMajor mapping.
	if ((!inference && activation == Activation::Sigmoid) || output.layout() != MatrixLayout::ColumnMajor) {
		++g_epilogue_fallbacks;
		linear_unfused(stream,input,weights,bias,output,pre,in,out,activation); return;
	}
	require_contiguous(input); require_contiguous(output);
	EpilogueKind epilogue = activation == Activation::ReLU ? EpilogueKind::ReluBias : EpilogueKind::Bias;
	int device=0; CUDA_CHECK_THROW(hipGetDevice(&device));
	MatmulKey key{}; key.device=device; key.role=GemmRole::Forward; key.epilogue=epilogue; key.aux=false;
	if (input.layout() == MatrixLayout::ColumnMajor) {
		key.ar=in; key.ac=out; key.br=in; key.bc=input.n(); key.cr=out; key.cc=input.n(); key.ta=true; key.tb=false;
	} else {
		key.ar=in; key.ac=out; key.br=input.n(); key.bc=in; key.cr=out; key.cc=input.n(); key.ta=true; key.tb=true;
	}
	auto desc=m_epilogue_state->descriptor(key,bias);
	gemm(stream,GemmRole::Forward,weights,key.ar,key.ac,key.ta,input.data(),key.br,key.bc,key.tb,output.data(),key.cr,key.cc,0.f,epilogue,desc);
	if (epilogue == EpilogueKind::ReluBias) ++g_relu_bias_launches; else ++g_bias_launches;
	if (activation == Activation::Sigmoid) { linear_kernel(activation_only,0,stream,input.n()*out,output.data(),activation); ++g_post_kernel_launches; CUDA_CHECK_THROW(hipGetLastError()); }
}

template <typename T> void HipBLASLtMLP<T>::inference_mixed_precision_impl(hipStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool inference) {
	std::vector<GPUMatrixDynamic<T>> buffers; buffers.reserve(m_n_hidden_layers); const GPUMatrixDynamic<T>* current=&input; const T* p=selected_params(inference);
	for (uint32_t i=0;i<m_n_hidden_layers;++i) { buffers.emplace_back(m_hidden_width,input.n(),stream,input.layout()); auto& l=m_layers[i]; linear(stream,*current,p+l.weight_offset,p+l.bias_offset,buffers.back(),nullptr,l.input_width,l.output_width,m_activation,true); current=&buffers.back(); }
	auto& l=m_layers.back(); linear(stream,*current,p+l.weight_offset,p+l.bias_offset,output,nullptr,l.input_width,l.output_width,m_output_activation,true);
}

template <typename T> std::unique_ptr<Context> HipBLASLtMLP<T>::forward_impl(hipStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output, bool inference, bool) {
	auto f=std::make_unique<ForwardContext>(); f->preactivations.reserve(m_n_hidden_layers); f->activations.reserve(m_n_hidden_layers); const GPUMatrixDynamic<T>* current=&input; const T* p=selected_params(inference);
	for (uint32_t i=0;i<m_n_hidden_layers;++i) { f->preactivations.emplace_back(m_hidden_width,input.n(),stream,input.layout()); f->activations.emplace_back(m_hidden_width,input.n(),stream,input.layout()); auto& l=m_layers[i]; linear(stream,*current,p+l.weight_offset,p+l.bias_offset,f->activations.back(),&f->preactivations.back(),l.input_width,l.output_width,m_activation,false); current=&f->activations.back(); }
	GPUMatrixDynamic<T>* actual=output; if (!actual) { f->owned_output={m_output_width,input.n(),stream,input.layout()}; actual=&f->owned_output; }
	f->output_preactivation={m_output_width,input.n(),stream,input.layout()}; auto& l=m_layers.back(); linear(stream,*current,p+l.weight_offset,p+l.bias_offset,*actual,&f->output_preactivation,l.input_width,l.output_width,m_output_activation,false); return f;
}

template <typename T> void HipBLASLtMLP<T>::backward_impl(hipStream_t stream, const Context& context, const GPUMatrixDynamic<T>& input, const GPUMatrixDynamic<T>& output, const GPUMatrixDynamic<T>& doutput, GPUMatrixDynamic<T>* dinput, bool inference, GradientMode mode) {
	auto& f=dynamic_cast<const ForwardContext&>(context); uint32_t batch=input.n(); const T* p=selected_params(inference);
	// Account the fully predictable Activation=None legacy path after all of
	// its launches, keeping diagnostics out of the kernel-submission sequence.
	const bool preaccount_legacy = m_activation == Activation::None;
	std::vector<GPUMatrixDynamic<T>> delta; delta.reserve(m_layers.size());
	for (uint32_t layer=0; layer<m_layers.size(); ++layer) delta.emplace_back(m_layers[layer].output_width,batch,stream,layer+1==m_layers.size()?output.layout():f.activations[layer].layout());
	linear_kernel(activation_gradient,0,stream,batch*m_output_width,f.output_preactivation.data(),output.data(),doutput.data(),delta.back().data(),m_output_activation); CUDA_CHECK_THROW(hipGetLastError());
	if (!preaccount_legacy) ++g_legacy_activation_grad_launches;
	float* gradients=nullptr; if (mode!=GradientMode::Ignore) { gradients=this->gradients(); if(!gradients) throw std::runtime_error{"HipBLASLtMLP gradient memory was not provided."}; }
	for (int32_t i=(int32_t)m_layers.size()-1;i>=0;--i) { auto& l=m_layers[i]; const auto& layer_input=i?f.activations[i-1]:input;
		if (gradients) { float beta=mode==GradientMode::Accumulate?1.f:0.f;
			gemm(stream,GemmRole::WeightGradient,layer_input.data(),layer_input.layout()==MatrixLayout::ColumnMajor?l.input_width:batch,layer_input.layout()==MatrixLayout::ColumnMajor?batch:l.input_width,layer_input.layout()==MatrixLayout::RowMajor,
				delta[i].data(),delta[i].layout()==MatrixLayout::ColumnMajor?l.output_width:batch,delta[i].layout()==MatrixLayout::ColumnMajor?batch:l.output_width,delta[i].layout()==MatrixLayout::ColumnMajor,
				gradients+l.weight_offset,l.input_width,l.output_width,beta);
			const bool hidden_fused_bias = i < (int32_t)m_n_hidden_layers && m_activation == Activation::ReLU && delta[i].layout() == MatrixLayout::ColumnMajor;
			if (!hidden_fused_bias) { linear_kernel(bias_gradient,0,stream,l.output_width,delta[i].data(),gradients+l.bias_offset,batch,delta[i].layout(),mode); if (!preaccount_legacy) ++g_legacy_bias_grad_launches; CUDA_CHECK_THROW(hipGetLastError()); } }
		auto input_gemm = [&](float* destination, MatrixLayout destination_layout) {
			if (destination_layout==MatrixLayout::ColumnMajor) gemm(stream,GemmRole::InputGradient,p+l.weight_offset,l.input_width,l.output_width,false,delta[i].data(),delta[i].layout()==MatrixLayout::ColumnMajor?l.output_width:batch,delta[i].layout()==MatrixLayout::ColumnMajor?batch:l.output_width,delta[i].layout()==MatrixLayout::RowMajor,destination,l.input_width,batch,0.f);
			else gemm(stream,GemmRole::InputGradient,delta[i].data(),delta[i].layout()==MatrixLayout::ColumnMajor?l.output_width:batch,delta[i].layout()==MatrixLayout::ColumnMajor?batch:l.output_width,delta[i].layout()==MatrixLayout::ColumnMajor,p+l.weight_offset,l.input_width,l.output_width,true,destination,batch,l.input_width,0.f);
		};
		if (i==0) { if(dinput) input_gemm(dinput->data(),dinput->layout()); }
		else { GPUMatrixDynamic<T> upstream{l.input_width,batch,stream,f.activations[i-1].layout()}; input_gemm(upstream.data(),upstream.layout());
			const bool fused = m_activation == Activation::ReLU && upstream.layout() == MatrixLayout::ColumnMajor && delta[i-1].layout() == MatrixLayout::ColumnMajor;
			if (fused) {
				const uint32_t rows_per_tile = 256 / l.input_width; const uint32_t n_partials = div_round_up(batch, rows_per_tile);
				const uint64_t partial_bytes = mode == GradientMode::Ignore ? 0 : (uint64_t)n_partials * l.input_width * sizeof(float);
				GPUMatrixDynamic<T> partials; float* partial_pointer = nullptr;
				if (partial_bytes) { partials = GPUMatrixDynamic<T>{l.input_width,n_partials,stream,MatrixLayout::ColumnMajor}; partial_pointer=partials.data(); const auto live=g_fused_partial_bytes_live.fetch_add(partial_bytes)+partial_bytes; update_partial_peak(live); }
				launch_fused_relu_backward(l.input_width,stream,upstream.data(),f.activations[i-1].data(),delta[i-1].data(),partial_pointer,gradients?gradients+m_layers[i-1].bias_offset:nullptr,batch,mode);
				if (mode == GradientMode::Ignore) ++g_fused_relu_only_launches;
				else { ++g_fused_stage1_launches; ++g_biasgrad_finalize_launches; }
				CUDA_CHECK_THROW(hipGetLastError());
				if (partial_bytes) g_fused_partial_bytes_live.fetch_sub(partial_bytes);
			} else { if (!preaccount_legacy) ++g_fused_backward_fallbacks; linear_kernel(activation_gradient,0,stream,batch*l.input_width,f.preactivations[i-1].data(),f.activations[i-1].data(),upstream.data(),delta[i-1].data(),m_activation); if (!preaccount_legacy) ++g_legacy_activation_grad_launches; CUDA_CHECK_THROW(hipGetLastError()); } }
	}
	if (preaccount_legacy) {
		g_legacy_activation_grad_launches.fetch_add(m_n_hidden_layers + 1);
		g_fused_backward_fallbacks.fetch_add(m_n_hidden_layers);
		if (mode != GradientMode::Ignore) g_legacy_bias_grad_launches.fetch_add(m_layers.size());
	}
}

template <typename T> void HipBLASLtMLP<T>::initialize_params(pcg32& rnd,float* params,float scale) { for(auto& l:m_layers){ float bound=scale*std::sqrt(6.f/(l.input_width+l.output_width)); generate_random_uniform<float>(rnd,(size_t)l.input_width*l.output_width,params+l.weight_offset,-bound,bound); CUDA_CHECK_THROW(hipMemset(params+l.bias_offset,0,l.output_width*sizeof(float))); } }
template <typename T> std::vector<std::pair<uint32_t,uint32_t>> HipBLASLtMLP<T>::layer_sizes() const { std::vector<std::pair<uint32_t,uint32_t>> r; for(auto& l:m_layers) r.emplace_back(l.input_width,l.output_width); return r; }
template <typename T> uint32_t HipBLASLtMLP<T>::width(uint32_t layer) const { if(layer>=m_n_hidden_layers) throw std::runtime_error{"HipBLASLtMLP layer out of range."}; return m_hidden_width; }
template <typename T> std::pair<const T*,MatrixLayout> HipBLASLtMLP<T>::forward_activations(const Context& c,uint32_t layer) const { if(layer>=m_n_hidden_layers) throw std::runtime_error{"HipBLASLtMLP activation out of range."}; auto& f=dynamic_cast<const ForwardContext&>(c); return {f.activations[layer].data(),f.activations[layer].layout()}; }
template <typename T> json HipBLASLtMLP<T>::hyperparams() const { return {{"otype","HipBLASLtMLP"},{"activation",to_string(m_activation)},{"output_activation",to_string(m_output_activation)},{"n_neurons",m_hidden_width},{"n_hidden_layers",m_n_hidden_layers},{"bias",true},{"precision","Fp32"}}; }
template class HipBLASLtMLP<float>;
} // namespace tcnn
