/*
 * TCNN_RDNA4_P2_FIX_003
 * Portable-only network factory for HIP/gfx1201.
 */

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/networks/hipblaslt_mlp.h>
#include <tiny-cuda-nn/networks/hipblaslt_mlp_fp16.h>
#include <tiny-cuda-nn/networks/portable_mlp.h>

#if defined(TCNN_WITH_ROCWMMA_WIDTH64_MLP)
// TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001: explicit opt-in only.
#include <tiny-cuda-nn/networks/rocwmma_width64_mlp.h>
#endif

#include <type_traits>

namespace tcnn {

std::string select_network(const json& network) {
	const std::string requested = network.value("otype", "PortableMLP");
	if (
		equals_case_insensitive(requested, "PortableMLP") ||
		equals_case_insensitive(requested, "ReferenceMLP")
	) {
		return "PortableMLP";
	}
	// TCNN_RDNA4_P3A1_HIPBLASLT_002: explicit accelerated AMD backend;
	// never aliases PortableMLP and never acts as an automatic fallback.
	if (equals_case_insensitive(requested, "HipBLASLtMLP")) return "HipBLASLtMLP";
	// TCNN_RDNA4_P3B1B_FP16_FORWARD_001: an explicit, forward-only FP16
	// backend. It never replaces or aliases either existing FP32 backend.
	if (equals_case_insensitive(requested, "HipBLASLtMLPFP16")) return "HipBLASLtMLPFP16";
	// TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001: never aliases another backend.
	if (equals_case_insensitive(requested, "RocWMMAWidth64MLP")) return "RocWMMAWidth64MLP";
	if (
		equals_case_insensitive(requested, "MLP") ||
		equals_case_insensitive(requested, "CutlassMLP") ||
		equals_case_insensitive(requested, "FullyFusedMLP") ||
		equals_case_insensitive(requested, "MegakernelMLP")
	) {
		throw std::runtime_error{fmt::format(
			"{} is deliberately excluded from the AMD portable build. Use PortableMLP.",
			requested
		)};
	}
	throw std::runtime_error{fmt::format("Invalid portable network type: {}", requested)};
}

uint32_t minimum_alignment(const json& network) {
	const std::string selected = select_network(network);
	if (equals_case_insensitive(selected, "PortableMLP")) return PortableMLP<float>::REQUIRED_ALIGNMENT();
	if (equals_case_insensitive(selected, "HipBLASLtMLP")) return HipBLASLtMLP<float>::REQUIRED_ALIGNMENT();
	if (equals_case_insensitive(selected, "HipBLASLtMLPFP16")) return HipBLASLtMLPFP16::REQUIRED_ALIGNMENT();
	if (equals_case_insensitive(selected, "RocWMMAWidth64MLP")) {
#if defined(TCNN_WITH_ROCWMMA_WIDTH64_MLP)
		return RocWMMAWidth64MLP::REQUIRED_ALIGNMENT();
#else
		throw std::runtime_error{
			"RocWMMAWidth64MLP was not compiled. Set "
			"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1 and rebuild."
		};
#endif
	}
	throw std::runtime_error{"AMD network selection failed."};
}

template <typename T>
Network<T>* create_network(const json& network) {
	// TCNN_RDNA4_P2E_FIX_003: reject an explicit non-FP32 request instead of
	// silently constructing the fixed FP32 backend.
	const std::string selected = select_network(network);
	if constexpr (std::is_same<T,__half>::value) {
		if (equals_case_insensitive(selected, "RocWMMAWidth64MLP")) {
			if (!network.contains("precision") || !equals_case_insensitive(network.at("precision").get<std::string>(), "Fp16"))
				throw std::runtime_error{"RocWMMAWidth64MLP requires precision=Fp16; no implicit precision selection is allowed."};
			if (!network.value("bias", true))
				throw std::runtime_error{"RocWMMAWidth64MLP requires bias=true."};
#if defined(TCNN_WITH_ROCWMMA_WIDTH64_MLP)
			return new RocWMMAWidth64MLP{
				network.at("n_input_dims").get<uint32_t>(),
				network.value("n_neurons", 64u),
				network.at("n_output_dims").get<uint32_t>(),
				network.value("n_hidden_layers", 2u),
				string_to_activation(network.value("activation", "ReLU")),
				string_to_activation(network.value("output_activation", "None"))
			};
#else
			throw std::runtime_error{
				"RocWMMAWidth64MLP was not compiled. Set "
				"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1 and rebuild."
			};
#endif
		}
		if (!equals_case_insensitive(selected, "HipBLASLtMLPFP16"))
			throw std::runtime_error{"Only HipBLASLtMLPFP16 and the explicitly compiled RocWMMAWidth64MLP are available through the AMD FP16 network factory."};
		if (!network.contains("precision") || !equals_case_insensitive(network.at("precision").get<std::string>(), "Fp16"))
			throw std::runtime_error{"HipBLASLtMLPFP16 requires precision=Fp16; no implicit precision selection is allowed."};
		return new HipBLASLtMLPFP16{
			network.at("n_input_dims").get<uint32_t>(), network.value("n_neurons",16u),
			network.at("n_output_dims").get<uint32_t>(), network.value("n_hidden_layers",1u),
			string_to_activation(network.value("activation","ReLU")),
			string_to_activation(network.value("output_activation","None"))};
	}
	if constexpr (std::is_same<T,float>::value) {
		if (equals_case_insensitive(selected, "HipBLASLtMLPFP16"))
			throw std::runtime_error{"HipBLASLtMLPFP16 requires the explicit FP16 API path."};
		if (equals_case_insensitive(selected, "RocWMMAWidth64MLP"))
			throw std::runtime_error{"RocWMMAWidth64MLP requires the explicit FP16 API path."};
		if (network.contains("precision") && !equals_case_insensitive(network.at("precision").get<std::string>(), "Fp32")) {
			throw std::runtime_error{"AMD PortableMLP and HipBLASLtMLP support precision Fp32 only."};
		}
		auto construct = [&](auto* identity) -> Network<T>* {
			using Backend = typename std::remove_pointer<decltype(identity)>::type;
			return new Backend{
				network.at("n_input_dims").get<uint32_t>(), network.value("n_neurons", 16u),
				network.at("n_output_dims").get<uint32_t>(), network.value("n_hidden_layers", 1u),
				string_to_activation(network.value("activation", "ReLU")),
				string_to_activation(network.value("output_activation", "None"))};
		};
		if (equals_case_insensitive(selected, "PortableMLP")) return construct((PortableMLP<T>*)nullptr);
		if (equals_case_insensitive(selected, "HipBLASLtMLP")) return construct((HipBLASLtMLP<T>*)nullptr);
	}
	throw std::runtime_error{"AMD network selection failed."};
}

template Network<float>* create_network<float>(const json& network);
template Network<__half>* create_network<__half>(const json& network);

std::vector<std::string> builtin_networks() {
	std::vector<std::string> result{
		"PortableMLP",
		"HipBLASLtMLP",
		"HipBLASLtMLPFP16",
	};
#if defined(TCNN_WITH_ROCWMMA_WIDTH64_MLP)
	result.emplace_back("RocWMMAWidth64MLP");
#endif
	return result;
}

} // namespace tcnn
