/*
 * TCNN_RDNA4_P2_FIX_003
 * Portable-only network factory for HIP/gfx1201.
 */

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/networks/hipblaslt_mlp.h>
#include <tiny-cuda-nn/networks/hipblaslt_mlp_fp16.h>
#include <tiny-cuda-nn/networks/portable_mlp.h>

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
	throw std::runtime_error{"AMD network selection failed."};
}

template <typename T>
Network<T>* create_network(const json& network) {
	// TCNN_RDNA4_P2E_FIX_003: reject an explicit non-FP32 request instead of
	// silently constructing the fixed FP32 backend.
	const std::string selected = select_network(network);
	if constexpr (std::is_same<T,__half>::value) {
		if (!equals_case_insensitive(selected, "HipBLASLtMLPFP16"))
			throw std::runtime_error{"Only HipBLASLtMLPFP16 is available through the AMD FP16 network factory."};
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
	return {"PortableMLP", "HipBLASLtMLP", "HipBLASLtMLPFP16"};
}

} // namespace tcnn
