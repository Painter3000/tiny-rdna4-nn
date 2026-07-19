/*
 * TCNN_RDNA4_P2_FIX_003
 * Portable-only network factory for HIP/gfx1201.
 */

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
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
	if (!equals_case_insensitive(selected, "PortableMLP")) {
		throw std::runtime_error{"Portable network selection failed."};
	}
	return PortableMLP<float>::REQUIRED_ALIGNMENT();
}

template <typename T>
Network<T>* create_network(const json& network) {
	static_assert(std::is_same<T, float>::value, "PortableMLP factory is FP32-only.");
	const std::string selected = select_network(network);
	if (!equals_case_insensitive(selected, "PortableMLP")) {
		throw std::runtime_error{"Portable network selection failed."};
	}
	return new PortableMLP<T>{
		network.at("n_input_dims").get<uint32_t>(),
		network.value("n_neurons", 16u),
		network.at("n_output_dims").get<uint32_t>(),
		network.value("n_hidden_layers", 1u),
		string_to_activation(network.value("activation", "ReLU")),
		string_to_activation(network.value("output_activation", "None"))
	};
}

template Network<float>* create_network<float>(const json& network);

std::vector<std::string> builtin_networks() {
	return {"PortableMLP"};
}

} // namespace tcnn
