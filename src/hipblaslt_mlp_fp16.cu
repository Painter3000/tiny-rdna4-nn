/* TCNN_RDNA4_P3B1B_FP16_FORWARD_001: opt-in FP16 forward-only hipBLASLt MLP. */
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/networks/hipblaslt_mlp_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace tcnn { namespace {
std::atomic<uint64_t> f_hits{0},f_misses{0},f_queries{0},f_handle_create{0},f_handle_reuse{0},f_desc{0},f_bias{0},f_relu{0};
// TCNN_RDNA4_P3B1C_FP16_BACKWARD_001: backward diagnostics and scratch accounting.
std::atomic<uint64_t> f_dx{0},f_dw{0},f_dz{0},f_db{0},f_scratch_live{0},f_scratch_peak{0};
void check(hipblasStatus_t s,const char* w){if(s!=HIPBLAS_STATUS_SUCCESS)throw std::runtime_error{fmt::format("HipBLASLtMLPFP16 {} failed: {}",w,(int)s)};}
enum class FEpi:uint8_t{Default,Bias,ReluBias};
struct FKey {
	int device; uint32_t ar,ac,br,bc,dr,dc; bool ta,tb; FEpi epi;
	hipDataType a_type,b_type,c_type,d_type,bias_type,scale_type; hipblasComputeType_t compute_type;
	MatrixLayout a_layout,b_layout,d_layout;
	bool operator==(const FKey&o)const{return device==o.device&&ar==o.ar&&ac==o.ac&&br==o.br&&bc==o.bc&&dr==o.dr&&dc==o.dc&&ta==o.ta&&tb==o.tb&&epi==o.epi&&a_type==o.a_type&&b_type==o.b_type&&c_type==o.c_type&&d_type==o.d_type&&bias_type==o.bias_type&&scale_type==o.scale_type&&compute_type==o.compute_type&&a_layout==o.a_layout&&b_layout==o.b_layout&&d_layout==o.d_layout;}
};
struct FHash {size_t operator()(const FKey&k)const{size_t h=k.device;for(auto v:{k.ar,k.ac,k.br,k.bc,k.dr,k.dc})h=h*1099511628211ull^v;h=h*131u^((size_t)k.ta<<1)^((size_t)k.tb<<2)^((size_t)k.epi<<3);h=h*131u^(size_t)k.a_type^((size_t)k.b_type<<4)^((size_t)k.c_type<<8)^((size_t)k.d_type<<12)^((size_t)k.bias_type<<16)^((size_t)k.scale_type<<20)^((size_t)k.compute_type<<24);return h^((size_t)k.a_layout<<5)^((size_t)k.b_layout<<7)^((size_t)k.d_layout<<9);}};
void configure(hipblasLtMatmulDesc_t d,const FKey&k){hipblasOperation_t ta=k.ta?HIPBLAS_OP_T:HIPBLAS_OP_N,tb=k.tb?HIPBLAS_OP_T:HIPBLAS_OP_N;check(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)),"set TRANSA");check(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)),"set TRANSB");if(k.epi!=FEpi::Default){auto e=k.epi==FEpi::ReluBias?HIPBLASLT_EPILOGUE_RELU_BIAS:HIPBLASLT_EPILOGUE_BIAS;check(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_EPILOGUE,&e,sizeof(e)),"set EPILOGUE");}}
struct FPlan {hipblasLtMatmulDesc_t desc=nullptr;hipblasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr,d=nullptr;hipblasLtMatmulAlgo_t algo{};explicit FPlan(const FKey&k){check(hipblasLtMatmulDescCreate(&desc,k.compute_type,k.scale_type),"create plan descriptor");configure(desc,k);check(hipblasLtMatrixLayoutCreate(&a,k.a_type,k.ar,k.ac,k.ar),"create A layout");check(hipblasLtMatrixLayoutCreate(&b,k.b_type,k.br,k.bc,k.br),"create B layout");check(hipblasLtMatrixLayoutCreate(&c,k.c_type,k.dr,k.dc,k.dr),"create C layout");check(hipblasLtMatrixLayoutCreate(&d,k.d_type,k.dr,k.dc,k.dr),"create D layout");}~FPlan(){if(d)hipblasLtMatrixLayoutDestroy(d);if(c)hipblasLtMatrixLayoutDestroy(c);if(b)hipblasLtMatrixLayoutDestroy(b);if(a)hipblasLtMatrixLayoutDestroy(a);if(desc)hipblasLtMatmulDescDestroy(desc);}};
struct FHandle {hipStream_t stream=nullptr;hipblasLtHandle_t handle=nullptr;std::mutex mutex;~FHandle(){if(handle)hipblasLtDestroy(handle);}};
struct FDevice {hipblasLtHandle_t planning=nullptr;std::mutex mutex;std::unordered_map<FKey,std::shared_ptr<FPlan>,FHash> plans;std::unordered_map<uintptr_t,std::shared_ptr<FHandle>> handles;FDevice(){check(hipblasLtCreate(&planning),"create planning handle");}~FDevice(){handles.clear();plans.clear();if(planning)hipblasLtDestroy(planning);}};
FDevice& fdevice(int device){static std::mutex m;static std::unordered_map<int,std::unique_ptr<FDevice>> map;std::lock_guard<std::mutex>g{m};auto&r=map[device];if(!r)r=std::make_unique<FDevice>();return*r;}
std::shared_ptr<FHandle> fhandle(int device,hipStream_t stream){auto&d=fdevice(device);std::lock_guard<std::mutex>g{d.mutex};auto key=(uintptr_t)stream;auto i=d.handles.find(key);if(i!=d.handles.end()){++f_handle_reuse;return i->second;}if(d.handles.size()>=64)throw std::runtime_error{"HipBLASLtMLPFP16 execution-handle capacity exceeded."};auto h=std::make_shared<FHandle>();h->stream=stream;check(hipblasLtCreate(&h->handle),"create execution handle");d.handles.emplace(key,h);++f_handle_create;return h;}
std::shared_ptr<FPlan> fplan(const FKey&key){auto&d=fdevice(key.device);std::lock_guard<std::mutex>g{d.mutex};auto i=d.plans.find(key);if(i!=d.plans.end()){++f_hits;return i->second;}++f_misses;auto p=std::make_shared<FPlan>(key);hipblasLtMatmulPreference_t pref=nullptr;check(hipblasLtMatmulPreferenceCreate(&pref),"create preference");uint64_t ws=0;check(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)),"set zero workspace");hipblasLtMatmulHeuristicResult_t c[32]{};int n=0;++f_queries;auto s=hipblasLtMatmulAlgoGetHeuristic(d.planning,p->desc,p->a,p->b,p->c,p->d,pref,32,c,&n);hipblasLtMatmulPreferenceDestroy(pref);check(s,"query heuristic");bool found=false;for(int j=0;j<n;++j)if(c[j].state==HIPBLAS_STATUS_SUCCESS&&c[j].workspaceSize==0){p->algo=c[j].algo;found=true;break;}if(!found)throw std::runtime_error{"HipBLASLtMLPFP16 found no exact zero-workspace FP16 plan."};d.plans.emplace(key,p);return p;}
bool width_ok(uint32_t w){return w==16||w==32||w==64||w==128;}
void contiguous(const GPUMatrixDynamic<__half>&x){if(x.layout()!=MatrixLayout::ColumnMajor||x.stride()!=x.m())throw std::runtime_error{"HipBLASLtMLPFP16 requires contiguous ColumnMajor matrices; no fallback is permitted."};}
void plain_gemm(hipStream_t stream,const __half* a,uint32_t ar,uint32_t ac,bool ta,const __half* b,uint32_t br,uint32_t bc,bool tb,void* d,uint32_t dr,uint32_t dc,hipDataType d_type){int device=0;CUDA_CHECK_THROW(hipGetDevice(&device));FKey key{device,ar,ac,br,bc,dr,dc,ta,tb,FEpi::Default,HIP_R_16F,HIP_R_16F,d_type,d_type,HIP_R_16F,HIP_R_32F,HIPBLAS_COMPUTE_32F,MatrixLayout::ColumnMajor,MatrixLayout::ColumnMajor,MatrixLayout::ColumnMajor};auto plan=fplan(key);auto handle=fhandle(device,stream);const float alpha=1,beta=0;std::lock_guard<std::mutex>guard{handle->mutex};check(hipblasLtMatmul(handle->handle,plan->desc,&alpha,a,plan->a,b,plan->b,&beta,d,plan->c,d,plan->d,&plan->algo,nullptr,0,stream),"FP16 backward matmul");}
void update_peak(uint64_t value){auto peak=f_scratch_peak.load();while(peak<value&&!f_scratch_peak.compare_exchange_weak(peak,value)){}}
struct ScratchAccount { uint64_t bytes; explicit ScratchAccount(uint64_t n):bytes{n}{auto live=f_scratch_live.fetch_add(n)+n;update_peak(live);}~ScratchAccount(){f_scratch_live.fetch_sub(bytes);} };
} // namespace

void launch_fp16_activation_biasgrad(uint32_t width,uint32_t batch,hipStream_t stream,const __half* upstream,const __half* activation,__half* dz,float* partials,float* db,uint32_t tiles,bool relu,bool compute_bias);
void launch_fp32_gradient_write(uint32_t n,hipStream_t stream,const float* source,__half* destination,GradientMode mode);

// TCNN_RDNA4_P3B1B1_FP16_FORWARD_HARDENING_001: descriptor ownership and
// accounting are committed together. A partially initialized Entry never
// decrements the global counter.
struct HipBLASLtMLPFP16::DescriptorState {
	struct Entry {
		FKey key{};
		hipStream_t stream = nullptr;
		hipblasLtMatmulDesc_t desc = nullptr;
		bool counted = false;
		std::mutex submit_mutex;
		~Entry() {
			if (desc) hipblasLtMatmulDescDestroy(desc);
			if (counted) --f_desc;
		}
	};
	std::mutex mutex;
	std::vector<std::shared_ptr<Entry>> entries;
	std::shared_ptr<Entry> get(const FKey& key, hipStream_t stream) {
		std::lock_guard<std::mutex> guard{mutex};
		for (auto& entry : entries) if (entry->key == key && entry->stream == stream) return entry;
		auto entry = std::make_shared<Entry>();
		entry->key = key;
		entry->stream = stream;
		check(hipblasLtMatmulDescCreate(&entry->desc,key.compute_type,key.scale_type),"create launch descriptor");
		configure(entry->desc,key);
		int32_t type = (int32_t)key.bias_type;
		check(hipblasLtMatmulDescSetAttribute(entry->desc,HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,&type,sizeof(type)),"set bias type");
		entries.emplace_back(entry);
		++f_desc;
		entry->counted = true;
		return entry;
	}
};

uint64_t hipblaslt_fp16_cache_hits(){return f_hits;}uint64_t hipblaslt_fp16_cache_misses(){return f_misses;}uint64_t hipblaslt_fp16_heuristic_queries(){return f_queries;}uint64_t hipblaslt_fp16_execution_handle_creations(){return f_handle_create;}uint64_t hipblaslt_fp16_execution_handle_reuses(){return f_handle_reuse;}uint64_t hipblaslt_fp16_descriptor_count(){return f_desc;}uint64_t hipblaslt_fp16_bias_launches(){return f_bias;}uint64_t hipblaslt_fp16_relu_bias_launches(){return f_relu;}
// All selected plans are zero-workspace; these counters cover owned backward scratch.
uint64_t hipblaslt_fp16_scratch_bytes_live(){return f_scratch_live;}uint64_t hipblaslt_fp16_scratch_bytes_peak(){return f_scratch_peak;}
uint64_t hipblaslt_fp16_dx_launches(){return f_dx;}uint64_t hipblaslt_fp16_dw_launches(){return f_dw;}uint64_t hipblaslt_fp16_dz_launches(){return f_dz;}uint64_t hipblaslt_fp16_db_launches(){return f_db;}
uint64_t hipblaslt_fp16_cache_size(){int d=0;CUDA_CHECK_THROW(hipGetDevice(&d));auto&x=fdevice(d);std::lock_guard<std::mutex>g{x.mutex};return x.plans.size();}
uint64_t hipblaslt_fp16_execution_handle_count(){int d=0;CUDA_CHECK_THROW(hipGetDevice(&d));auto&x=fdevice(d);std::lock_guard<std::mutex>g{x.mutex};return x.handles.size();}

HipBLASLtMLPFP16::HipBLASLtMLPFP16(uint32_t input,uint32_t hidden,uint32_t output,uint32_t layers,Activation activation,Activation output_activation):m_input_width{input},m_hidden_width{hidden},m_output_width{output},m_n_hidden_layers{layers},m_activation{activation},m_output_activation{output_activation}{int d=0;CUDA_CHECK_THROW(hipGetDevice(&d));hipDeviceProp_t p{};CUDA_CHECK_THROW(hipGetDeviceProperties(&p,d));if(std::strcmp(p.gcnArchName,"gfx1201")!=0)throw std::runtime_error{"HipBLASLtMLPFP16 supports gfx1201 only."};if(!width_ok(input)||!width_ok(hidden)||!width_ok(output))throw std::runtime_error{"HipBLASLtMLPFP16 supports input, hidden, and output widths 16, 32, 64, or 128 only."};if(layers!=1&&layers!=2&&layers!=4)throw std::runtime_error{"HipBLASLtMLPFP16 supports 1, 2, or 4 hidden layers."};if((activation!=Activation::None&&activation!=Activation::ReLU)||(output_activation!=Activation::None&&output_activation!=Activation::ReLU))throw std::runtime_error{"HipBLASLtMLPFP16 supports None and ReLU only."};for(uint32_t i=0;i<=layers;++i){uint32_t in=i?hidden:input,out=i==layers?output:hidden;size_t w=m_total_n_params,b=w+(size_t)in*out;m_layers.push_back({in,out,w,b});m_total_n_params=b+out;}m_descriptors=std::make_unique<DescriptorState>();}
HipBLASLtMLPFP16::~HipBLASLtMLPFP16()=default;
void HipBLASLtMLPFP16::linear(hipStream_t stream,const GPUMatrixDynamic<__half>&input,const __half*weights,const __half*bias,GPUMatrixDynamic<__half>&output,uint32_t in,uint32_t out,Activation activation){contiguous(input);contiguous(output);if(input.n()==0)throw std::runtime_error{"HipBLASLtMLPFP16 rejects zero-batch input explicitly."};int device=0;CUDA_CHECK_THROW(hipGetDevice(&device));FKey key{device,in,out,in,input.n(),out,input.n(),true,false,activation==Activation::ReLU?FEpi::ReluBias:FEpi::Bias,HIP_R_16F,HIP_R_16F,HIP_R_16F,HIP_R_16F,HIP_R_16F,HIP_R_32F,HIPBLAS_COMPUTE_32F,MatrixLayout::ColumnMajor,MatrixLayout::ColumnMajor,MatrixLayout::ColumnMajor};auto plan=fplan(key);auto descriptor=m_descriptors->get(key,stream);auto handle=fhandle(device,stream);const float alpha=1,beta=0;std::lock_guard<std::mutex>descriptor_guard{descriptor->submit_mutex};const void*pointer=bias;check(hipblasLtMatmulDescSetAttribute(descriptor->desc,HIPBLASLT_MATMUL_DESC_BIAS_POINTER,&pointer,sizeof(pointer)),"update bias pointer");std::lock_guard<std::mutex>handle_guard{handle->mutex};check(hipblasLtMatmul(handle->handle,descriptor->desc,&alpha,weights,plan->a,input.data(),plan->b,&beta,output.data(),plan->c,output.data(),plan->d,&plan->algo,nullptr,0,stream),"FP16 forward matmul");if(activation==Activation::ReLU)++f_relu;else++f_bias;}
void HipBLASLtMLPFP16::inference_mixed_precision_impl(hipStream_t stream,const GPUMatrixDynamic<__half>&input,GPUMatrixDynamic<__half>&output,bool inference){std::vector<GPUMatrixDynamic<__half>> buffers;buffers.reserve(m_n_hidden_layers);auto current=&input;const auto*p=selected_params(inference);if(!p)throw std::runtime_error{"HipBLASLtMLPFP16 parameters were not provided."};for(uint32_t i=0;i<m_n_hidden_layers;++i){buffers.emplace_back(m_hidden_width,input.n(),stream,MatrixLayout::ColumnMajor);auto&l=m_layers[i];linear(stream,*current,p+l.weights,p+l.bias,buffers.back(),l.in,l.out,m_activation);current=&buffers.back();}auto&l=m_layers.back();linear(stream,*current,p+l.weights,p+l.bias,output,l.in,l.out,m_output_activation);}
std::unique_ptr<Context> HipBLASLtMLPFP16::forward_impl(hipStream_t stream,const GPUMatrixDynamic<__half>&input,GPUMatrixDynamic<__half>*output,bool inference,bool){auto f=std::make_unique<ForwardContext>();f->activations.reserve(m_n_hidden_layers);auto current=&input;const auto*p=selected_params(inference);if(!p)throw std::runtime_error{"HipBLASLtMLPFP16 parameters were not provided."};for(uint32_t i=0;i<m_n_hidden_layers;++i){f->activations.emplace_back(m_hidden_width,input.n(),stream,MatrixLayout::ColumnMajor);auto&l=m_layers[i];linear(stream,*current,p+l.weights,p+l.bias,f->activations.back(),l.in,l.out,m_activation);current=&f->activations.back();}if(!output){f->owned_output={m_output_width,input.n(),stream,MatrixLayout::ColumnMajor};output=&f->owned_output;}auto&l=m_layers.back();linear(stream,*current,p+l.weights,p+l.bias,*output,l.in,l.out,m_output_activation);return f;}

bool hipblaslt_fp16_test_null_parameter_guard() {
	HipBLASLtMLPFP16 model{16,16,16,1,Activation::None,Activation::None};
	model.set_params(nullptr,nullptr,nullptr);
	GPUMatrixDynamic<__half> input{16,16,nullptr,MatrixLayout::ColumnMajor};
	GPUMatrixDynamic<__half> output{16,16,nullptr,MatrixLayout::ColumnMajor};
	try { model.forward_impl(nullptr,input,&output,false,false); }
	catch (const std::exception& error) { return std::string{error.what()}.find("parameters were not provided") != std::string::npos; }
	return false;
}

bool hipblaslt_fp16_test_invalid_descriptor_counter() {
	const uint64_t before = f_desc.load();
	try {
		HipBLASLtMLPFP16::DescriptorState state;
		FKey invalid{0,16,16,16,16,16,16,true,false,FEpi::Bias,HIP_R_16F,HIP_R_16F,HIP_R_16F,HIP_R_16F,HIP_R_16F,HIP_R_32F,(hipblasComputeType_t)-1,MatrixLayout::ColumnMajor,MatrixLayout::ColumnMajor,MatrixLayout::ColumnMajor};
		state.get(invalid,nullptr);
	} catch (const std::runtime_error&) { return f_desc.load() == before; }
	return false;
}
// TCNN_RDNA4_P3B1C_FP16_BACKWARD_001: FP16 operands, FP32 GEMM accumulation,
// deterministic FP32 bias reduction, and a single final FP16 gradient write.
void HipBLASLtMLPFP16::backward_impl(hipStream_t stream,const Context& context,const GPUMatrixDynamic<__half>& input,const GPUMatrixDynamic<__half>& output,const GPUMatrixDynamic<__half>& doutput,GPUMatrixDynamic<__half>* dinput,bool inference,GradientMode mode){
	auto& forward=dynamic_cast<const ForwardContext&>(context);const __half* params=selected_params(inference);if(!params)throw std::runtime_error{"HipBLASLtMLPFP16 parameters were not provided."};__half* gradients=mode==GradientMode::Ignore?nullptr:this->gradients();if(mode!=GradientMode::Ignore&&!gradients)throw std::runtime_error{"HipBLASLtMLPFP16 gradient memory was not provided."};const uint32_t batch=input.n(),layer_count=m_layers.size(),tiles=div_round_up(batch,256u);uint64_t scratch_bytes=0;for(const auto&l:m_layers){scratch_bytes+=(uint64_t)l.out*batch*sizeof(__half);if(mode!=GradientMode::Ignore)scratch_bytes+=(uint64_t)l.in*l.out*sizeof(float)+(uint64_t)l.out*(tiles+1)*sizeof(float);}if(dinput)scratch_bytes+=(uint64_t)m_input_width*batch*sizeof(__half);ScratchAccount account{scratch_bytes};
	std::vector<GPUMatrixDynamic<__half>> dz;dz.reserve(layer_count);for(const auto&l:m_layers)dz.emplace_back(l.out,batch,stream,MatrixLayout::ColumnMajor);std::vector<GPUMatrixDynamic<float>> partials,dbs;if(mode!=GradientMode::Ignore){partials.reserve(layer_count);dbs.reserve(layer_count);for(const auto&l:m_layers){partials.emplace_back(l.out,tiles,stream,MatrixLayout::ColumnMajor);dbs.emplace_back(l.out,1,stream,MatrixLayout::ColumnMajor);}}
	launch_fp16_activation_biasgrad(m_output_width,batch,stream,doutput.data(),output.data(),dz.back().data(),mode==GradientMode::Ignore?nullptr:partials.back().data(),mode==GradientMode::Ignore?nullptr:dbs.back().data(),tiles,m_output_activation==Activation::ReLU,mode!=GradientMode::Ignore);++f_dz;if(mode!=GradientMode::Ignore)++f_db;
	for(int32_t i=(int32_t)layer_count-1;i>=0;--i){const auto&l=m_layers[i];const GPUMatrixDynamic<__half>& layer_input=i?forward.activations[i-1]:input;
		if(mode!=GradientMode::Ignore){GPUMatrixDynamic<float> dw{l.in,l.out,stream,MatrixLayout::ColumnMajor};plain_gemm(stream,layer_input.data(),l.in,batch,false,dz[i].data(),l.out,batch,true,dw.data(),l.in,l.out,HIP_R_32F);++f_dw;launch_fp32_gradient_write(l.in*l.out,stream,dw.data(),gradients+l.weights,mode);launch_fp32_gradient_write(l.out,stream,dbs[i].data(),gradients+l.bias,mode);}
		if(i>0||dinput){GPUMatrixDynamic<__half> owned;GPUMatrixDynamic<__half>* da=nullptr;if(i==0)da=dinput;else{owned={l.in,batch,stream,MatrixLayout::ColumnMajor};da=&owned;}plain_gemm(stream,params+l.weights,l.in,l.out,false,dz[i].data(),l.out,batch,false,da->data(),l.in,batch,HIP_R_16F);++f_dx;if(i>0){const auto&previous=m_layers[i-1];launch_fp16_activation_biasgrad(previous.out,batch,stream,da->data(),forward.activations[i-1].data(),dz[i-1].data(),mode==GradientMode::Ignore?nullptr:partials[i-1].data(),mode==GradientMode::Ignore?nullptr:dbs[i-1].data(),tiles,m_activation==Activation::ReLU,mode!=GradientMode::Ignore);++f_dz;if(mode!=GradientMode::Ignore)++f_db;}}
	}
}
void HipBLASLtMLPFP16::initialize_params(pcg32&r,float*p,float scale){for(auto&l:m_layers){float bound=scale*std::sqrt(6.f/(l.in+l.out));generate_random_uniform<float>(r,(size_t)l.in*l.out,p+l.weights,-bound,bound);CUDA_CHECK_THROW(hipMemset(p+l.bias,0,l.out*sizeof(float)));}}
std::vector<std::pair<uint32_t,uint32_t>> HipBLASLtMLPFP16::layer_sizes()const{std::vector<std::pair<uint32_t,uint32_t>>r;for(auto&l:m_layers)r.emplace_back(l.in,l.out);return r;}uint32_t HipBLASLtMLPFP16::width(uint32_t l)const{if(l>=m_n_hidden_layers)throw std::runtime_error{"HipBLASLtMLPFP16 layer out of range."};return m_hidden_width;}std::pair<const __half*,MatrixLayout> HipBLASLtMLPFP16::forward_activations(const Context&c,uint32_t l)const{if(l>=m_n_hidden_layers)throw std::runtime_error{"HipBLASLtMLPFP16 activation out of range."};auto&f=dynamic_cast<const ForwardContext&>(c);return{f.activations[l].data(),f.activations[l].layout()};}json HipBLASLtMLPFP16::hyperparams()const{return{{"otype","HipBLASLtMLPFP16"},{"activation",to_string(m_activation)},{"output_activation",to_string(m_output_activation)},{"n_neurons",m_hidden_width},{"n_hidden_layers",m_n_hidden_layers},{"bias",true},{"operand_precision","Fp16"},{"accumulation_precision","Fp32"},{"hidden_output_precision","Fp16"},{"final_output_precision","Fp16"},{"parameter_gradient_internal_precision","Fp32"},{"parameter_gradient_external_precision","Fp16"},{"input_gradient_internal_precision","Fp16"},{"backward",true}};}
} // namespace tcnn
