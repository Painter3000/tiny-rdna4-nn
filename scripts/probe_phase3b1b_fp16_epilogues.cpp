// TCNN_RDNA4_P3B1B_FP16_FORWARD_001: empirical FP16 BIAS/RELU_BIAS checkpoint.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <json/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {
constexpr const char* kMarkerB = "TCNN_RDNA4_P3B1B_FP16_FORWARD_001";
constexpr size_t kGuard = 256;
constexpr size_t kWorkspaceLimit = 64ull << 20;

void hip_ok(hipError_t s, const char* what) {
	if (s != hipSuccess) throw std::runtime_error{std::string{what} + ": " + hipGetErrorString(s)};
}
void lt_ok(hipblasStatus_t s, const char* what) {
	if (s != HIPBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{what} + ": status=" + std::to_string((int)s)};
}
size_t cm(int64_t r, int64_t c, int64_t ld) { return (size_t)r + (size_t)c * ld; }
float hround(float x) { return __half2float(__float2half(x)); }

struct Guarded {
	void* base = nullptr;
	size_t bytes;
	uint8_t fill, guard;
	Guarded(size_t n, uint8_t f, uint8_t g) : bytes{n}, fill{f}, guard{g} {
		hip_ok(hipMalloc(&base, kGuard + bytes + kGuard), "hipMalloc guarded"); reset();
	}
	~Guarded() { if (base) hipFree(base); }
	void* data() const { return static_cast<uint8_t*>(base) + kGuard; }
	void reset() {
		std::vector<uint8_t> h(kGuard + bytes + kGuard, guard);
		std::fill(h.begin() + kGuard, h.begin() + kGuard + bytes, fill);
		hip_ok(hipMemcpy(base, h.data(), h.size(), hipMemcpyHostToDevice), "reset guarded");
	}
	json inspect() const {
		std::vector<uint8_t> h(kGuard + bytes + kGuard);
		hip_ok(hipMemcpy(h.data(), base, h.size(), hipMemcpyDeviceToHost), "inspect guarded");
		return {{"prefix_guard_intact", std::all_of(h.begin(), h.begin()+kGuard, [&](auto v){return v==guard;})},
			{"suffix_guard_intact", std::all_of(h.end()-kGuard, h.end(), [&](auto v){return v==guard;})},
			{"payload_unchanged", std::all_of(h.begin()+kGuard, h.end()-kGuard, [&](auto v){return v==fill;})}};
	}
};

struct Problem {
	std::vector<__half> a, b, bias;
	std::vector<float> mm, bias_ref, relu_ref;
};

Problem make_problem(int64_t m, int64_t n, int64_t k) {
	Problem p;
	p.a.resize(m*k); p.b.resize(k*n); p.bias.resize(m);
	for (int64_t inner=0; inner<k; ++inner) for (int64_t row=0; row<m; ++row) {
		float v = row == 0 ? 0.0f : ((row*17 + inner*13) % 29 - 14) * 0.03125f;
		if ((row+inner)%31==0) v = 4.0f;
		p.a[cm(row,inner,m)] = __float2half(v);
	}
	for (int64_t col=0; col<n; ++col) for (int64_t inner=0; inner<k; ++inner) {
		float v = ((col*11 + inner*7) % 31 - 15) * 0.0234375f;
		if ((col+inner)%37==0) v = -3.0f;
		p.b[cm(inner,col,k)] = __float2half(v);
	}
	p.mm.resize(m*n);
	for (int64_t col=0; col<n; ++col) for (int64_t row=0; row<m; ++row) {
		float acc=0.0f;
		for (int64_t inner=0; inner<k; ++inner)
			acc += __half2float(p.a[cm(row,inner,m)]) * __half2float(p.b[cm(inner,col,k)]);
		p.mm[cm(row,col,m)] = acc;
	}
	for (int64_t row=0; row<m; ++row) {
		if (row == 0) { p.bias[row] = __float2half(0.0f); continue; }
		float offset;
		switch (row % 8) {
			case 0: offset=0.0f; break; case 1: offset=0.0009765625f; break;
			case 2: offset=-0.0009765625f; break; case 3: offset=0.25f; break;
			case 4: offset=-0.25f; break; case 5: offset=32.0f; break;
			case 6: offset=-32.0f; break; default: offset=1.0f; break;
		}
		p.bias[row] = __float2half(-p.mm[cm(row,0,m)] + offset);
	}
	p.bias_ref.resize(m*n); p.relu_ref.resize(m*n);
	for (int64_t col=0; col<n; ++col) for (int64_t row=0; row<m; ++row) {
		float x=p.mm[cm(row,col,m)] + __half2float(p.bias[row]);
		p.bias_ref[cm(row,col,m)]=x; p.relu_ref[cm(row,col,m)]=std::max(x,0.0f);
	}
	return p;
}

json probe(int64_t m, int64_t n, int64_t k, const std::string& epi_name,
	const std::string& d_name, int repeat) {
	const bool relu=epi_name=="RELU_BIAS", d16=d_name=="f16";
	const auto d_type=d16 ? HIP_R_16F : HIP_R_32F;
	const auto epilogue=relu ? HIPBLASLT_EPILOGUE_RELU_BIAS : HIPBLASLT_EPILOGUE_BIAS;
	Problem hp=make_problem(m,n,k);
	void *a=nullptr,*b=nullptr,*bias=nullptr,*c=nullptr;
	hip_ok(hipMalloc(&a,hp.a.size()*sizeof(__half)),"malloc A");
	hip_ok(hipMalloc(&b,hp.b.size()*sizeof(__half)),"malloc B");
	hip_ok(hipMalloc(&bias,hp.bias.size()*sizeof(__half)),"malloc bias");
	const size_t dbytes=(size_t)m*n*(d16?sizeof(__half):sizeof(float));
	hip_ok(hipMalloc(&c,dbytes),"malloc C"); hip_ok(hipMemset(c,0,dbytes),"clear C");
	hip_ok(hipMemcpy(a,hp.a.data(),hp.a.size()*sizeof(__half),hipMemcpyHostToDevice),"copy A");
	hip_ok(hipMemcpy(b,hp.b.data(),hp.b.size()*sizeof(__half),hipMemcpyHostToDevice),"copy B");
	hip_ok(hipMemcpy(bias,hp.bias.data(),hp.bias.size()*sizeof(__half),hipMemcpyHostToDevice),"copy bias");
	Guarded out(dbytes,0xa5,0x5a);
	hipblasLtHandle_t planning=nullptr, handles[2]={}; hipStream_t streams[2]={};
	hipblasLtMatmulDesc_t op=nullptr; hipblasLtMatrixLayout_t la=nullptr,lb=nullptr,lc=nullptr,ld=nullptr;
	hipblasLtMatmulPreference_t pref=nullptr;
	lt_ok(hipblasLtCreate(&planning),"create planning handle");
	for(int i=0;i<2;++i){lt_ok(hipblasLtCreate(&handles[i]),"create execution handle");hip_ok(hipStreamCreate(&streams[i]),"create stream");}
	lt_ok(hipblasLtMatmulDescCreate(&op,HIPBLAS_COMPUTE_32F,HIP_R_32F),"create matmul desc");
	hipblasOperation_t nn=HIPBLAS_OP_N;
	lt_ok(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_TRANSA,&nn,sizeof(nn)),"set trans A");
	lt_ok(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_TRANSB,&nn,sizeof(nn)),"set trans B");
	lt_ok(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)),"set epilogue");
	lt_ok(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias)),"set bias pointer");
	int32_t bias_type=(int32_t)HIP_R_16F;
	lt_ok(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,&bias_type,sizeof(bias_type)),"set bias type");
	lt_ok(hipblasLtMatrixLayoutCreate(&la,HIP_R_16F,m,k,m),"layout A");
	lt_ok(hipblasLtMatrixLayoutCreate(&lb,HIP_R_16F,k,n,k),"layout B");
	lt_ok(hipblasLtMatrixLayoutCreate(&lc,d_type,m,n,m),"layout C");
	lt_ok(hipblasLtMatrixLayoutCreate(&ld,d_type,m,n,m),"layout D");
	lt_ok(hipblasLtMatmulPreferenceCreate(&pref),"create preference");
	uint64_t maxws=kWorkspaceLimit;
	lt_ok(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&maxws,sizeof(maxws)),"set workspace pref");
	std::array<hipblasLtMatmulHeuristicResult_t,32> candidates{}; int count=0;
	auto hs=hipblasLtMatmulAlgoGetHeuristic(planning,op,la,lb,lc,ld,pref,candidates.size(),candidates.data(),&count);
	int selected=-1; for(int i=0;i<count;++i) if(candidates[i].state==HIPBLAS_STATUS_SUCCESS && candidates[i].workspaceSize<=kWorkspaceLimit){selected=i;break;}
	json result={{"marker",kMarkerB},{"m",m},{"n",n},{"k",k},{"direction","NN"},{"epilogue",epi_name},{"d_type",d16?"HIP_R_16F":"HIP_R_32F"},{"bias_type","HIP_R_16F"},{"compute_type","HIPBLAS_COMPUTE_32F"},{"fresh_repeat",repeat},{"heuristic_status",(int)hs},{"heuristic_count",count},{"selected_candidate",selected},{"stream_count",2},{"execution_handle_count",2}};
	if(hs!=HIPBLAS_STATUS_SUCCESS || selected<0){result["probe_state"]="UNSUPPORTED";} else {
		auto &chosen=candidates[selected]; Guarded ws(std::max<size_t>(chosen.workspaceSize,1),0xc3,0x7e);
		const float alpha=1,beta=0; std::vector<float> got; json launches=json::array();
		for(int launch=0;launch<6;++launch){int si=launch%2;out.reset();auto ms=hipblasLtMatmul(handles[si],op,&alpha,a,la,b,lb,&beta,c,lc,out.data(),ld,&chosen.algo,chosen.workspaceSize?ws.data():nullptr,chosen.workspaceSize,streams[si]);auto ss=hipStreamSynchronize(streams[si]);auto mem=out.inspect();launches.push_back({{"launch",launch+1},{"stream_index",si},{"matmul_status",(int)ms},{"synchronize_status",(int)ss},{"output_memory",mem}});if(ms==HIPBLAS_STATUS_SUCCESS&&ss==hipSuccess){got.resize(m*n);if(d16){std::vector<__half> h(m*n);hip_ok(hipMemcpy(h.data(),out.data(),dbytes,hipMemcpyDeviceToHost),"copy D16");for(size_t i=0;i<h.size();++i)got[i]=__half2float(h[i]);}else hip_ok(hipMemcpy(got.data(),out.data(),dbytes,hipMemcpyDeviceToHost),"copy D32");}}
		const auto& rawref=relu?hp.relu_ref:hp.bias_ref; double max_abs=0,max_rel=0,l2n=0,l2d=0;int mask_bad=0,nan=0,inf=0,sign_flips=0,near_zero=0,exact_zero=0;
		for(size_t i=0;i<got.size();++i){float ref=d16?hround(rawref[i]):rawref[i];double e=std::abs((double)got[i]-ref);max_abs=std::max(max_abs,e);if(std::abs(ref)>1e-3)max_rel=std::max(max_rel,e/std::abs(ref));l2n+=e*e;l2d+=(double)ref*ref;if(std::isnan(got[i]))++nan;if(std::isinf(got[i]))++inf;if(relu&&((got[i]>0)!=(ref>0)))++mask_bad;if(std::abs(rawref[i])<=0.002)++near_zero;if(rawref[i]==0)++exact_zero;if((hp.mm[i]>0)!=(rawref[i]>0))++sign_flips;}
		result["launches"]=launches;result["workspace_memory"]=ws.inspect();result["steady_state"]={{"launches",4},{"new_handles",0},{"new_heuristic_queries",0},{"new_descriptors",0},{"workspace_growth_bytes",0}};result["algorithm"]={{"index",hipblaslt_ext::getIndexFromAlgo(chosen.algo)},{"solution_name",hipblaslt_ext::getSolutionNameFromAlgo(planning,chosen.algo)},{"kernel_name",hipblaslt_ext::getKernelNameFromAlgo(planning,chosen.algo)},{"workspace_bytes",chosen.workspaceSize}};result["numerics"]={{"max_abs",max_abs},{"max_rel_outside_1e-3",max_rel},{"normalized_l2",std::sqrt(l2n/std::max(l2d,1e-30))},{"nan_count",nan},{"inf_count",inf},{"relu_mask_mismatches",mask_bad}};result["target_coverage"]={{"near_zero_count",near_zero},{"exact_zero_count",exact_zero},{"bias_sign_flip_count",sign_flips},{"large_finite_bias",true},{"bias_axis_rows",m}};result["probe_state"]="EXECUTED";
	}
	if(pref)hipblasLtMatmulPreferenceDestroy(pref);if(ld)hipblasLtMatrixLayoutDestroy(ld);if(lc)hipblasLtMatrixLayoutDestroy(lc);if(lb)hipblasLtMatrixLayoutDestroy(lb);if(la)hipblasLtMatrixLayoutDestroy(la);if(op)hipblasLtMatmulDescDestroy(op);for(int i=0;i<2;++i){if(handles[i])hipblasLtDestroy(handles[i]);if(streams[i])hipStreamDestroy(streams[i]);}if(planning)hipblasLtDestroy(planning);if(c)hipFree(c);if(bias)hipFree(bias);if(b)hipFree(b);if(a)hipFree(a);return result;
}
} // namespace

int main(int argc,char** argv){json r={{"marker",kMarkerB}};try{if(argc!=7)throw std::runtime_error{"usage: probe M N K BIAS|RELU_BIAS f16|f32 FRESH_REPEAT"};std::string e=argv[4],d=argv[5];if(e!="BIAS"&&e!="RELU_BIAS")throw std::runtime_error{"invalid epilogue"};if(d!="f16"&&d!="f32")throw std::runtime_error{"invalid D type"};r=probe(std::stoll(argv[1]),std::stoll(argv[2]),std::stoll(argv[3]),e,d,std::stoi(argv[6]));r["process_status"]="PASS";}catch(const std::exception& x){r["process_status"]="ERROR";r["error"]=x.what();}std::cout<<r.dump()<<std::endl;return r.value("process_status","ERROR")=="PASS"?0:1;}
