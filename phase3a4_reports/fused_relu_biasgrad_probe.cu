#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

enum class Mode : uint32_t { Overwrite, Accumulate, Ignore };

template <uint32_t WIDTH, bool COMPUTE_BIAS>
__global__ void fused_stage1(const float* upstream, const float* mask, float* dz,
	float* partial, uint32_t batch) {
	constexpr uint32_t ROWS = 256 / WIDTH;
	__shared__ float tile[ROWS][WIDTH];
	const uint32_t feature = threadIdx.x;
	const uint32_t sample = blockIdx.x * ROWS + threadIdx.y;
	float value = 0.0f;
	if (sample < batch) {
		const uint32_t index = sample * WIDTH + feature;
		value = mask[index] > 0.0f ? upstream[index] : 0.0f;
		dz[index] = value;
	}
	if constexpr (COMPUTE_BIAS) {
		tile[threadIdx.y][feature] = value;
		__syncthreads();
		if (threadIdx.y == 0) {
			float sum = 0.0f;
#pragma unroll
			for (uint32_t row = 0; row < ROWS; ++row) sum += tile[row][feature];
			partial[blockIdx.x * WIDTH + feature] = sum;
		}
	}
}

template <uint32_t WIDTH, bool ACCUMULATE>
__global__ void finalize_biasgrad(const float* partial, float* db, uint32_t count) {
	const uint32_t feature = threadIdx.x;
	if (feature >= WIDTH) return;
	float sum = 0.0f;
	for (uint32_t tile = 0; tile < count; ++tile) sum += partial[tile * WIDTH + feature];
	if constexpr (ACCUMULATE) db[feature] += sum; else db[feature] = sum;
}

static void check(hipError_t status, const char* operation) {
	if (status != hipSuccess) { std::cerr << operation << ": " << hipGetErrorString(status) << "\n"; std::exit(2); }
}

template <uint32_t WIDTH>
bool run_case(std::ofstream& out, uint32_t batch, bool explicit_stream, Mode mode, bool& first) {
	constexpr uint32_t ROWS = 256 / WIDTH;
	const uint32_t partial_count = (batch + ROWS - 1) / ROWS;
	const size_t elements = static_cast<size_t>(batch) * WIDTH;
	const float guard = 1234567.0f;
	std::vector<float> upstream(elements), mask(elements), expected_dz(elements), expected_db(WIDTH, 0.0f);
	for (size_t i=0;i<elements;++i) {
		upstream[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.25f;
		switch (i % 8) { case 0: mask[i]=-3; break; case 1: mask[i]=-0.0f; break; case 2: mask[i]=+0.0f; break; case 3: mask[i]=2; break; case 4: mask[i]=1; break; default: mask[i]=-1; }
		expected_dz[i] = mask[i] > 0.0f ? upstream[i] : 0.0f;
		expected_db[i % WIDTH] += expected_dz[i];
	}
	hipStream_t stream=nullptr; if(explicit_stream) check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"stream create");
	float *du,*dm,*dd,*dp,*db; const size_t guarded=(elements+2)*sizeof(float), db_guarded=(WIDTH+2)*sizeof(float);
	check(hipMalloc(&du,elements*4),"malloc upstream"); check(hipMalloc(&dm,elements*4),"malloc mask"); check(hipMalloc(&dd,guarded),"malloc dz");
	check(hipMalloc(&db,db_guarded),"malloc db"); dp=nullptr; if(mode!=Mode::Ignore)check(hipMalloc(&dp,static_cast<size_t>(partial_count)*WIDTH*4),"malloc partial");
	check(hipMemcpyAsync(du,upstream.data(),elements*4,hipMemcpyHostToDevice,stream),"copy upstream");check(hipMemcpyAsync(dm,mask.data(),elements*4,hipMemcpyHostToDevice,stream),"copy mask");
	std::vector<float> dz_init(elements+2,guard),db_init(WIDTH+2,guard);for(uint32_t i=0;i<WIDTH;i++)db_init[i+1]=mode==Mode::Accumulate?expected_db[i]:17.0f;
	check(hipMemcpyAsync(dd,dz_init.data(),guarded,hipMemcpyHostToDevice,stream),"init dz");check(hipMemcpyAsync(db,db_init.data(),db_guarded,hipMemcpyHostToDevice,stream),"init db");
	dim3 block(WIDTH,ROWS); float* dz=dd+1; float* db_payload=db+1;
	if(mode==Mode::Ignore) hipLaunchKernelGGL((fused_stage1<WIDTH,false>),dim3(partial_count),block,0,stream,du,dm,dz,nullptr,batch);
	else { hipLaunchKernelGGL((fused_stage1<WIDTH,true>),dim3(partial_count),block,0,stream,du,dm,dz,dp,batch); if(mode==Mode::Accumulate)hipLaunchKernelGGL((finalize_biasgrad<WIDTH,true>),dim3(1),dim3(WIDTH),0,stream,dp,db_payload,partial_count);else hipLaunchKernelGGL((finalize_biasgrad<WIDTH,false>),dim3(1),dim3(WIDTH),0,stream,dp,db_payload,partial_count); }
	check(hipGetLastError(),"kernel launch");check(hipStreamSynchronize(stream),"probe synchronize");
	std::vector<float> actual_dz(elements+2),actual_db(WIDTH+2);check(hipMemcpy(actual_dz.data(),dd,guarded,hipMemcpyDeviceToHost),"copy dz back");check(hipMemcpy(actual_db.data(),db,db_guarded,hipMemcpyDeviceToHost),"copy db back");
	double dz_error=0,db_error=0;for(size_t i=0;i<elements;i++)dz_error=std::max(dz_error,std::abs((double)actual_dz[i+1]-expected_dz[i]));
	if(mode!=Mode::Ignore)for(uint32_t i=0;i<WIDTH;i++){const float expected=mode==Mode::Accumulate?2*expected_db[i]:expected_db[i];db_error=std::max(db_error,std::abs((double)actual_db[i+1]-expected));}
	bool guards=actual_dz.front()==guard&&actual_dz.back()==guard&&actual_db.front()==guard&&actual_db.back()==guard;
	bool db_unchanged=true;if(mode==Mode::Ignore)for(uint32_t i=0;i<WIDTH;i++)db_unchanged&=actual_db[i+1]==17.0f;
	bool pass=dz_error==0.0&&db_error<2e-5&&guards&&(mode!=Mode::Ignore||db_unchanged);
	if(!first)out<<",\n";first=false;out<<"    {\"width\":"<<WIDTH<<",\"batch\":"<<batch<<",\"stream\":\""<<(explicit_stream?"explicit":"default")<<"\",\"mode\":\""<<(mode==Mode::Overwrite?"Overwrite":mode==Mode::Accumulate?"Accumulate":"Ignore")<<"\",\"rows_per_tile\":"<<ROWS<<",\"partials\":"<<partial_count<<",\"partial_bytes\":"<<(mode==Mode::Ignore?0:static_cast<size_t>(partial_count)*WIDTH*4)<<",\"dz_max_abs\":"<<dz_error<<",\"db_max_abs\":"<<db_error<<",\"guards_pass\":"<<(guards?"true":"false")<<",\"db_unchanged_for_ignore\":"<<(db_unchanged?"true":"false")<<",\"signed_zero_derivative_zero\":true,\"pass\":"<<(pass?"true":"false")<<"}";
	if(dp)hipFree(dp);hipFree(db);hipFree(dd);hipFree(dm);hipFree(du);if(stream)hipStreamDestroy(stream);return pass;
}

template <uint32_t WIDTH> bool run_width(std::ofstream& out,bool&first){bool ok=true;for(uint32_t b:{1u,7u,31u,64u,257u,1024u,4096u})for(bool s:{false,true})for(Mode m:{Mode::Overwrite,Mode::Accumulate,Mode::Ignore})ok&=run_case<WIDTH>(out,b,s,m,first);return ok;}

int main(int argc,char**argv){const char*path=argc>1?argv[1]:"fused_relu_biasgrad_probe.json";hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0),"device properties");std::ofstream out(path);out<<"{\n  \"device\":\""<<p.name<<"\",\n  \"arch\":\""<<p.gcnArchName<<"\",\n  \"results\":[\n";bool first=true,ok=true;ok&=run_width<16>(out,first);ok&=run_width<32>(out,first);ok&=run_width<64>(out,first);ok&=run_width<128>(out,first);out<<"\n  ],\n  \"result\":\""<<(ok?"PASS":"FAIL")<<"\"\n}\n";std::cout<<"PHASE3A4_FUSED_PROBE="<<(ok?"PASS":"FAIL")<<"\n";return ok?0:1;}
