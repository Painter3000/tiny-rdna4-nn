// P3D0B-BETA-001: measured gfx1201 FP32 beta-power recurrence.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

__global__ void probe(uint32_t* b1, uint32_t* b2, uint32_t n) {
    if (blockIdx.x || threadIdx.x) return;
    float x1=1.0f,x2=1.0f;
    for(uint32_t i=0;i<n;++i) {
        x1=x1*0.9f;x2=x2*0.999f;
        b1[i]=__float_as_uint(x1);b2[i]=__float_as_uint(x2);
    }
}
int main(int argc,char**argv) {
    if(argc!=2)return 2;constexpr uint32_t N=30000;uint32_t *a=nullptr,*b=nullptr;
    if(hipMalloc(&a,N*4)!=hipSuccess||hipMalloc(&b,N*4)!=hipSuccess)return 3;
    hipLaunchKernelGGL(probe,dim3(1),dim3(1),0,nullptr,a,b,N);
    if(hipGetLastError()!=hipSuccess||hipDeviceSynchronize()!=hipSuccess)return 4;
    std::vector<uint32_t> h(2*N);
    if(hipMemcpy(h.data(),a,N*4,hipMemcpyDeviceToHost)!=hipSuccess||
       hipMemcpy(h.data()+N,b,N*4,hipMemcpyDeviceToHost)!=hipSuccess)return 5;
    std::ofstream o(argv[1],std::ios::binary);o.write((char*)h.data(),h.size()*4);
    if(hipFree(a)!=hipSuccess||hipFree(b)!=hipSuccess||!o)return 6;
    std::cout<<"P3D0B-BETA-POWER-PROBE: PASS\n";return 0;
}
