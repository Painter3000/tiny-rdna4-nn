/* TCNN_RDNA4_P3B1C_FP16_BACKWARD_001: deterministic FP16 activation/db path. */
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/object.h>
namespace tcnn {
namespace {
constexpr uint32_t kTile=256;
__global__ void activation_biasgrad_stage1(uint32_t width,uint32_t batch,const __half* upstream,const __half* activation,__half* dz,float* partials,bool relu,bool compute_bias){uint32_t row=blockIdx.x,tile=blockIdx.y,sample=tile*kTile+threadIdx.x;float value=0;if(sample<batch){size_t index=row+(size_t)sample*width;float up=__half2float(upstream[index]);float mask=relu?(__half2float(activation[index])>0.f?1.f:0.f):1.f;value=up*mask;dz[index]=__float2half(value);}__shared__ float sums[kTile];sums[threadIdx.x]=compute_bias?value:0.f;__syncthreads();for(uint32_t stride=kTile/2;stride;stride>>=1){if(threadIdx.x<stride)sums[threadIdx.x]+=sums[threadIdx.x+stride];__syncthreads();}if(compute_bias&&threadIdx.x==0)partials[row+(size_t)tile*width]=sums[0];}
__global__ void activation_biasgrad_stage2(uint32_t width,uint32_t tiles,const float* partials,float* db){uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=width)return;float sum=0;for(uint32_t tile=0;tile<tiles;++tile)sum+=partials[row+(size_t)tile*width];db[row]=sum;}
__global__ void gradient_write(uint32_t n,const float* source,__half* destination,GradientMode mode){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float value=source[i];if(mode==GradientMode::Accumulate)value+=__half2float(destination[i]);destination[i]=__float2half(value);}
}
void launch_fp16_activation_biasgrad(uint32_t width,uint32_t batch,hipStream_t stream,const __half* upstream,const __half* activation,__half* dz,float* partials,float* db,uint32_t tiles,bool relu,bool compute_bias){dim3 grid(width,tiles);activation_biasgrad_stage1<<<grid,kTile,0,stream>>>(width,batch,upstream,activation,dz,partials,relu,compute_bias);CUDA_CHECK_THROW(hipGetLastError());if(compute_bias){linear_kernel(activation_biasgrad_stage2,0,stream,width,tiles,partials,db);CUDA_CHECK_THROW(hipGetLastError());}}
void launch_fp32_gradient_write(uint32_t n,hipStream_t stream,const float* source,__half* destination,GradientMode mode){linear_kernel(gradient_write,0,stream,n,source,destination,mode);CUDA_CHECK_THROW(hipGetLastError());}
} // namespace tcnn
