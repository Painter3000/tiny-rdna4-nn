#define main phase3a_frozen_standalone_main
#include "../src/impl/qualified/phase3a_fused_backward.hip"
#undef main
#include <sstream>

// P3DA-R1-HIPBLASLT-LAYOUT-FIX-V1
// Row-major W[K,N] bytes already represent column-major A[N,K] = W^T.
// With B[K,M] = X^T and C[N,M] = Y^T, op(A)=op(B)=N yields Y=X*W.
namespace {
constexpr size_t P3B_LT_WORKSPACE_BYTES = 32u * 1024u * 1024u;
struct LtHeader { uint32_t magic, rows; };
struct Provenance {
    uint64_t x, target, weights[3], forward, dy, dx, dw[3], temporary[4];
    uint64_t workspace, workspace_bytes;
};
template<class T> void readv(std::ifstream& f,std::vector<T>& v){
    f.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T));
    if(!f){std::cerr<<"P3B-LT truncated input\n";std::exit(8);}
}
template<class T> void writev(std::ofstream& f,const std::vector<T>& v){
    f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
}
__global__ void gate(const Half* in,const float* z,uint32_t n,Half* out){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=z[i]>0.0f?in[i]:static_cast<Half>(0.0f);
}
__global__ void gate_half(const Half* in,const Half* mask,uint32_t n,Half* out){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=static_cast<float>(mask[i])>0.0f?in[i]:static_cast<Half>(0.0f);
}
void matmul(hipblasLtHandle_t h,const Half* a,uint64_t ar,uint64_t ac,
            hipblasOperation_t ta,const Half* b,uint64_t br,uint64_t bc,
            hipblasOperation_t tb,void* d,uint64_t dr,uint64_t dc,
            hipDataType dt,void* workspace,const char* operation_id,
            std::ofstream& trace,bool skip){
    hipblasLtMatmulDesc_t desc=nullptr;hipblasLtMatrixLayout_t la=nullptr,lb=nullptr,lc=nullptr,ld=nullptr;
    hipblasLtMatmulPreference_t pref=nullptr;
    lt_check(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32F,HIP_R_32F),"desc");
    lt_check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)),"ta");
    lt_check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)),"tb");
    lt_check(hipblasLtMatrixLayoutCreate(&la,HIP_R_16F,ar,ac,ar),"la");
    lt_check(hipblasLtMatrixLayoutCreate(&lb,HIP_R_16F,br,bc,br),"lb");
    lt_check(hipblasLtMatrixLayoutCreate(&lc,dt,dr,dc,dr),"lc");
    lt_check(hipblasLtMatrixLayoutCreate(&ld,dt,dr,dc,dr),"ld");
    lt_check(hipblasLtMatmulPreferenceCreate(&pref),"pref");
    uint64_t ws=P3B_LT_WORKSPACE_BYTES;
    lt_check(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)),"pref ws");
    hipblasLtMatmulHeuristicResult_t cand[32]{};int count=0,sel=-1;
    hipblasStatus_t heuristic_rc=hipblasLtMatmulAlgoGetHeuristic(h,desc,la,lb,lc,ld,pref,32,cand,&count);
    lt_check(heuristic_rc,"heuristic");
    for(int i=0;i<count;++i)if(cand[i].state==HIPBLAS_STATUS_SUCCESS&&cand[i].workspaceSize<=ws){sel=i;break;}
    if(sel<0){std::cerr<<"P3B-LT no algorithm\n";std::exit(9);}
    float alpha=1.0f,beta=0.0f;
    hipblasStatus_t matmul_rc=HIPBLAS_STATUS_NOT_INITIALIZED;
    hipError_t sync_rc=hipErrorNotReady;int executed=0;
    if(!skip){
      matmul_rc=hipblasLtMatmul(h,desc,&alpha,a,la,b,lb,&beta,d,lc,d,ld,&cand[sel].algo,workspace,ws,nullptr);
      if(matmul_rc==HIPBLAS_STATUS_SUCCESS){executed=1;sync_rc=hipDeviceSynchronize();}
    }
    trace<<"{\"operation_id\":\""<<operation_id<<"\",\"a\":"<<(uint64_t)a
         <<",\"b\":"<<(uint64_t)b<<",\"c\":"<<(uint64_t)d<<",\"d\":"<<(uint64_t)d
         <<",\"a_rows\":"<<ar<<",\"a_cols\":"<<ac<<",\"b_rows\":"<<br
         <<",\"b_cols\":"<<bc<<",\"d_rows\":"<<dr<<",\"d_cols\":"<<dc
         <<",\"lda\":"<<ar<<",\"ldb\":"<<br<<",\"ldd\":"<<dr
         <<",\"trans_a\":"<<(int)ta<<",\"trans_b\":"<<(int)tb
         <<",\"input_dtype\":\"fp16\",\"output_dtype\":\""<<(dt==HIP_R_16F?"fp16":"fp32")
         <<"\",\"compute_type\":\"fp32\",\"scale_type\":\"fp32\",\"algorithm_index\":"<<sel
         <<",\"algorithm_workspace\":"<<cand[sel].workspaceSize
         <<",\"requested_workspace\":"<<ws<<",\"workspace\":"<<(uint64_t)workspace
         <<",\"heuristic_returncode\":"<<(int)heuristic_rc
         <<",\"matmul_returncode\":"<<(int)matmul_rc<<",\"sync_returncode\":"<<(int)sync_rc
         <<",\"execution_counter\":"<<executed<<"}\n";
    if(skip){trace.flush();std::cerr<<"FAIL: hipblaslt_not_executed\n";std::exit(26);}
    lt_check(matmul_rc,"matmul");HIP_CHECK(sync_rc);
    hipblasLtMatmulPreferenceDestroy(pref);hipblasLtMatrixLayoutDestroy(ld);hipblasLtMatrixLayoutDestroy(lc);
    hipblasLtMatrixLayoutDestroy(lb);hipblasLtMatrixLayoutDestroy(la);hipblasLtMatmulDescDestroy(desc);
}
}
int main(int argc,char**argv){
    if(argc!=4)return 2;const char* negative=std::getenv("P3B_NEGATIVE");
    if(negative&&std::string(negative)=="nonzero_returncode"){std::cerr<<"FAIL: nonzero_returncode\n";return 27;}
    std::ifstream in(argv[1],std::ios::binary);std::ofstream trace(argv[3]);
    LtHeader hd{};in.read((char*)&hd,sizeof(hd));
    if(!in||hd.magic!=0x50334231u||!hd.rows||hd.rows>64)return 3;
    uint32_t rows=hd.rows,n=rows*64;std::vector<Half>x(n),target(n);std::array<std::vector<Half>,3>w,wp;
    readv(in,x);readv(in,target);for(int l=0;l<3;++l){w[l].resize(4096);readv(in,w[l]);wp[l]=w[l];}
    if(in.peek()!=std::ifstream::traits_type::eof())return 4;
    Half *dxin=nullptr,*dt=nullptr,*dw[3]{},*h0=nullptr,*h1=nullptr,*y=nullptr,*dy=nullptr,*dh1=nullptr,*dz1=nullptr,*dh0=nullptr,*dz0=nullptr;
    float *outdx=nullptr,*outdw[3]{};void*workspace=nullptr;size_t hb=n*sizeof(Half),fb=n*sizeof(float);
    HIP_CHECK(hipMalloc(&dxin,hb));HIP_CHECK(hipMalloc(&dt,hb));HIP_CHECK(hipMemcpy(dxin,x.data(),hb,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dt,target.data(),hb,hipMemcpyHostToDevice));
    for(int l=0;l<3;++l){HIP_CHECK(hipMalloc(&dw[l],8192));HIP_CHECK(hipMemcpy(dw[l],wp[l].data(),8192,hipMemcpyHostToDevice));HIP_CHECK(hipMalloc(&outdw[l],16384));}
    HIP_CHECK(hipMalloc(&h0,hb));HIP_CHECK(hipMalloc(&h1,hb));HIP_CHECK(hipMalloc(&y,hb));HIP_CHECK(hipMalloc(&dy,hb));
    HIP_CHECK(hipMalloc(&dh1,hb));HIP_CHECK(hipMalloc(&dz1,hb));HIP_CHECK(hipMalloc(&dh0,hb));HIP_CHECK(hipMalloc(&dz0,hb));
    HIP_CHECK(hipMalloc(&outdx,fb));
    if(negative&&std::string(negative)=="missing_workspace"){std::cerr<<"FAIL: missing_workspace\n";return 21;}
    HIP_CHECK(hipMalloc(&workspace,P3B_LT_WORKSPACE_BYTES));
    hipblasLtHandle_t handle=nullptr;lt_check(hipblasLtCreate(&handle),"handle");
    auto skip=[&](const char* id){return negative&&std::string(negative)=="skip_dx_l1"&&std::string(id)=="dx_l1";};
    matmul(handle,dw[0],64,64,HIPBLAS_OP_N,dxin,64,rows,HIPBLAS_OP_N,h0,64,rows,HIP_R_16F,workspace,"forward_l0",trace,skip("forward_l0"));
    hipLaunchKernelGGL(gate_half,dim3((n+255)/256),dim3(256),0,nullptr,h0,h0,n,h0);HIP_CHECK(hipDeviceSynchronize());
    matmul(handle,dw[1],64,64,HIPBLAS_OP_N,h0,64,rows,HIPBLAS_OP_N,h1,64,rows,HIP_R_16F,workspace,"forward_l1",trace,skip("forward_l1"));
    hipLaunchKernelGGL(gate_half,dim3((n+255)/256),dim3(256),0,nullptr,h1,h1,n,h1);HIP_CHECK(hipDeviceSynchronize());
    matmul(handle,dw[2],64,64,HIPBLAS_OP_N,h1,64,rows,HIPBLAS_OP_N,y,64,rows,HIP_R_16F,workspace,"forward_l2",trace,skip("forward_l2"));
    std::vector<Half>hy(n),hdy(n);HIP_CHECK(hipMemcpy(hy.data(),y,hb,hipMemcpyDeviceToHost));
    for(uint32_t i=0;i<n;++i)hdy[i]=static_cast<Half>(0.03125f*(static_cast<float>(hy[i])-static_cast<float>(target[i])));
    HIP_CHECK(hipMemcpy(dy,hdy.data(),hb,hipMemcpyHostToDevice));
    matmul(handle,dw[2],64,64,HIPBLAS_OP_N,dy,64,rows,HIPBLAS_OP_N,dh1,64,rows,HIP_R_16F,workspace,"dx_l2",trace,skip("dx_l2"));
    hipLaunchKernelGGL(gate_half,dim3((n+255)/256),dim3(256),0,nullptr,dh1,h1,n,dz1);HIP_CHECK(hipDeviceSynchronize());
    matmul(handle,dw[1],64,64,HIPBLAS_OP_N,dz1,64,rows,HIPBLAS_OP_N,dh0,64,rows,HIP_R_16F,workspace,"dx_l1",trace,skip("dx_l1"));
    hipLaunchKernelGGL(gate_half,dim3((n+255)/256),dim3(256),0,nullptr,dh0,h0,n,dz0);HIP_CHECK(hipDeviceSynchronize());
    matmul(handle,dw[0],64,64,HIPBLAS_OP_N,dz0,64,rows,HIPBLAS_OP_N,outdx,64,rows,HIP_R_32F,workspace,"dx_l0",trace,skip("dx_l0"));
    matmul(handle,dy,64,rows,HIPBLAS_OP_N,h1,64,rows,HIPBLAS_OP_T,outdw[2],64,64,HIP_R_32F,workspace,"dw2",trace,skip("dw2"));
    matmul(handle,dz1,64,rows,HIPBLAS_OP_N,h0,64,rows,HIPBLAS_OP_T,outdw[1],64,64,HIP_R_32F,workspace,"dw1",trace,skip("dw1"));
    matmul(handle,dz0,64,rows,HIPBLAS_OP_N,dxin,64,rows,HIPBLAS_OP_T,outdw[0],64,64,HIP_R_32F,workspace,"dw0",trace,skip("dw0"));
    Provenance pv{(uint64_t)dxin,(uint64_t)dt,{(uint64_t)dw[0],(uint64_t)dw[1],(uint64_t)dw[2]},(uint64_t)y,(uint64_t)dy,(uint64_t)outdx,{(uint64_t)outdw[0],(uint64_t)outdw[1],(uint64_t)outdw[2]},{(uint64_t)h0,(uint64_t)h1,(uint64_t)dz1,(uint64_t)dz0},(uint64_t)workspace,P3B_LT_WORKSPACE_BYTES};
    std::vector<float>hdx(n),hdw(12288);HIP_CHECK(hipMemcpy(hdx.data(),outdx,fb,hipMemcpyDeviceToHost));for(int l=0;l<3;++l)HIP_CHECK(hipMemcpy(hdw.data()+l*4096,outdw[l],16384,hipMemcpyDeviceToHost));
    std::ofstream out(argv[2],std::ios::binary);out.write((char*)&hd,sizeof(hd));out.write((char*)&pv,sizeof(pv));writev(out,hy);writev(out,hdy);writev(out,hdx);writev(out,hdw);
    hipblasLtDestroy(handle);if(!out)return 5;std::cout<<"HIPBLASLT_DYNAMIC_SINGLE_STEP: PASS\n";return 0;
}
