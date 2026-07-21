#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#define HC(x) do { auto s=(x); if(s!=hipSuccess){std::cerr<<#x<<":"<<hipGetErrorString(s)<<"\n";return 2;} } while(0)
#define BC(x) do { auto s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){std::cerr<<#x<<":"<<(int)s<<"\n";return 3;} } while(0)

int main(int argc,char**argv){
	constexpr int W=16,B=7; const char* path=argc>1?argv[1]:"drelu_semantics.json";
	std::vector<float>a(W*W,0),g(W*B),aux(W*B),out(W*B),bg(W,77.0f);
	for(int i=0;i<W;i++)a[i+i*W]=1;
	const float av[6]={-3,-1,-0.0f,+0.0f,1,3}; const float gv[5]={-2,-1,0,1,2};
	for(int c=0;c<B;c++)for(int r=0;r<W;r++){g[r+c*W]=gv[(r+c)%5];aux[r+c*W]=av[(r+2*c)%6];}
	float *da,*dg,*dx,*do_,*dbg; HC(hipMalloc(&da,a.size()*4));HC(hipMalloc(&dg,g.size()*4));HC(hipMalloc(&dx,aux.size()*4));HC(hipMalloc(&do_,out.size()*4));HC(hipMalloc(&dbg,bg.size()*4));
	HC(hipMemcpy(da,a.data(),a.size()*4,hipMemcpyHostToDevice));HC(hipMemcpy(dg,g.data(),g.size()*4,hipMemcpyHostToDevice));HC(hipMemcpy(dx,aux.data(),aux.size()*4,hipMemcpyHostToDevice));
	hipblasLtHandle_t h;BC(hipblasLtCreate(&h)); hipblasLtMatrixLayout_t la,lb,ld;BC(hipblasLtMatrixLayoutCreate(&la,HIP_R_32F,W,W,W));BC(hipblasLtMatrixLayoutCreate(&lb,HIP_R_32F,W,B,W));BC(hipblasLtMatrixLayoutCreate(&ld,HIP_R_32F,W,B,W));
	std::ofstream f(path);f<<"{\n\"cases\":[\n";
	for(int which=0;which<2;which++)for(int sentinel=0;sentinel<2;sentinel++){
		std::fill(bg.begin(),bg.end(),sentinel?31.0f:77.0f);HC(hipMemcpy(dbg,bg.data(),bg.size()*4,hipMemcpyHostToDevice));HC(hipMemset(do_,0,out.size()*4));
		hipblasLtMatmulDesc_t d;BC(hipblasLtMatmulDescCreate(&d,HIPBLAS_COMPUTE_32F,HIP_R_32F));auto epi=static_cast<hipblasLtEpilogue_t>(which?152u:136u);BC(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_EPILOGUE,&epi,sizeof(epi)));void* ap=dx;int64_t ald=W;BC(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,&ap,sizeof(ap)));BC(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,&ald,sizeof(ald)));if(which){void*bp=dbg;hipDataType t=HIP_R_32F;BC(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_BIAS_POINTER,&bp,sizeof(bp)));BC(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,&t,sizeof(t)));}
		hipblasLtMatmulPreference_t p;BC(hipblasLtMatmulPreferenceCreate(&p));uint64_t z=0;BC(hipblasLtMatmulPreferenceSetAttribute(p,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&z,sizeof(z)));hipblasLtMatmulHeuristicResult_t hr[8]{};int n=0;BC(hipblasLtMatmulAlgoGetHeuristic(h,d,la,lb,ld,ld,p,8,hr,&n));if(!n)return 4;float alpha=1,beta=0;BC(hipblasLtMatmul(h,d,&alpha,da,la,dg,lb,&beta,do_,ld,do_,ld,&hr[0].algo,nullptr,0,nullptr));HC(hipDeviceSynchronize());HC(hipMemcpy(out.data(),do_,out.size()*4,hipMemcpyDeviceToHost));HC(hipMemcpy(bg.data(),dbg,bg.size()*4,hipMemcpyDeviceToHost));
		if(which||sentinel)f<<",\n";f<<"{\"epilogue\":\""<<(which?"DRELU_BGRAD":"DRELU")<<"\",\"bias_sentinel\":"<<(sentinel?31:77)<<",\"samples\":[";for(int i=0;i<12;i++){if(i)f<<",";f<<"{\"aux\":"<<aux[i]<<",\"grad\":"<<g[i]<<",\"out\":"<<out[i]<<"}";}f<<"],\"bias\":[";for(int i=0;i<W;i++){if(i)f<<",";f<<bg[i];}f<<"]}";
		hipblasLtMatmulPreferenceDestroy(p);hipblasLtMatmulDescDestroy(d);
	}f<<"\n]}\n";hipblasLtMatrixLayoutDestroy(ld);hipblasLtMatrixLayoutDestroy(lb);hipblasLtMatrixLayoutDestroy(la);hipblasLtDestroy(h);hipFree(dbg);hipFree(do_);hipFree(dx);hipFree(dg);hipFree(da);return 0;
}
