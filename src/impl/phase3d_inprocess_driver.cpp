// P3D0-LOOP-001: orchestration only; qualified kernels are included unchanged.
#include "phase3d_inprocess_loop.hpp"
#define main phase3a_frozen_standalone_main
#include "../phase3a_fused_backward/impl/native/phase3a_fused_backward.hip"
#undef main
#undef HIP_CHECK
#define main phase3b_adam_frozen_standalone_main
#include "../phase3b_training_adam/impl/native/phase3b_adam_update.hip"
#undef main

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;
namespace {
struct StepHeader { uint32_t magic, rows; };
constexpr uint32_t COUNT = 3 * 64 * 64;

template<class T> std::vector<T> read_vec(const fs::path& p, size_t n) {
    std::vector<T> v(n); std::ifstream f(p, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
    if (!f || f.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("P3D0-BRIDGE-read:" + p.string());
    return v;
}
template<class T> void write_vec(const fs::path& p, const std::vector<T>& v) {
    std::ofstream f(p, std::ios::binary); f.write(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(T));
    if (!f) throw std::runtime_error("P3D0-BRIDGE-write:" + p.string());
}
template<class T> void alloc(T** p, size_t bytes) { HIP_CHECK(hipMalloc(p, bytes)); if (!*p) throw std::runtime_error("P3D0-LOOP-null"); }

struct DeviceState {
    float *master{}, *m{}, *v{};
    Half *compute{}, *x{}, *target{}, *dy{}, *h0{}, *h1{}, *y{}, *dz1{}, *dz0{};
    float *z0{}, *z1{}, *dx{}, *parts{}, *dw{};
    uint16_t* witness{};
    uint32_t rows{}, tiles{};
    size_t hb{}, fb{};
    std::array<std::pair<std::string,void*>,14> stable{};
    void allocate(uint32_t r) {
        rows=r;tiles=(r+TILE-1)/TILE;hb=r*WIDTH*sizeof(Half);fb=r*WIDTH*sizeof(float);
        alloc(&master,COUNT*sizeof(float));alloc(&compute,COUNT*sizeof(Half));alloc(&m,COUNT*sizeof(float));alloc(&v,COUNT*sizeof(float));
        alloc(&x,hb);alloc(&target,hb);alloc(&dy,hb);alloc(&h0,hb);alloc(&h1,hb);alloc(&y,hb);alloc(&dz1,hb);alloc(&dz0,hb);
        alloc(&z0,fb);alloc(&z1,fb);alloc(&dx,fb);alloc(&parts,3*tiles*WIDTH*WIDTH*sizeof(float));alloc(&dw,COUNT*sizeof(float));alloc(&witness,4*256*sizeof(uint16_t));
        stable={{{"W_master",master},{"W_compute",compute},{"m",m},{"v",v},{"x",x},{"target",target},{"forward",y},{"dY",dy},{"dX",dx},{"dW",dw},{"parts",parts},{"z0",z0},{"z1",z1},{"witness",witness}}};
    }
    void verify_stable() const {
        for (auto& [id,p]:stable) if (!p || (reinterpret_cast<uintptr_t>(p)%alignof(float))) throw std::runtime_error("P3D0-LOOP-buffer:"+id);
    }
    void release() {
        for (void* p:{(void*)master,(void*)compute,(void*)m,(void*)v,(void*)x,(void*)target,(void*)dy,(void*)h0,(void*)h1,(void*)y,(void*)dz1,(void*)dz0,(void*)z0,(void*)z1,(void*)dx,(void*)parts,(void*)dw,(void*)witness}) HIP_CHECK(hipFree(p));
    }
};

std::vector<uint8_t> bytes(const fs::path& p) {
    std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};
}
}

int main(int argc,char**argv) {
    if (argc!=5 && argc!=7) {
        std::cerr<<"usage: DRIVER INPUT_ROOT OUTPUT STEPS CASE_ID | DRIVER INPUT_ROOT OUTPUT START END CASE_ID STATE_DIR\n";
        return 2;
    }
    fs::path ref=argv[1],out=argv[2],state_dir;
    uint32_t start=1,end=0;std::string cid;
    if(argc==5){end=std::stoul(argv[3]);cid=argv[4];state_dir=ref/"step_0";}
    else{start=std::stoul(argv[3]);end=std::stoul(argv[4]);cid=argv[5];state_dir=argv[6];}
    if(start<1||end<start||end>100)return 3;fs::create_directories(out);
    try {
        auto pkg=bytes(ref/"step_1/input.bin");if(pkg.size()<8)throw std::runtime_error("P3D0-BRIDGE-input");
        StepHeader hd{};std::memcpy(&hd,pkg.data(),8);if(hd.magic!=0x50334231u||hd.rows<1||hd.rows>64)return 4;
        DeviceState d;p3d0::StateMachine sm;d.allocate(hd.rows);sm.transition(p3d0::LoopState::ALLOCATED,0);
        auto master=read_vec<float>(state_dir/"W_master.fp32.bin",COUNT);
        auto compute=read_vec<uint16_t>(state_dir/"W_compute.fp16.bin",COUNT);
        auto hm=read_vec<float>(state_dir/"m.fp32.bin",COUNT),hv=read_vec<float>(state_dir/"v.fp32.bin",COUNT);
        HIP_CHECK(hipMemcpy(d.master,master.data(),COUNT*4,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(d.compute,compute.data(),COUNT*2,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d.m,hm.data(),COUNT*4,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(d.v,hv.data(),COUNT*4,hipMemcpyHostToDevice));
        sm.transition(p3d0::LoopState::STATE_LOADED,start-1);d.verify_stable();
        std::ofstream inv(out/"allocation_inventory.txt");for(auto&[id,p]:d.stable)inv<<id<<" "<<p<<"\n";
        float b1p=1.0f,b2p=1.0f;uint32_t optimizer_step=start-1;
        if(start>1) {
            std::ifstream meta(state_dir/"optimizer_state.txt");
            meta>>optimizer_step>>b1p>>b2p;
            if(!meta||optimizer_step!=start-1)throw std::runtime_error("P3DA-RESUME-optimizer-state");
        } else for(uint32_t i=1;i<start;++i){b1p*=0.9f;b2p*=0.999f;}
        for(uint32_t step=start;step<=end;++step) {
            sm.transition(p3d0::LoopState::STEP_READY,step);d.verify_stable();
            pkg=bytes(ref/("step_"+std::to_string(step))/"input.bin");StepHeader sh{};std::memcpy(&sh,pkg.data(),8);
            size_t n=sh.rows*64,off=8;
            if(sh.rows!=d.rows||(pkg.size()!=8+4*n&&pkg.size()!=8+4*n+COUNT*2))throw std::runtime_error("P3D0-BRIDGE-input-shape");
            std::vector<uint16_t> hx(n),ht(n),expected_compute(COUNT);
            std::memcpy(hx.data(),pkg.data()+off,n*2);off+=n*2;std::memcpy(ht.data(),pkg.data()+off,n*2);off+=n*2;
            if(pkg.size()==8+4*n+COUNT*2) {
                std::memcpy(expected_compute.data(),pkg.data()+off,COUNT*2);
                std::vector<uint16_t> live_compute(COUNT);HIP_CHECK(hipMemcpy(live_compute.data(),d.compute,COUNT*2,hipMemcpyDeviceToHost));
                if(live_compute!=expected_compute)throw std::runtime_error("P3D0-BRIDGE-stale-W_compute");
            }
            HIP_CHECK(hipMemcpy(d.x,hx.data(),d.hb,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(d.target,ht.data(),d.hb,hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d.parts,0,3*d.tiles*WIDTH*WIDTH*sizeof(float)));HIP_CHECK(hipMemset(d.witness,0,4*256*sizeof(uint16_t)));
            Half* w0=d.compute;Half* w1=d.compute+4096;Half* w2=d.compute+8192;
            hipLaunchKernelGGL(phase3a_forward_saved_kernel,dim3(d.rows),dim3(WIDTH),0,nullptr,d.x,w0,w1,w2,d.rows,d.z0,d.h0,d.z1,d.h1,d.y);
            HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());sm.transition(p3d0::LoopState::FORWARD_COMPLETE,step);
            std::vector<Half> hy(n),hdy(n);HIP_CHECK(hipMemcpy(hy.data(),d.y,d.hb,hipMemcpyDeviceToHost));
            for(size_t i=0;i<n;++i)hdy[i]=static_cast<Half>(0.03125f*(static_cast<float>(hy[i])-static_cast<float>(reinterpret_cast<Half*>(ht.data())[i])));
            HIP_CHECK(hipMemcpy(d.dy,hdy.data(),d.hb,hipMemcpyHostToDevice));
            hipLaunchKernelGGL(phase3a_backward_tile_kernel,dim3(d.tiles),dim3(THREADS),0,nullptr,d.x,w0,w1,w2,d.z0,d.h0,d.z1,d.h1,d.dy,d.rows,d.dz1,d.dz0,d.dx,d.parts,d.witness);
            HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());sm.transition(p3d0::LoopState::BACKWARD_COMPLETE,step);
            hipLaunchKernelGGL(phase3a_dw_fixed_order_reduce_kernel,dim3((COUNT+255)/256),dim3(256),0,nullptr,d.parts,d.tiles,d.dw);
            HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());sm.transition(p3d0::LoopState::REDUCTION_COMPLETE,step);
            b1p=b1p*0.9f;b2p=b2p*0.999f;
            hipLaunchKernelGGL(phase3b_adam_fp32_master_update_kernel,dim3((COUNT+255)/256),dim3(256),0,nullptr,d.master,reinterpret_cast<__half*>(d.compute),d.m,d.v,d.dw,COUNT,0.9f,0.999f,b1p,b2p,1.0e-8f,1.0e-3f);
            HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());sm.transition(p3d0::LoopState::ADAM_COMPLETE,step);
            ++optimizer_step;sm.transition(p3d0::LoopState::CAST_COMPLETE,step);
            fs::path sd=out/("step_"+std::to_string(step));fs::create_directories(sd);
            std::vector<float> fwdx(n),gdw(COUNT);std::vector<uint16_t> fy(n);
            HIP_CHECK(hipMemcpy(fy.data(),d.y,d.hb,hipMemcpyDeviceToHost));HIP_CHECK(hipMemcpy(fwdx.data(),d.dx,d.fb,hipMemcpyDeviceToHost));HIP_CHECK(hipMemcpy(gdw.data(),d.dw,COUNT*4,hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(master.data(),d.master,COUNT*4,hipMemcpyDeviceToHost));HIP_CHECK(hipMemcpy(compute.data(),d.compute,COUNT*2,hipMemcpyDeviceToHost));HIP_CHECK(hipMemcpy(hm.data(),d.m,COUNT*4,hipMemcpyDeviceToHost));HIP_CHECK(hipMemcpy(hv.data(),d.v,COUNT*4,hipMemcpyDeviceToHost));
            write_vec(sd/"forward.fp16.bin",fy);write_vec(sd/"dY.fp16.bin",std::vector<uint16_t>(reinterpret_cast<uint16_t*>(hdy.data()),reinterpret_cast<uint16_t*>(hdy.data())+n));
            write_vec(sd/"dX.fp32.bin",fwdx);write_vec(sd/"dW.fp32.bin",gdw);write_vec(sd/"W_master.fp32.bin",master);write_vec(sd/"W_compute.fp16.bin",compute);write_vec(sd/"m.fp32.bin",hm);write_vec(sd/"v.fp32.bin",hv);
            std::ofstream meta(sd/"optimizer_state.txt");meta<<optimizer_step<<"\n"<<std::setprecision(9)<<b1p<<"\n"<<b2p<<"\n";
            sm.transition(p3d0::LoopState::STEP_COMMITTED,step);
        }
        sm.transition(p3d0::LoopState::FINISHED,end);d.verify_stable();d.release();
        std::cout<<"P3D0-GATE-INPROCESS-LOOP: PASS case="<<cid<<" start="<<start<<" end="<<end<<"\n";return 0;
    } catch(const std::exception& e) { std::cerr<<"P3D0-GATE-FAIL:"<<e.what()<<"\n";return 10; }
}
