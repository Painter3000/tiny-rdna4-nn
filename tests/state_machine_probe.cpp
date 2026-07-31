#include "../src/impl/phase3d_inprocess_loop.hpp"
#include <iostream>
int main() {
    p3d0::StateMachine s;
    s.transition(p3d0::LoopState::ALLOCATED,0);
    s.transition(p3d0::LoopState::STATE_LOADED,0);
    for (unsigned step=1;step<=4;++step) {
        s.transition(p3d0::LoopState::STEP_READY,step);
        s.transition(p3d0::LoopState::FORWARD_COMPLETE,step);
        s.transition(p3d0::LoopState::BACKWARD_COMPLETE,step);
        s.transition(p3d0::LoopState::REDUCTION_COMPLETE,step);
        s.transition(p3d0::LoopState::ADAM_COMPLETE,step);
        s.transition(p3d0::LoopState::CAST_COMPLETE,step);
        s.transition(p3d0::LoopState::STEP_COMMITTED,step);
    }
    s.transition(p3d0::LoopState::FINISHED,4);
    std::cout<<"INPROCESS_LOOP_STATE_MACHINE: PASS\n";
}
