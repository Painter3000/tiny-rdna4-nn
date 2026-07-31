#include "phase3d_inprocess_loop.hpp"

namespace p3d0 {
const char* state_name(LoopState s) {
    switch (s) {
    case LoopState::UNINITIALIZED:return "UNINITIALIZED";
    case LoopState::ALLOCATED:return "ALLOCATED";
    case LoopState::STATE_LOADED:return "STATE_LOADED";
    case LoopState::STEP_READY:return "STEP_READY";
    case LoopState::FORWARD_COMPLETE:return "FORWARD_COMPLETE";
    case LoopState::BACKWARD_COMPLETE:return "BACKWARD_COMPLETE";
    case LoopState::REDUCTION_COMPLETE:return "REDUCTION_COMPLETE";
    case LoopState::ADAM_COMPLETE:return "ADAM_COMPLETE";
    case LoopState::CAST_COMPLETE:return "CAST_COMPLETE";
    case LoopState::STEP_COMMITTED:return "STEP_COMMITTED";
    case LoopState::FINISHED:return "FINISHED";
    case LoopState::FAILED:return "FAILED";
    }
    return "UNKNOWN";
}

void StateMachine::fail(const std::string& reason) {
    state_ = LoopState::FAILED;
    failure_reason_ = reason;
}

void StateMachine::transition(LoopState next, uint32_t step) {
    if (state_ == LoopState::FAILED) throw std::logic_error("P3D0-STATE-already-failed");
    bool ok = false;
    if (state_ == LoopState::UNINITIALIZED && next == LoopState::ALLOCATED) ok = true;
    else if (state_ == LoopState::ALLOCATED && next == LoopState::STATE_LOADED) {
        ok = true; committed_step_ = step;
    }
    else if ((state_ == LoopState::STATE_LOADED || state_ == LoopState::STEP_COMMITTED)
             && next == LoopState::STEP_READY && step == committed_step_ + 1) {
        ok = true; active_step_ = step;
    } else if (state_ == LoopState::STEP_READY && next == LoopState::FORWARD_COMPLETE && step == active_step_) ok = true;
    else if (state_ == LoopState::FORWARD_COMPLETE && next == LoopState::BACKWARD_COMPLETE && step == active_step_) ok = true;
    else if (state_ == LoopState::BACKWARD_COMPLETE && next == LoopState::REDUCTION_COMPLETE && step == active_step_) ok = true;
    else if (state_ == LoopState::REDUCTION_COMPLETE && next == LoopState::ADAM_COMPLETE && step == active_step_) ok = true;
    else if (state_ == LoopState::ADAM_COMPLETE && next == LoopState::CAST_COMPLETE && step == active_step_) ok = true;
    else if (state_ == LoopState::CAST_COMPLETE && next == LoopState::STEP_COMMITTED && step == active_step_) {
        ok = true; committed_step_ = step;
    } else if (state_ == LoopState::STEP_COMMITTED && next == LoopState::FINISHED && step == committed_step_) ok = true;
    if (!ok) {
        fail(std::string("P3D0-STATE-illegal-transition:") + state_name(state_) + "->" + state_name(next));
        throw std::logic_error(failure_reason_);
    }
    state_ = next;
}
}
