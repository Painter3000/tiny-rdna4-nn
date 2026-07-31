#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

namespace p3d0 {
enum class LoopState {
    UNINITIALIZED, ALLOCATED, STATE_LOADED, STEP_READY, FORWARD_COMPLETE,
    BACKWARD_COMPLETE, REDUCTION_COMPLETE, ADAM_COMPLETE, CAST_COMPLETE,
    STEP_COMMITTED, FINISHED, FAILED
};

const char* state_name(LoopState state);

class StateMachine {
public:
    void transition(LoopState next, uint32_t step_id);
    void fail(const std::string& reason);
    LoopState state() const { return state_; }
    uint32_t committed_step() const { return committed_step_; }
    const std::string& failure_reason() const { return failure_reason_; }
private:
    LoopState state_ = LoopState::UNINITIALIZED;
    uint32_t active_step_ = 0;
    uint32_t committed_step_ = 0;
    std::string failure_reason_;
};
}
