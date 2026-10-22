#pragma once

#അന്തര്ഭാവ <atomic>
#അന്തര്ഭാവ <cstdint>

#define SPINE_CACHE_വരി  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_വരി)))

ഘടന spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

വരിയിൽ ശൂന്യം spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt       = --b->pending_;
    എങ്കിൽ (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } അല്ലെങ്കിൽ {
        ആകുമ്പോൾ (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ അസ്ഥിര("pause " ::: "memory");
        }
    }
}

വരിയിൽ ശൂന്യം spine_barrier_init(spine_barrier_t * b, പൂർണ്ണസംഖ്യ അക്കം_barriers, uint64_t thread_എണ്ണം) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < അക്കം_barriers; i++) {
        b[i].total_ = thread_എണ്ണം;
        b[i].pending_.store(thread_എണ്ണം);
        b[i].rounds_.store(0);
    }
}
