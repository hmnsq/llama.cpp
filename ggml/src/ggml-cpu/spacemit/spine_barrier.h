#pragma once

#ଅନ୍ତର୍ଭାଵ <atomic>
#ଅନ୍ତର୍ଭାଵ <cstdint>

#define SPINE_CACHE_ଧାଡ଼ି  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_ଧାଡ଼ି)))

ସଂରଚନା spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt       = --b->pending_;
    ଯଦି (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } ଅନ୍ୟଥା {
        ଯେତେବେଳେ (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ ଅସ୍ଥିର("pause " ::: "memory");
        }
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ spine_barrier_init(spine_barrier_t * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_barriers, uint64_t thread_ଗଣନା) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଅଙ୍କ_barriers; i++) {
        b[i].total_ = thread_ଗଣନା;
        b[i].pending_.store(thread_ଗଣନା);
        b[i].rounds_.store(0);
    }
}
