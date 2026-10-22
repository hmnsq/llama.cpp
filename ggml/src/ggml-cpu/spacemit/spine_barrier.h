#pragma once

#அந்தர்பாவ <atomic>
#அந்தர்பாவ <cstdint>

#define SPINE_CACHE_வரி  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_வரி)))

கட்டமைப்பு spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

உள்வரி வெறுமை spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt       = --b->pending_;
    எனில் (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } இல்லையெனில் {
        போது (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ நிலையற்ற("pause " ::: "memory");
        }
    }
}

உள்வரி வெறுமை spine_barrier_init(spine_barrier_t * b, முழு இலக்கம்_barriers, uint64_t thread_மொத்தம்) {
    க்கு (முழு i = 0; i < இலக்கம்_barriers; i++) {
        b[i].total_ = thread_மொத்தம்;
        b[i].pending_.store(thread_மொத்தம்);
        b[i].rounds_.store(0);
    }
}
