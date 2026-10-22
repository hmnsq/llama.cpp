#pragma once

#include <atomic>
#include <cstdint>

#define SPINE_CACHE_लिन्या  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_लिन्या)))

struct spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

inline void spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt       = --b->pending_;
    if (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } else {
        while (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ volatile("pause " ::: "memory");
        }
    }
}

inline void spine_barrier_init(spine_barrier_t * b, int नुमेरो_barriers, uint64_t thread_बिलांग) {
    for (int i = 0; i < नुमेरो_barriers; i++) {
        b[i].total_ = thread_बिलांग;
        b[i].pending_.store(thread_बिलांग);
        b[i].rounds_.store(0);
    }
}
