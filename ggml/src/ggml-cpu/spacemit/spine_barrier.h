#pragma once

#অন্তর্ভাব <atomic>
#অন্তর্ভাব <cstdint>

#define SPINE_CACHE_সারি  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_সারি)))

সংরচনা spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

অন্তঃস্থ শূন্য spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt       = --b->pending_;
    যদি (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } অন্যথা {
        যাবত্ (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ চঞ্চল("pause " ::: "memory");
        }
    }
}

অন্তঃস্থ শূন্য spine_barrier_init(spine_barrier_t * b, পূর্ণাঙ্ক অঙ্ক_barriers, uint64_t thread_গুনতি) {
    প্রতি (পূর্ণাঙ্ক i = 0; i < অঙ্ক_barriers; i++) {
        b[i].total_ = thread_গুনতি;
        b[i].pending_.store(thread_গুনতি);
        b[i].rounds_.store(0);
    }
}
