#pragma once

#شامل <atomic>
#شامل <cstdint>

#define SPINE_CACHE_سطر  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_سطر)))

ساختار spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

درونخطی تهی spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt       = --b->pending_;
    اگر (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } وگرنه {
        وقتی (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ فرار("pause " ::: "memory");
        }
    }
}

درونخطی تهی spine_barrier_init(spine_barrier_t * b, صحیح رقم_barriers, uint64_t thread_شمار) {
    برای (صحیح i = 0; i < رقم_barriers; i++) {
        b[i].total_ = thread_شمار;
        b[i].pending_.store(thread_شمار);
        b[i].rounds_.store(0);
    }
}
