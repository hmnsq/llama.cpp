#pragma once

#अन्तर्भाव <atomic>
#अन्तर्भाव <cstdint>

#define SPINE_CACHE_ओळ  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_ओळ)))

संरचना spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

ओळीत शून्य spine_barrier_wait(spine_barrier_t * b) {
    स्वतः cur_round = b->rounds_.load(std::memory_order_acquire);
    स्वतः cnt       = --b->pending_;
    जर (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } अन्यथा {
        पर्यंत (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ अस्थिर("pause " ::: "memory");
        }
    }
}

ओळीत शून्य spine_barrier_init(spine_barrier_t * b, पूर्णांक अंक_barriers, uint64_t thread_मोजणी) {
    साठी (पूर्णांक i = 0; i < अंक_barriers; i++) {
        b[i].total_ = thread_मोजणी;
        b[i].pending_.store(thread_मोजणी);
        b[i].rounds_.store(0);
    }
}
