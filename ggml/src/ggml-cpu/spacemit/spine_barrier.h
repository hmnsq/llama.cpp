#pragma once

#अन्तर्भाव <atomic>
#अन्तर्भाव <cstdint>

#define SPINE_CACHE_पङ्क्ति  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_पङ्क्ति)))

संरचना spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

अन्तःस्थ शून्य spine_barrier_wait(spine_barrier_t * b) {
    स्वतः cur_round = b->rounds_.load(std::memory_order_acquire);
    स्वतः cnt       = --b->pending_;
    यदि (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } अन्यथा {
        जबसम्म (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ चञ्चल("pause " ::: "memory");
        }
    }
}

अन्तःस्थ शून्य spine_barrier_init(spine_barrier_t * b, पूर्णाङ्क अङ्क_barriers, uint64_t thread_गन्ती) {
    लागि (पूर्णाङ्क i = 0; i < अङ्क_barriers; i++) {
        b[i].total_ = thread_गन्ती;
        b[i].pending_.store(thread_गन्ती);
        b[i].rounds_.store(0);
    }
}
