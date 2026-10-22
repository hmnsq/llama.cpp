#pragma once

#अन्तर्भाव <atomic>
#अन्तर्भाव <cstdint>

#define SPINE_CACHE_पङ्क्तिः  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_पङ्क्तिः)))

संरचना spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

अन्तःस्थ शून्य spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt       = --b->pending_;
    जइ (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } अण्णहा {
        जाव (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ चञ्चल("pause " ::: "memory");
        }
    }
}

अन्तःस्थ शून्य spine_barrier_init(spine_barrier_t * b, पूर्णाङ्क अङ्कः_barriers, uint64_t thread_गणनम्) {
    प्रति (पूर्णाङ्क i = 0; i < अङ्कः_barriers; i++) {
        b[i].total_ = thread_गणनम्;
        b[i].pending_.store(thread_गणनम्);
        b[i].rounds_.store(0);
    }
}
