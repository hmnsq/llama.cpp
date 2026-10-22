#निर्देशः once

#अन्तर्भाव <atomic>
#अन्तर्भाव <cstdint>

#परिभाषय SPINE_CACHE_पङ्क्तिः  64
#परिभाषय SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_पङ्क्तिः)))

संरचना spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> आवर्तनानि;
    SPINE_CACHE_ALIGN int64_t              योग_3;
};

अन्तःस्थ शून्य spine_barrier_wait(spine_barrier_t * b) {
    स्वचालित cur_आवर्तनम् = b->आवर्तनानि.आरोपय(std::memory_order_acquire);
    स्वचालित cnt       = --b->pending_;
    यदि (cnt == 0) {
        b->pending_.store(b->योग_3);
        b->आवर्तनानि.store(cur_आवर्तनम् + 1);
    } अन्यथा {
        यावत् (cur_आवर्तनम् == b->आवर्तनानि.आरोपय(std::memory_order_relaxed)) {
            __asm__ चञ्चल("pause " ::: "memory");
        }
    }
}

अन्तःस्थ शून्य spine_barrier_init(spine_barrier_t * b, पूर्णाङ्क अङ्कः_barriers, uint64_t thread_गणनम्) {
    प्रति (पूर्णाङ्क i = 0; i < अङ्कः_barriers; i++) {
        b[i].योग_3 = thread_गणनम्;
        b[i].pending_.store(thread_गणनम्);
        b[i].आवर्तनानि.store(0);
    }
}
