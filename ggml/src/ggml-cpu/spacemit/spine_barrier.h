#आशय once

#समावेश <atomic>
#समावेश <cstdint>

#घोषणा SPINE_CACHE_पंक्ति  64
#घोषणा SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_पंक्ति)))

काष्ठा spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              योग_3;
};

अंतरभूत व्योम spine_barrier_wait(spine_barrier_t * b) {
    स्वतः cur_round = b->rounds_.load(std::memory_order_acquire);
    स्वतः cnt       = --b->pending_;
    अगर (cnt == 0) {
        b->pending_.store(b->योग_3);
        b->rounds_.store(cur_round + 1);
    } अन्यथा {
        जबतक (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ अस्थिर("pause " ::: "memory");
        }
    }
}

अंतरभूत व्योम spine_barrier_init(spine_barrier_t * b, पूर्णांक अंक_barriers, uint64_t thread_गिनती) {
    क्रम (पूर्णांक i = 0; i < अंक_barriers; i++) {
        b[i].योग_3 = thread_गिनती;
        b[i].pending_.store(thread_गिनती);
        b[i].rounds_.store(0);
    }
}
