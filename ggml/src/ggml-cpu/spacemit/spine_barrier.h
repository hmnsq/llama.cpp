#సూచన once

#అన్తర్భావ <atomic>
#అన్తర్భావ <cstdint>

#నిర్వచించు SPINE_CACHE_పంక్తి  64
#నిర్వచించు SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_పంక్తి)))

నిర్మాణం spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> విడతలు;
    SPINE_CACHE_ALIGN int64_t              మొత్తం_2;
};

అంతఃవరుస శూన్య spine_barrier_wait(spine_barrier_t * b) {
    స్వయంచాలక cur_విడత = b->విడతలు.ఎక్కించు(std::memory_order_acquire);
    స్వయంచాలక cnt       = --b->pending_;
    ఒకవేళ (cnt == 0) {
        b->pending_.store(b->మొత్తం_2);
        b->విడతలు.store(cur_విడత + 1);
    } లేకపోతే {
        అయితే (cur_విడత == b->విడతలు.ఎక్కించు(std::memory_order_relaxed)) {
            __asm__ అస్థిర("pause " ::: "memory");
        }
    }
}

అంతఃవరుస శూన్య spine_barrier_init(spine_barrier_t * b, పూర్ణాంకం అంకె_barriers, uint64_t thread_లెక్క) {
    కొరకు (పూర్ణాంకం i = 0; i < అంకె_barriers; i++) {
        b[i].మొత్తం_2 = thread_లెక్క;
        b[i].pending_.store(thread_లెక్క);
        b[i].విడతలు.store(0);
    }
}
