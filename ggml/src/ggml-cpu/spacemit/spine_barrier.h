#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SPINE_CACHE_ਸਤਰ  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_ਸਤਰ)))

ਬਣਤਰ spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> ਗੇੜ;
    SPINE_CACHE_ALIGN int64_t              ਕੁੱਲ_2;
};

ਸਤਰਅੰਦਰ ਖਾਲੀ spine_barrier_wait(spine_barrier_t * b) {
    ਸਵੈਚਾਲਿਤ cur_ਗੇੜਾ = b->ਗੇੜ.ਲੱਦੋ(std::memory_order_acquire);
    ਸਵੈਚਾਲਿਤ cnt       = --b->pending_;
    ਜੇ (cnt == 0) {
        b->pending_.store(b->ਕੁੱਲ_2);
        b->ਗੇੜ.store(cur_ਗੇੜਾ + 1);
    } ਨਹੀਂਤਾਂ {
        ਜਦੋਂ (cur_ਗੇੜਾ == b->ਗੇੜ.ਲੱਦੋ(std::memory_order_relaxed)) {
            __asm__ ਅਸਥਿਰ("pause " ::: "memory");
        }
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ spine_barrier_init(spine_barrier_t * b, ਪੂਰਨ ਅੰਕ_barriers, uint64_t thread_ਗਿਣਤੀ) {
    ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_barriers; i++) {
        b[i].ਕੁੱਲ_2 = thread_ਗਿਣਤੀ;
        b[i].pending_.store(thread_ਗਿਣਤੀ);
        b[i].ਗੇੜ.store(0);
    }
}
