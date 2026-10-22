#ہدایت once

#ضمن <atomic>
#ضمن <cstdint>

#تعریف_کرو SPINE_CACHE_سطر  64
#تعریف_کرو SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_سطر)))

بنية spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> گیڑ;
    SPINE_CACHE_ALIGN int64_t              کل_2;
};

مضمن فراغ spine_barrier_wait(spine_barrier_t * b) {
    خودکار cur_گیڑا = b->گیڑ.لدو(std::memory_order_acquire);
    خودکار cnt       = --b->pending_;
    إذا (cnt == 0) {
        b->pending_.store(b->کل_2);
        b->گیڑ.store(cur_گیڑا + 1);
    } وإلا {
        طالما (cur_گیڑا == b->گیڑ.لدو(std::memory_order_relaxed)) {
            __asm__ متطاير("pause " ::: "memory");
        }
    }
}

مضمن فراغ spine_barrier_init(spine_barrier_t * b, صحيح عدد_barriers, uint64_t thread_گنتی) {
    لكل (صحيح i = 0; i < عدد_barriers; i++) {
        b[i].کل_2 = thread_گنتی;
        b[i].pending_.store(thread_گنتی);
        b[i].گیڑ.store(0);
    }
}
