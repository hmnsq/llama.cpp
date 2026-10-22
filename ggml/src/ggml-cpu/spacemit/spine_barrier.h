#هدايت once

#ضمن <atomic>
#ضمن <cstdint>

#وصف_ڪريو SPINE_CACHE_سٽ  64
#وصف_ڪريو SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_سٽ)))

بنية spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> ڦيرا;
    SPINE_CACHE_ALIGN int64_t              ڪل_2;
};

مضمن فراغ spine_barrier_wait(spine_barrier_t * b) {
    خودڪار cur_ڦيرو = b->ڦيرا.ڀريو(std::memory_order_acquire);
    خودڪار cnt       = --b->pending_;
    إذا (cnt == 0) {
        b->pending_.store(b->ڪل_2);
        b->ڦيرا.store(cur_ڦيرو + 1);
    } وإلا {
        طالما (cur_ڦيرو == b->ڦيرا.ڀريو(std::memory_order_relaxed)) {
            __asm__ متطاير("pause " ::: "memory");
        }
    }
}

مضمن فراغ spine_barrier_init(spine_barrier_t * b, صحيح انگ_barriers, uint64_t thread_ڳڻپ) {
    لكل (صحيح i = 0; i < انگ_barriers; i++) {
        b[i].ڪل_2 = thread_ڳڻپ;
        b[i].pending_.store(thread_ڳڻپ);
        b[i].ڦيرا.store(0);
    }
}
