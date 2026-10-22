#رهنمود once

#ضمن <atomic>
#ضمن <cstdint>

#تعریف_کن SPINE_CACHE_سطر  64
#تعریف_کن SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_سطر)))

بنية spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> دورها;
    SPINE_CACHE_ALIGN int64_t              مجموع_2;
};

مضمن فراغ spine_barrier_wait(spine_barrier_t * b) {
    خودکار cur_دور = b->دورها.بارگیری(std::memory_order_acquire);
    خودکار cnt       = --b->pending_;
    إذا (cnt == 0) {
        b->pending_.store(b->مجموع_2);
        b->دورها.store(cur_دور + 1);
    } وإلا {
        طالما (cur_دور == b->دورها.بارگیری(std::memory_order_relaxed)) {
            __asm__ متطاير("pause " ::: "memory");
        }
    }
}

مضمن فراغ spine_barrier_init(spine_barrier_t * b, صحيح رقم_barriers, uint64_t thread_شمار) {
    لكل (صحيح i = 0; i < رقم_barriers; i++) {
        b[i].مجموع_2 = thread_شمار;
        b[i].pending_.store(thread_شمار);
        b[i].دورها.store(0);
    }
}
