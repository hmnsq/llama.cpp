#لارښوونه once

#ضمن <atomic>
#ضمن <cstdint>

#تعریف_کړه SPINE_CACHE_کرښه  64
#تعریف_کړه SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_کرښه)))

بنية spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> پړاوونه;
    SPINE_CACHE_ALIGN int64_t              ټول_2;
};

مضمن فراغ spine_barrier_wait(spine_barrier_t * b) {
    خپلکاری cur_پړاو = b->پړاوونه.بارول(std::memory_order_acquire);
    خپلکاری cnt       = --b->pending_;
    إذا (cnt == 0) {
        b->pending_.store(b->ټول_2);
        b->پړاوونه.store(cur_پړاو + 1);
    } وإلا {
        طالما (cur_پړاو == b->پړاوونه.بارول(std::memory_order_relaxed)) {
            __asm__ متطاير("pause " ::: "memory");
        }
    }
}

مضمن فراغ spine_barrier_init(spine_barrier_t * b, صحيح عدد_barriers, uint64_t thread_شمېر) {
    لكل (صحيح i = 0; i < عدد_barriers; i++) {
        b[i].ټول_2 = thread_شمېر;
        b[i].pending_.store(thread_شمېر);
        b[i].پړاوونه.store(0);
    }
}
