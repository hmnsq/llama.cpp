#ہدایت once

#شامل <atomic>
#شامل <cstdint>

#تعریف_کرو SPINE_CACHE_سطر  64
#تعریف_کرو SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_سطر)))

ساخت spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> دور;
    SPINE_CACHE_ALIGN int64_t              کل_2;
};

اندرونی خالی spine_barrier_wait(spine_barrier_t * b) {
    خودکار cur_باری = b->دور.بھرو(std::memory_order_acquire);
    خودکار cnt       = --b->pending_;
    اگر (cnt == 0) {
        b->pending_.store(b->کل_2);
        b->دور.store(cur_باری + 1);
    } ورنہ {
        جب (cur_باری == b->دور.بھرو(std::memory_order_relaxed)) {
            __asm__ متلون("pause " ::: "memory");
        }
    }
}

اندرونی خالی spine_barrier_init(spine_barrier_t * b, صحیح عدد_barriers, uint64_t thread_گنتی) {
    برائے (صحیح i = 0; i < عدد_barriers; i++) {
        b[i].کل_2 = thread_گنتی;
        b[i].pending_.store(thread_گنتی);
        b[i].دور.store(0);
    }
}
