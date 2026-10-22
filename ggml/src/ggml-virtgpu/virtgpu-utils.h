#pragma once

#অন্তর্ভাব <atomic>
#অন্তর্ভাব <cassert>
#অন্তর্ভাব <cerrno>
#অন্তর্ভাব <cstdarg>
#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstdio>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <ctime>

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x)   __builtin_expect(!!(x), 1)

#ifndef UNUSED
#    define UNUSED(x) (শূন্য) (x)
#endif

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

স্থির অন্তঃরেখ bool util_is_power_of_two_nonzero64(uint64_t v) {
    ফেরত IS_POT_NONZERO(v);
}

স্থির অন্তঃরেখ uint64_t align64(uint64_t মান, uint64_t alignment) {
    প্রতিজ্ঞা(util_is_power_of_two_nonzero64(alignment));
    ফেরত ALIGN_POT(মান, alignment);
}

কাঠামো list_শীর্ষ {
    list_শীর্ষ * পূর্ব;
    list_শীর্ষ * পরবর্তী;
};

কাঠামো util_sparse_array {
    size_t   elem_আকার;
    unsigned গ্রন্থি_আকার_log2;

    uintptr_t মূল;
};

শূন্য * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
শূন্য   util_sparse_array_init(util_sparse_array * arr, size_t elem_আকার, size_t গ্রন্থি_আকার);

অন্তঃরেখ শূন্য os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    যখন (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

কাঠামো timer_তথ্য {
    long long start;
    long long যোগ;
    long long গুনতি;
};

স্থির অন্তঃরেখ শূন্য start_timer(timer_তথ্য * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
স্থির অন্তঃরেখ long long stop_timer(timer_তথ্য * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->যোগ += duration;
    timer->গুনতি += 1;

    ফেরত duration;
}
