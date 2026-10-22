#pragma once

#অন্তর্ভাৱ <atomic>
#অন্তর্ভাৱ <cassert>
#অন্তর্ভাৱ <cerrno>
#অন্তর্ভাৱ <cstdarg>
#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <cstdio>
#অন্তর্ভাৱ <cstdlib>
#অন্তর্ভাৱ <ctime>

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

স্থির অন্তঃস্থ bool util_is_power_of_two_nonzero64(uint64_t v) {
    ঘূৰাই IS_POT_NONZERO(v);
}

স্থির অন্তঃস্থ uint64_t align64(uint64_t মান, uint64_t alignment) {
    প্রতিজ্ঞা(util_is_power_of_two_nonzero64(alignment));
    ঘূৰাই ALIGN_POT(মান, alignment);
}

সংরচনা list_head {
    list_head * prev;
    list_head * next;
};

সংরচনা util_sparse_array {
    size_t   elem_size;
    unsigned গ্ৰন্থি_size_log2;

    uintptr_t মূল;
};

শূন্য * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
শূন্য   util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t গ্ৰন্থি_size);

অন্তঃস্থ শূন্য os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    যেতিয়ালৈ (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

সংরচনা timer_data {
    long long start;
    long long total;
    long long গুনতি;
};

স্থির অন্তঃস্থ শূন্য start_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
স্থির অন্তঃস্থ long long stop_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->গুনতি += 1;

    ঘূৰাই duration;
}
