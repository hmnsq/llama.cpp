#pragma once

#شامل <atomic>
#شامل <cassert>
#شامل <cerrno>
#شامل <cstdarg>
#شامل <cstddef>
#شامل <cstdint>
#شامل <cstdio>
#شامل <cstdlib>
#شامل <ctime>

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x)   __builtin_expect(!!(x), 1)

#ifndef UNUSED
#    define UNUSED(x) (تهی) (x)
#endif

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

ایستا درونخطی bool util_is_power_of_two_nonzero64(uint64_t v) {
    بازگشت IS_POT_NONZERO(v);
}

ایستا درونخطی uint64_t align64(uint64_t مقدار, uint64_t alignment) {
    أكد(util_is_power_of_two_nonzero64(alignment));
    بازگشت ALIGN_POT(مقدار, alignment);
}

ساختار list_head {
    list_head * prev;
    list_head * next;
};

ساختار util_sparse_array {
    size_t   elem_size;
    unsigned گره_size_log2;

    uintptr_t ریشه;
};

تهی * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
تهی   util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t گره_size);

درونخطی تهی os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    وقتی (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

ساختار timer_data {
    long long start;
    long long total;
    long long شمار;
};

ایستا درونخطی تهی start_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
ایستا درونخطی long long stop_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->شمار += 1;

    بازگشت duration;
}
