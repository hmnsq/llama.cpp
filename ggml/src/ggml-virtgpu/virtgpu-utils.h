#pragma once

#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x)   __builtin_expect(!!(x), 1)

#ifndef UNUSED
#    define UNUSED(x) (void) (x)
#endif

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

static anta_hstha bool util_is_power_of_two_nonzero64(uint64_t v) {
    ပြန်ပေး IS_POT_NONZERO(v);
}

static anta_hstha uint64_t align64(uint64_t value, uint64_t alignment) {
    assert(util_is_power_of_two_nonzero64(alignment));
    ပြန်ပေး ALIGN_POT(value, alignment);
}

sa_nracanaa list_head {
    list_head * prev;
    list_head * next;
};

sa_nracanaa util_sparse_array {
    size_t   elem_size;
    unsigned node_size_log2;

    uintptr_t root;
};

void * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
void   util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t node_size);

anta_hstha void os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    နေစဉ် (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

sa_nracanaa timer_data {
    long long start;
    long long total;
    long long count;
};

static anta_hstha void start_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
static anta_hstha long long stop_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->count += 1;

    ပြန်ပေး duration;
}
