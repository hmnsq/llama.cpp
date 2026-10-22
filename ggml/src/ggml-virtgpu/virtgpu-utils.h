#pragma once

#അന്തര്ഭാവ <atomic>
#അന്തര്ഭാവ <cassert>
#അന്തര്ഭാവ <cerrno>
#അന്തര്ഭാവ <cstdarg>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstdio>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <ctime>

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x)   __builtin_expect(!!(x), 1)

#ifndef UNUSED
#    define UNUSED(x) (ശൂന്യം) (x)
#endif

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

സ്ഥിതം വരിയിൽ bool util_is_power_of_two_nonzero64(uint64_t v) {
    തിരികെ IS_POT_NONZERO(v);
}

സ്ഥിതം വരിയിൽ uint64_t align64(uint64_t മൂല്യം, uint64_t alignment) {
    പ്രതിജ്ഞാ(util_is_power_of_two_nonzero64(alignment));
    തിരികെ ALIGN_POT(മൂല്യം, alignment);
}

ഘടന list_head {
    list_head * prev;
    list_head * next;
};

ഘടന util_sparse_array {
    size_t   elem_size;
    unsigned കണ്ണി_size_log2;

    uintptr_t വേര്;
};

ശൂന്യം * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
ശൂന്യം   util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t കണ്ണി_size);

വരിയിൽ ശൂന്യം os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    ആകുമ്പോൾ (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

ഘടന timer_data {
    long long start;
    long long total;
    long long എണ്ണം;
};

സ്ഥിതം വരിയിൽ ശൂന്യം start_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
സ്ഥിതം വരിയിൽ long long stop_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->എണ്ണം += 1;

    തിരികെ duration;
}
