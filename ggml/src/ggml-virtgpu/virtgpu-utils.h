#pragma once

#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <cstdarg>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <ctime>

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x)   __builtin_expect(!!(x), 1)

#ifndef UNUSED
#    define UNUSED(x) (ಶೂನ್ಯ) (x)
#endif

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool util_is_power_of_two_nonzero64(uint64_t v) {
    ಪ್ರತ್ಯಾಗಮ IS_POT_NONZERO(v);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint64_t align64(uint64_t ಮೌಲ್ಯ, uint64_t alignment) {
    ಪ್ರತಿಜ್ಞಾ(util_is_power_of_two_nonzero64(alignment));
    ಪ್ರತ್ಯಾಗಮ ALIGN_POT(ಮೌಲ್ಯ, alignment);
}

ಸಂರಚನಾ list_head {
    list_head * prev;
    list_head * next;
};

ಸಂರಚನಾ util_sparse_array {
    size_t   elem_size;
    unsigned ಗಂಟು_size_log2;

    uintptr_t ಬೇರು;
};

ಶೂನ್ಯ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
ಶೂನ್ಯ   util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t ಗಂಟು_size);

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    ಯಾವತ್ (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

ಸಂರಚನಾ timer_data {
    long long start;
    long long total;
    long long ಎಣಿಕೆ;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ start_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ long long stop_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->ಎಣಿಕೆ += 1;

    ಪ್ರತ್ಯಾಗಮ duration;
}
