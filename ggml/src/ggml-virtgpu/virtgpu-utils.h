#pragma once

#அந்தர்பாவ <atomic>
#அந்தர்பாவ <cassert>
#அந்தர்பாவ <cerrno>
#அந்தர்பாவ <cstdarg>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstdio>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <ctime>

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x)   __builtin_expect(!!(x), 1)

#ifndef UNUSED
#    define UNUSED(x) (வெறுமை) (x)
#endif

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

நிலையான உள்வரி bool util_is_power_of_two_nonzero64(uint64_t v) {
    திரும்பு IS_POT_NONZERO(v);
}

நிலையான உள்வரி uint64_t align64(uint64_t மதிப்பு, uint64_t alignment) {
    ப்ரதிஜ்ஞா(util_is_power_of_two_nonzero64(alignment));
    திரும்பு ALIGN_POT(மதிப்பு, alignment);
}

கட்டமைப்பு list_head {
    list_head * prev;
    list_head * next;
};

கட்டமைப்பு util_sparse_array {
    size_t   elem_size;
    unsigned முனை_size_log2;

    uintptr_t வேர்;
};

வெறுமை * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
வெறுமை   util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t முனை_size);

உள்வரி வெறுமை os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    போது (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

கட்டமைப்பு timer_data {
    long long start;
    long long total;
    long long மொத்தம்;
};

நிலையான உள்வரி வெறுமை start_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
நிலையான உள்வரி long long stop_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->மொத்தம் += 1;

    திரும்பு duration;
}
