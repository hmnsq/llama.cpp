#pragma once

#અન્તર્ભાવ <atomic>
#અન્તર્ભાવ <cassert>
#અન્તર્ભાવ <cerrno>
#અન્તર્ભાવ <cstdarg>
#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstdio>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <ctime>

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x)   __builtin_expect(!!(x), 1)

#ifndef UNUSED
#    define UNUSED(x) (શૂન્ય) (x)
#endif

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

સ્થિર અંતઃરેખ bool util_is_power_of_two_nonzero64(uint64_t v) {
    પાછા IS_POT_NONZERO(v);
}

સ્થિર અંતઃરેખ uint64_t align64(uint64_t કિંમત, uint64_t alignment) {
    પ્રતિજ્ઞા(util_is_power_of_two_nonzero64(alignment));
    પાછા ALIGN_POT(કિંમત, alignment);
}

સંરચના list_head {
    list_head * prev;
    list_head * next;
};

સંરચના util_sparse_array {
    size_t   elem_size;
    unsigned ગાંઠ_size_log2;

    uintptr_t મૂળ;
};

શૂન્ય * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
શૂન્ય   util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t ગાંઠ_size);

અંતઃરેખ શૂન્ય os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    જ્યારે (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

સંરચના timer_data {
    long long start;
    long long total;
    long long ગણતરી;
};

સ્થિર અંતઃરેખ શૂન્ય start_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
સ્થિર અંતઃરેખ long long stop_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->ગણતરી += 1;

    પાછા duration;
}
