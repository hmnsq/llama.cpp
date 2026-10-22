#pragma once

#ଅନ୍ତର୍ଭାଵ <atomic>
#ଅନ୍ତର୍ଭାଵ <cassert>
#ଅନ୍ତର୍ଭାଵ <cerrno>
#ଅନ୍ତର୍ଭାଵ <cstdarg>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cstdio>
#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <ctime>

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x)   __builtin_expect(!!(x), 1)

#ifndef UNUSED
#    define UNUSED(x) (ଶୂନ୍ୟ) (x)
#endif

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool util_is_power_of_two_nonzero64(uint64_t v) {
    ଫେରନ୍ତୁ IS_POT_NONZERO(v);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint64_t align64(uint64_t ମୂଲ୍ୟ, uint64_t alignment) {
    ପ୍ରତିଜ୍ଞା(util_is_power_of_two_nonzero64(alignment));
    ଫେରନ୍ତୁ ALIGN_POT(ମୂଲ୍ୟ, alignment);
}

ସଂରଚନା list_head {
    list_head * prev;
    list_head * next;
};

ସଂରଚନା util_sparse_array {
    size_t   elem_size;
    unsigned ଗଣ୍ଠି_size_log2;

    uintptr_t ମୂଳ;
};

ଶୂନ୍ୟ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
ଶୂନ୍ୟ   util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t ଗଣ୍ଠି_size);

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    ଯେତେବେଳେ (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

ସଂରଚନା timer_data {
    long long start;
    long long total;
    long long ଗଣନା_2;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ start_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
ସ୍ଥିର ଧାଡ଼ିଭିତରେ long long stop_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->ଗଣନା_2 += 1;

    ଫେରନ୍ତୁ duration;
}
