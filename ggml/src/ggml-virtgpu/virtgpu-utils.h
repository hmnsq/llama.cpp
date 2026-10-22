#pragma once

#अन्तर्भाव <atomic>
#अन्तर्भाव <cassert>
#अन्तर्भाव <cerrno>
#अन्तर्भाव <cstdarg>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <ctime>

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x)   __builtin_expect(!!(x), 1)

#ifndef UNUSED
#    define UNUSED(x) (शून्य) (x)
#endif

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

स्थिर अन्तःस्थ bool util_is_power_of_two_nonzero64(uint64_t v) {
    प्रत्यागम IS_POT_NONZERO(v);
}

स्थिर अन्तःस्थ uint64_t align64(uint64_t मूल्य, uint64_t alignment) {
    प्रतिज्ञा(util_is_power_of_two_nonzero64(alignment));
    प्रत्यागम ALIGN_POT(मूल्य, alignment);
}

संरचना list_head {
    list_head * prev;
    list_head * next;
};

संरचना util_sparse_array {
    size_t   elem_size;
    unsigned गाँठ_size_log2;

    uintptr_t मूल;
};

शून्य * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
शून्य   util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t गाँठ_size);

अन्तःस्थ शून्य os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    यावत् (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

संरचना timer_data {
    long long start;
    long long total;
    long long गिनती;
};

स्थिर अन्तःस्थ शून्य start_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
स्थिर अन्तःस्थ long long stop_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->गिनती += 1;

    प्रत्यागम duration;
}
