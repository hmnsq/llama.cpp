#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <cstdarg>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <ctime>

#ವ್ಯಾಖ್ಯಾನಿಸು unlikely(x) __builtin_expect(!!(x), 0)
#ವ್ಯಾಖ್ಯಾನಿಸು likely(x)   __builtin_expect(!!(x), 1)

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ ಬಳಸದ
#    define ಬಳಸದ(x) (ಶೂನ್ಯ) (x)
#ಆದರೆ_ಅಂತ್ಯ

/** Checks is a value is a power of two. Does not handle zero. */
#ವ್ಯಾಖ್ಯಾನಿಸು IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#ವ್ಯಾಖ್ಯಾನಿಸು IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#ವ್ಯಾಖ್ಯಾನಿಸು ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#ವ್ಯಾಖ್ಯಾನಿಸು p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool util_is_power_ದ_two_nonzero64(uint64_t v) {
    ಹಿಂತಿರುಗಿ IS_POT_NONZERO(v);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t align64(uint64_t ಮೌಲ್ಯ, uint64_t alignment) {
    ಪ್ರತಿಜ್ಞಾ(util_is_power_ದ_two_nonzero64(alignment));
    ಹಿಂತಿರುಗಿ ALIGN_POT(ಮೌಲ್ಯ, alignment);
}

ರಚನೆ list_ತಲೆ {
    list_ತಲೆ * prev;
    list_ತಲೆ * ಮುಂದಿನ;
};

ರಚನೆ util_sparse_array {
    size_t   elem_ಗಾತ್ರ;
    ಚಿಹ್ನೆರಹಿತ ಗಂಟು_ಗಾತ್ರ_log2;

    uintptr_t ಬೇರು;
};

ಶೂನ್ಯ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
ಶೂನ್ಯ   util_sparse_array_init(util_sparse_array * arr, size_t elem_ಗಾತ್ರ, size_t ಗಂಟು_ಗಾತ್ರ);

ಸಾಲೊಳಗೆ ಶೂನ್ಯ os_ಸಮಯ_sleep(int64_t usecs) {
    timespec ಸಮಯ;
    ಸಮಯ.tv_sec  = usecs / 1000000;
    ಸಮಯ.tv_nsec = (usecs % 1000000) * 1000;
    ಆಗ (clock_nanosleep(CLOCK_MONOTONIC, 0, &ಸಮಯ, &ಸಮಯ) == EINTR)
        ;
}

ರಚನೆ timer_ದತ್ತಾಂಶ {
    ದೀರ್ಘ ದೀರ್ಘ ಆರಂಭ;
    ದೀರ್ಘ ದೀರ್ಘ ಒಟ್ಟು;
    ದೀರ್ಘ ದೀರ್ಘ ಎಣಿಕೆ_2;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಆರಂಭ_timer(timer_ದತ್ತಾಂಶ * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->ಆರಂಭ = (ದೀರ್ಘ ದೀರ್ಘ) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದೀರ್ಘ ದೀರ್ಘ stop_timer(timer_ದತ್ತಾಂಶ * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ದೀರ್ಘ ದೀರ್ಘ timer_ಕೊನೆ = (ದೀರ್ಘ ದೀರ್ಘ) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    ದೀರ್ಘ ದೀರ್ಘ duration = (timer_ಕೊನೆ - timer->ಆರಂಭ);
    timer->ಒಟ್ಟು += duration;
    timer->ಎಣಿಕೆ_2 += 1;

    ಹಿಂತಿರುಗಿ duration;
}
