#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cerrno>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdarg>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <ctime>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ unlikely(x) __builtin_expect(!!(x), 0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ likely(x)   __builtin_expect(!!(x), 1)

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ ਅਣਵਰਤਿਆ
#    define ਅਣਵਰਤਿਆ(x) (ਖਾਲੀ) (x)
#ਜੇ_ਅੰਤ

/** Checks is a value is a power of two. Does not handle zero. */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

ਸਥਿਰ ਸਤਰਅੰਦਰ bool util_is_power_ਦਾ_two_nonzero64(uint64_t v) {
    ਵਾਪਸ IS_POT_NONZERO(v);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t align64(uint64_t ਮੁੱਲ, uint64_t alignment) {
    ਪ੍ਰਤਿਜ੍ਞਾ(util_is_power_ਦਾ_two_nonzero64(alignment));
    ਵਾਪਸ ALIGN_POT(ਮੁੱਲ, alignment);
}

ਬਣਤਰ list_ਸਿਰ {
    list_ਸਿਰ * prev;
    list_ਸਿਰ * ਅਗਲਾ;
};

ਬਣਤਰ util_sparse_array {
    size_t   elem_ਮਾਪ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਗੰਢ_ਮਾਪ_log2;

    uintptr_t ਜੜ੍ਹ;
};

ਖਾਲੀ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
ਖਾਲੀ   util_sparse_array_init(util_sparse_array * arr, size_t elem_ਮਾਪ, size_t ਗੰਢ_ਮਾਪ);

ਸਤਰਅੰਦਰ ਖਾਲੀ os_ਸਮਾਂ_sleep(int64_t usecs) {
    timespec ਸਮਾਂ;
    ਸਮਾਂ.tv_sec  = usecs / 1000000;
    ਸਮਾਂ.tv_nsec = (usecs % 1000000) * 1000;
    ਜਦੋਂ (clock_nanosleep(CLOCK_MONOTONIC, 0, &ਸਮਾਂ, &ਸਮਾਂ) == EINTR)
        ;
}

ਬਣਤਰ timer_ਅੰਕੜੇ {
    ਲੰਮਾ ਲੰਮਾ ਸ਼ੁਰੂ;
    ਲੰਮਾ ਲੰਮਾ ਕੁੱਲ;
    ਲੰਮਾ ਲੰਮਾ ਗਿਣਤੀ_2;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਸ਼ੁਰੂ_timer(timer_ਅੰਕੜੇ * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->ਸ਼ੁਰੂ = (ਲੰਮਾ ਲੰਮਾ) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
ਸਥਿਰ ਸਤਰਅੰਦਰ ਲੰਮਾ ਲੰਮਾ stop_timer(timer_ਅੰਕੜੇ * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ਲੰਮਾ ਲੰਮਾ timer_ਸਿਰਾ = (ਲੰਮਾ ਲੰਮਾ) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    ਲੰਮਾ ਲੰਮਾ duration = (timer_ਸਿਰਾ - timer->ਸ਼ੁਰੂ);
    timer->ਕੁੱਲ += duration;
    timer->ਗਿਣਤੀ_2 += 1;

    ਵਾਪਸ duration;
}
