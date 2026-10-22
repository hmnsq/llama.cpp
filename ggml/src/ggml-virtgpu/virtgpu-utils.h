#رهنمود once

#ضمن <atomic>
#ضمن <cassert>
#ضمن <cerrno>
#ضمن <cstdarg>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <cstdio>
#ضمن <cstdlib>
#ضمن <ctime>

#تعریف_کن unlikely(x) __builtin_expect(!!(x), 0)
#تعریف_کن likely(x)   __builtin_expect(!!(x), 1)

#اگر_تعریف_نشده استفادهنشده
#    define استفادهنشده(x) (فراغ) (x)
#پایان_اگر

/** Checks is a value is a power of two. Does not handle zero. */
#تعریف_کن IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#تعریف_کن IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#تعریف_کن ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#تعریف_کن p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

ساكن مضمن bool util_is_power_از_two_nonzero64(uint64_t v) {
    أرجع IS_POT_NONZERO(v);
}

ساكن مضمن uint64_t align64(uint64_t مقدار, uint64_t alignment) {
    أكد(util_is_power_از_two_nonzero64(alignment));
    أرجع ALIGN_POT(مقدار, alignment);
}

بنية list_سر {
    list_سر * prev;
    list_سر * بعدی;
};

بنية util_sparse_array {
    size_t   elem_اندازه;
    بیعلامت گره_اندازه_log2;

    uintptr_t ریشه;
};

فراغ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
فراغ   util_sparse_array_init(util_sparse_array * arr, size_t elem_اندازه, size_t گره_اندازه);

مضمن فراغ os_زمان_sleep(int64_t usecs) {
    timespec زمان;
    زمان.tv_sec  = usecs / 1000000;
    زمان.tv_nsec = (usecs % 1000000) * 1000;
    طالما (clock_nanosleep(CLOCK_MONOTONIC, 0, &زمان, &زمان) == EINTR)
        ;
}

بنية timer_داده {
    دراز دراز شروع;
    دراز دراز مجموع;
    دراز دراز شمار;
};

ساكن مضمن فراغ شروع_timer(timer_داده * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->شروع = (دراز دراز) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
ساكن مضمن دراز دراز stop_timer(timer_داده * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    دراز دراز timer_انتها = (دراز دراز) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    دراز دراز duration = (timer_انتها - timer->شروع);
    timer->مجموع += duration;
    timer->شمار += 1;

    أرجع duration;
}
