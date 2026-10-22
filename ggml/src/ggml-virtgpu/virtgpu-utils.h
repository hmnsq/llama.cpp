#لارښوونه once

#ضمن <atomic>
#ضمن <cassert>
#ضمن <cerrno>
#ضمن <cstdarg>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <cstdio>
#ضمن <cstdlib>
#ضمن <ctime>

#تعریف_کړه unlikely(x) __builtin_expect(!!(x), 0)
#تعریف_کړه likely(x)   __builtin_expect(!!(x), 1)

#که_نه_تعریف_شوی ناکارول_شوی
#    define ناکارول_شوی(x) (فراغ) (x)
#د_که_پای

/** Checks is a value is a power of two. Does not handle zero. */
#تعریف_کړه IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#تعریف_کړه IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#تعریف_کړه ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#تعریف_کړه p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

ساكن مضمن bool util_is_power_د_two_nonzero64(uint64_t v) {
    أرجع IS_POT_NONZERO(v);
}

ساكن مضمن uint64_t align64(uint64_t ارزښت, uint64_t alignment) {
    أكد(util_is_power_د_two_nonzero64(alignment));
    أرجع ALIGN_POT(ارزښت, alignment);
}

بنية list_سر {
    list_سر * prev;
    list_سر * بل;
};

بنية util_sparse_array {
    size_t   elem_اندازه;
    بینښه غوټه_اندازه_log2;

    uintptr_t ریښه;
};

فراغ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
فراغ   util_sparse_array_init(util_sparse_array * arr, size_t elem_اندازه, size_t غوټه_اندازه);

مضمن فراغ os_وخت_sleep(int64_t usecs) {
    timespec وخت;
    وخت.tv_sec  = usecs / 1000000;
    وخت.tv_nsec = (usecs % 1000000) * 1000;
    طالما (clock_nanosleep(CLOCK_MONOTONIC, 0, &وخت, &وخت) == EINTR)
        ;
}

بنية timer_معلومات {
    اوږد اوږد پیل;
    اوږد اوږد ټول;
    اوږد اوږد شمېر;
};

ساكن مضمن فراغ پیل_timer(timer_معلومات * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->پیل = (اوږد اوږد) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
ساكن مضمن اوږد اوږد stop_timer(timer_معلومات * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    اوږد اوږد timer_پای_ټکی = (اوږد اوږد) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    اوږد اوږد duration = (timer_پای_ټکی - timer->پیل);
    timer->ټول += duration;
    timer->شمېر += 1;

    أرجع duration;
}
