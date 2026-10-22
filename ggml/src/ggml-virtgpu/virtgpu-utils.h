#ہدایت once

#ضمن <atomic>
#ضمن <cassert>
#ضمن <cerrno>
#ضمن <cstdarg>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <cstdio>
#ضمن <cstdlib>
#ضمن <ctime>

#تعریف_کرو unlikely(x) __builtin_expect(!!(x), 0)
#تعریف_کرو likely(x)   __builtin_expect(!!(x), 1)

#جے_بنا_تعریف بن_ورتیا
#    define بن_ورتیا(x) (فراغ) (x)
#جے_انت

/** Checks is a value is a power of two. Does not handle zero. */
#تعریف_کرو IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#تعریف_کرو IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#تعریف_کرو ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#تعریف_کرو p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

ساكن مضمن bool util_is_power_دا_two_nonzero64(uint64_t v) {
    أرجع IS_POT_NONZERO(v);
}

ساكن مضمن uint64_t align64(uint64_t قدر, uint64_t alignment) {
    أكد(util_is_power_دا_two_nonzero64(alignment));
    أرجع ALIGN_POT(قدر, alignment);
}

بنية list_سر {
    list_سر * prev;
    list_سر * اگلا;
};

بنية util_sparse_array {
    size_t   elem_ناپ;
    بنانشانی گرہ_ناپ_log2;

    uintptr_t جڑ;
};

فراغ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
فراغ   util_sparse_array_init(util_sparse_array * arr, size_t elem_ناپ, size_t گرہ_ناپ);

مضمن فراغ os_ویلا_sleep(int64_t usecs) {
    timespec ویلا;
    ویلا.tv_sec  = usecs / 1000000;
    ویلا.tv_nsec = (usecs % 1000000) * 1000;
    طالما (clock_nanosleep(CLOCK_MONOTONIC, 0, &ویلا, &ویلا) == EINTR)
        ;
}

بنية timer_انکڑے {
    لما لما شروع;
    لما لما کل;
    لما لما گنتی;
};

ساكن مضمن فراغ شروع_timer(timer_انکڑے * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->شروع = (لما لما) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
ساكن مضمن لما لما stop_timer(timer_انکڑے * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    لما لما timer_سرا = (لما لما) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    لما لما duration = (timer_سرا - timer->شروع);
    timer->کل += duration;
    timer->گنتی += 1;

    أرجع duration;
}
