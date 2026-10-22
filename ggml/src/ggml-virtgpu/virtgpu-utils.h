#هدايت once

#ضمن <atomic>
#ضمن <cassert>
#ضمن <cerrno>
#ضمن <cstdarg>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <cstdio>
#ضمن <cstdlib>
#ضمن <ctime>

#وصف_ڪريو unlikely(x) __builtin_expect(!!(x), 0)
#وصف_ڪريو likely(x)   __builtin_expect(!!(x), 1)

#جيڪڏهن_وصف_نه_ٿيل اڻ_استعمال_ٿيل
#    define اڻ_استعمال_ٿيل(x) (فراغ) (x)
#جيڪڏهن_پورو

/** Checks is a value is a power of two. Does not handle zero. */
#وصف_ڪريو IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#وصف_ڪريو IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#وصف_ڪريو ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#وصف_ڪريو p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

ساكن مضمن bool util_is_power_جو_two_nonzero64(uint64_t v) {
    أرجع IS_POT_NONZERO(v);
}

ساكن مضمن uint64_t align64(uint64_t قدر, uint64_t alignment) {
    أكد(util_is_power_جو_two_nonzero64(alignment));
    أرجع ALIGN_POT(قدر, alignment);
}

بنية list_سر {
    list_سر * prev;
    list_سر * ايندڙ;
};

بنية util_sparse_array {
    size_t   elem_سائيز;
    بنانشان ڳنڍ_سائيز_log2;

    uintptr_t پاڙ;
};

فراغ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
فراغ   util_sparse_array_init(util_sparse_array * arr, size_t elem_سائيز, size_t ڳنڍ_سائيز);

مضمن فراغ os_وقت_sleep(int64_t usecs) {
    timespec وقت;
    وقت.tv_sec  = usecs / 1000000;
    وقت.tv_nsec = (usecs % 1000000) * 1000;
    طالما (clock_nanosleep(CLOCK_MONOTONIC, 0, &وقت, &وقت) == EINTR)
        ;
}

بنية timer_معلومات {
    ڊگهو ڊگهو شروعات;
    ڊگهو ڊگهو ڪل;
    ڊگهو ڊگهو ڳڻپ;
};

ساكن مضمن فراغ شروعات_timer(timer_معلومات * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->شروعات = (ڊگهو ڊگهو) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
ساكن مضمن ڊگهو ڊگهو stop_timer(timer_معلومات * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ڊگهو ڊگهو timer_پڇاڙي = (ڊگهو ڊگهو) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    ڊگهو ڊگهو duration = (timer_پڇاڙي - timer->شروعات);
    timer->ڪل += duration;
    timer->ڳڻپ += 1;

    أرجع duration;
}
