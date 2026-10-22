#ہدایت once

#شامل <atomic>
#شامل <cassert>
#شامل <cerrno>
#شامل <cstdarg>
#شامل <cstddef>
#شامل <cstdint>
#شامل <cstdio>
#شامل <cstdlib>
#شامل <ctime>

#تعریف_کرو unlikely(x) __builtin_expect(!!(x), 0)
#تعریف_کرو likely(x)   __builtin_expect(!!(x), 1)

#اگر_غیر_تعریف_شدہ غیر_مستعمل
#    define غیر_مستعمل(x) (خالی) (x)
#اگر_ختم

/** Checks is a value is a power of two. Does not handle zero. */
#تعریف_کرو IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#تعریف_کرو IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#تعریف_کرو ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#تعریف_کرو p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

ساکن اندرونی bool util_is_power_کا_two_nonzero64(uint64_t v) {
    واپس IS_POT_NONZERO(v);
}

ساکن اندرونی uint64_t align64(uint64_t قدر, uint64_t alignment) {
    أكد(util_is_power_کا_two_nonzero64(alignment));
    واپس ALIGN_POT(قدر, alignment);
}

ساخت list_سر {
    list_سر * پچھلا;
    list_سر * اگلا;
};

ساخت util_sparse_array {
    size_t   elem_سائز;
    غیرعلامتی گرہ_سائز_log2;

    uintptr_t جڑ;
};

خالی * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
خالی   util_sparse_array_init(util_sparse_array * arr, size_t elem_سائز, size_t گرہ_سائز);

اندرونی خالی os_وقت_sleep(int64_t usecs) {
    timespec وقت;
    وقت.tv_sec  = usecs / 1000000;
    وقت.tv_nsec = (usecs % 1000000) * 1000;
    جب (clock_nanosleep(CLOCK_MONOTONIC, 0, &وقت, &وقت) == EINTR)
        ;
}

ساخت timer_ڈیٹا {
    طویل طویل آغاز;
    طویل طویل کل;
    طویل طویل گنتی;
};

ساکن اندرونی خالی آغاز_timer(timer_ڈیٹا * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->آغاز = (طویل طویل) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
ساکن اندرونی طویل طویل stop_timer(timer_ڈیٹا * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    طویل طویل timer_انتہا = (طویل طویل) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    طویل طویل duration = (timer_انتہا - timer->آغاز);
    timer->کل += duration;
    timer->گنتی += 1;

    واپس duration;
}
