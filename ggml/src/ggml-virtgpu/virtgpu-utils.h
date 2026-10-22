#निर्देशः once

#अन्तर्भाव <atomic>
#अन्तर्भाव <cassert>
#अन्तर्भाव <cerrno>
#अन्तर्भाव <cstdarg>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <ctime>

#परिभाषय unlikely(x) __builtin_expect(!!(x), 0)
#परिभाषय likely(x)   __builtin_expect(!!(x), 1)

#यदि_अपरिभाषितम् अप्रयुक्तम्
#    define अप्रयुक्तम्(x) (शून्य) (x)
#यदि_अन्तः

/** Checks is a value is a power of two. Does not handle zero. */
#परिभाषय IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#परिभाषय IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#परिभाषय ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#परिभाषय p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

स्थिर अन्तःस्थ bool util_is_power_सम्बन्धि_two_nonzero64(uint64_t v) {
    प्रत्यागम IS_POT_NONZERO(v);
}

स्थिर अन्तःस्थ uint64_t align64(uint64_t मानम्, uint64_t alignment) {
    प्रतिज्ञा(util_is_power_सम्बन्धि_two_nonzero64(alignment));
    प्रत्यागम ALIGN_POT(मानम्, alignment);
}

संरचना list_शीर्ष {
    list_शीर्ष * पूर्व;
    list_शीर्ष * अग्र;
};

संरचना util_sparse_array {
    size_t   elem_आकार;
    अचिह्नित ग्रन्थिः_आकार_log2;

    uintptr_t मूलम्;
};

शून्य * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
शून्य   util_sparse_array_init(util_sparse_array * arr, size_t elem_आकार, size_t ग्रन्थिः_आकार);

अन्तःस्थ शून्य os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    यावत् (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

संरचना timer_दत्त {
    दीर्घ दीर्घ start;
    दीर्घ दीर्घ योग;
    दीर्घ दीर्घ गणनम्;
};

स्थिर अन्तःस्थ शून्य start_timer(timer_दत्त * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (दीर्घ दीर्घ) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
स्थिर अन्तःस्थ दीर्घ दीर्घ stop_timer(timer_दत्त * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    दीर्घ दीर्घ timer_अन्तः = (दीर्घ दीर्घ) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    दीर्घ दीर्घ duration = (timer_अन्तः - timer->start);
    timer->योग += duration;
    timer->गणनम् += 1;

    प्रत्यागम duration;
}
