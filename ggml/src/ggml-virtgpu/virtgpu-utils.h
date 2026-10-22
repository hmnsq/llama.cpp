#సూచన once

#అన్తర్భావ <atomic>
#అన్తర్భావ <cassert>
#అన్తర్భావ <cerrno>
#అన్తర్భావ <cstdarg>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <ctime>

#నిర్వచించు unlikely(x) __builtin_expect(!!(x), 0)
#నిర్వచించు likely(x)   __builtin_expect(!!(x), 1)

#నిర్వచితం_కాకపోతే వాడని
#    define వాడని(x) (శూన్య) (x)
#అయితే_ముగింపు

/** Checks is a value is a power of two. Does not handle zero. */
#నిర్వచించు IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#నిర్వచించు IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#నిర్వచించు ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#నిర్వచించు p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

స్థిర అంతఃవరుస bool util_is_power_యొక్క_two_nonzero64(uint64_t v) {
    తిరిగి IS_POT_NONZERO(v);
}

స్థిర అంతఃవరుస uint64_t align64(uint64_t విలువ, uint64_t alignment) {
    ప్రతిజ్ఞా(util_is_power_యొక్క_two_nonzero64(alignment));
    తిరిగి ALIGN_POT(విలువ, alignment);
}

నిర్మాణం list_తల {
    list_తల * prev;
    list_తల * తదుపరి;
};

నిర్మాణం util_sparse_array {
    size_t   elem_కొలత;
    గుర్తులేని కణుపు_కొలత_log2;

    uintptr_t మూలం;
};

శూన్య * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
శూన్య   util_sparse_array_init(util_sparse_array * arr, size_t elem_కొలత, size_t కణుపు_కొలత);

అంతఃవరుస శూన్య os_సమయం_sleep(int64_t usecs) {
    timespec సమయం;
    సమయం.tv_sec  = usecs / 1000000;
    సమయం.tv_nsec = (usecs % 1000000) * 1000;
    అయితే (clock_nanosleep(CLOCK_MONOTONIC, 0, &సమయం, &సమయం) == EINTR)
        ;
}

నిర్మాణం timer_దత్తాంశం {
    దీర్ఘ దీర్ఘ ప్రారంభం;
    దీర్ఘ దీర్ఘ మొత్తం;
    దీర్ఘ దీర్ఘ లెక్క;
};

స్థిర అంతఃవరుస శూన్య ప్రారంభం_timer(timer_దత్తాంశం * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->ప్రారంభం = (దీర్ఘ దీర్ఘ) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
స్థిర అంతఃవరుస దీర్ఘ దీర్ఘ stop_timer(timer_దత్తాంశం * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    దీర్ఘ దీర్ఘ timer_చివర = (దీర్ఘ దీర్ఘ) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    దీర్ఘ దీర్ఘ duration = (timer_చివర - timer->ప్రారంభం);
    timer->మొత్తం += duration;
    timer->లెక్క += 1;

    తిరిగి duration;
}
