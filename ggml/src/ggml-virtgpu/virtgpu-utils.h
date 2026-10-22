#आशय once

#समावेश <atomic>
#समावेश <cassert>
#समावेश <cerrno>
#समावेश <cstdarg>
#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <cstdio>
#समावेश <cstdlib>
#समावेश <ctime>

#घोषणा unlikely(x) __builtin_expect(!!(x), 0)
#घोषणा likely(x)   __builtin_expect(!!(x), 1)

#अगर_अघोषित UNUSED
#    define UNUSED(x) (व्योम) (x)
#पूर्ण_अगर

/** Checks is a value is a power of two. Does not handle zero. */
#घोषणा IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#घोषणा IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#घोषणा ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#घोषणा p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

अटल अंतरभूत bool util_is_power_of_two_nonzero64(uint64_t v) {
    वापस IS_POT_NONZERO(v);
}

अटल अंतरभूत uint64_t align64(uint64_t मूल्य, uint64_t alignment) {
    निश्चित(util_is_power_of_two_nonzero64(alignment));
    वापस ALIGN_POT(मूल्य, alignment);
}

काष्ठा list_शीर्ष {
    list_शीर्ष * पूर्व;
    list_शीर्ष * अग्र;
};

काष्ठा util_sparse_array {
    माप_प्रकार   elem_आकार;
    अचिन्हित गाँठ_आकार_log2;

    uintptr_t मूल;
};

व्योम * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);
व्योम   util_sparse_array_init(util_sparse_array * arr, माप_प्रकार elem_आकार, माप_प्रकार गाँठ_आकार);

अंतरभूत व्योम os_time_sleep(int64_t usecs) {
    timespec समय;
    समय.tv_sec  = usecs / 1000000;
    समय.tv_nsec = (usecs % 1000000) * 1000;
    जबतक (clock_nanosleep(CLOCK_MONOTONIC, 0, &समय, &समय) == EINTR)
        ;
}

काष्ठा timer_तथ्य {
    दीर्घ दीर्घ start;
    दीर्घ दीर्घ योग;
    दीर्घ दीर्घ गिनती;
};

अटल अंतरभूत व्योम start_timer(timer_तथ्य * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (दीर्घ दीर्घ) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
अटल अंतरभूत दीर्घ दीर्घ stop_timer(timer_तथ्य * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    दीर्घ दीर्घ timer_end = (दीर्घ दीर्घ) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    दीर्घ दीर्घ duration = (timer_end - timer->start);
    timer->योग += duration;
    timer->गिनती += 1;

    वापस duration;
}
