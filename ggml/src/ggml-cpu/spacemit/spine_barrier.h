#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cstdint>

#ವ್ಯಾಖ್ಯಾನಿಸು SPINE_CACHE_ಸಾಲು  64
#ವ್ಯಾಖ್ಯಾನಿಸು SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_ಸಾಲು)))

ರಚನೆ spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> ಸುತ್ತುಗಳು;
    SPINE_CACHE_ALIGN int64_t              ಒಟ್ಟು_2;
};

ಸಾಲೊಳಗೆ ಶೂನ್ಯ spine_barrier_wait(spine_barrier_t * b) {
    ಸ್ವಯಂಚಾಲಿತ cur_ಸುತ್ತು = b->ಸುತ್ತುಗಳು.ತುಂಬು(std::memory_order_acquire);
    ಸ್ವಯಂಚಾಲಿತ cnt       = --b->pending_;
    ಆದರೆ (cnt == 0) {
        b->pending_.store(b->ಒಟ್ಟು_2);
        b->ಸುತ್ತುಗಳು.store(cur_ಸುತ್ತು + 1);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆಗ (cur_ಸುತ್ತು == b->ಸುತ್ತುಗಳು.ತುಂಬು(std::memory_order_relaxed)) {
            __asm__ ಅಸ್ಥಿರ("pause " ::: "memory");
        }
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ spine_barrier_init(spine_barrier_t * b, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_barriers, uint64_t thread_ಎಣಿಕೆ) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_barriers; i++) {
        b[i].ಒಟ್ಟು_2 = thread_ಎಣಿಕೆ;
        b[i].pending_.store(thread_ಎಣಿಕೆ);
        b[i].ಸುತ್ತುಗಳು.store(0);
    }
}
