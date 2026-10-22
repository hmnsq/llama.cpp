#pragma once

#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cstdint>

#define SPINE_CACHE_ಸಾಲು  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_ಸಾಲು)))

ಸಂರಚನಾ spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt       = --b->pending_;
    ಯದಿ (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } ಅನ್ಯಥಾ {
        ಯಾವತ್ (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ ಚಞ್ಚಲ("pause " ::: "memory");
        }
    }
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ spine_barrier_init(spine_barrier_t * b, ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_barriers, uint64_t thread_ಎಣಿಕೆ) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_barriers; i++) {
        b[i].total_ = thread_ಎಣಿಕೆ;
        b[i].pending_.store(thread_ಎಣಿಕೆ);
        b[i].rounds_.store(0);
    }
}
