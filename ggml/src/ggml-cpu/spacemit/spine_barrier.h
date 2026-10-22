#pragma once

#અન્તર્ભાવ <atomic>
#અન્તર્ભાવ <cstdint>

#define SPINE_CACHE_લીટી  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_લીટી)))

સંરચના spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

અંતઃરેખ શૂન્ય spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt       = --b->pending_;
    જો (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } નહીંતર {
        જ્યારે (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ અસ્થિર("pause " ::: "memory");
        }
    }
}

અંતઃરેખ શૂન્ય spine_barrier_init(spine_barrier_t * b, પૂર્ણાંક અંક_barriers, uint64_t thread_ગણતરી) {
    માટે (પૂર્ણાંક i = 0; i < અંક_barriers; i++) {
        b[i].total_ = thread_ગણતરી;
        b[i].pending_.store(thread_ગણતરી);
        b[i].rounds_.store(0);
    }
}
