#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*work_queue_func_t)(unsigned ಪೂರ್ಣಾಙ್ಕ n, unsigned ಪೂರ್ಣಾಙ್ಕ i, ಶೂನ್ಯ *);

ಸಂರಚನಾ work_queue_s;
ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t ರಾಶಿ_size);
size_t       work_queue_alignof(ಶೂನ್ಯ);
work_queue_t work_queue_init(ಶೂನ್ಯ * ptr, uint32_t n_threads, uint32_t capacity, uint32_t ರಾಶಿ_size);
ಶೂನ್ಯ         work_queue_free(work_queue_t q);

ಶೂನ್ಯ work_queue_wakeup(work_queue_t q);
ಶೂನ್ಯ work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, ಶೂನ್ಯ * data, unsigned ಪೂರ್ಣಾಙ್ಕ n);

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool work_queue_run(work_queue_t q, work_queue_func_t func, ಶೂನ್ಯ * data, unsigned ಪೂರ್ಣಾಙ್ಕ n) {
    ಯದಿ (n <= 1) {
        func(n, 0, data);
        ಪ್ರತ್ಯಾಗಮ true;
    }
    ಪ್ರತ್ಯಾಗಮ work_queue_run_async(q, func, data, n);
}

// Legacy compatibility
ಪ್ರಕಾರನಿರ್ದೇಶ work_queue_func_t     worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
