#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <stddef.h>

வகைவரை வெறுமை (*work_queue_func_t)(unsigned முழு n, unsigned முழு i, வெறுமை *);

கட்டமைப்பு work_queue_s;
வகைவரை கட்டமைப்பு work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t அடுக்கு_size);
size_t       work_queue_alignof(வெறுமை);
work_queue_t work_queue_init(வெறுமை * ptr, uint32_t n_threads, uint32_t capacity, uint32_t அடுக்கு_size);
வெறுமை         work_queue_free(work_queue_t q);

வெறுமை work_queue_wakeup(work_queue_t q);
வெறுமை work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, வெறுமை * data, unsigned முழு n);

நிலையான உள்வரி bool work_queue_run(work_queue_t q, work_queue_func_t func, வெறுமை * data, unsigned முழு n) {
    எனில் (n <= 1) {
        func(n, 0, data);
        திரும்பு true;
    }
    திரும்பு work_queue_run_async(q, func, data, n);
}

// Legacy compatibility
வகைவரை work_queue_func_t     worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
