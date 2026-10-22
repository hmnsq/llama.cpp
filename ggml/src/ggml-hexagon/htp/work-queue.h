#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <stddef.h>

تعریفنوع تهی (*work_queue_func_t)(unsigned صحیح n, unsigned صحیح i, تهی *);

ساختار work_queue_s;
تعریفنوع ساختار work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t پشته_size);
size_t       work_queue_alignof(تهی);
work_queue_t work_queue_init(تهی * ptr, uint32_t n_threads, uint32_t capacity, uint32_t پشته_size);
تهی         work_queue_free(work_queue_t q);

تهی work_queue_wakeup(work_queue_t q);
تهی work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, تهی * data, unsigned صحیح n);

ایستا درونخطی bool work_queue_run(work_queue_t q, work_queue_func_t func, تهی * data, unsigned صحیح n) {
    اگر (n <= 1) {
        func(n, 0, data);
        بازگشت true;
    }
    بازگشت work_queue_run_async(q, func, data, n);
}

// Legacy compatibility
تعریفنوع work_queue_func_t     worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
