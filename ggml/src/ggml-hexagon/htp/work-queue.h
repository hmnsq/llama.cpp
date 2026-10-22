#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#অন্তর্ভাৱ <stdbool.h>
#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <stddef.h>

প্রকারনির্দেশ শূন্য (*work_queue_func_t)(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য *);

সংরচনা work_queue_s;
প্রকারনির্দেশ সংরচনা work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t স্তূপ_size);
size_t       work_queue_alignof(শূন্য);
work_queue_t work_queue_init(শূন্য * ptr, uint32_t n_threads, uint32_t capacity, uint32_t স্তূপ_size);
শূন্য         work_queue_free(work_queue_t q);

শূন্য work_queue_wakeup(work_queue_t q);
শূন্য work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, শূন্য * data, unsigned পূর্ণাঙ্ক n);

স্থির অন্তঃস্থ bool work_queue_run(work_queue_t q, work_queue_func_t func, শূন্য * data, unsigned পূর্ণাঙ্ক n) {
    যদি (n <= 1) {
        func(n, 0, data);
        ঘূৰাই true;
    }
    ঘূৰাই work_queue_run_async(q, func, data, n);
}

// Legacy compatibility
প্রকারনির্দেশ work_queue_func_t     worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
