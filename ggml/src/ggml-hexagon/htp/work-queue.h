#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stddef.h>

প্রকারলেখ শূন্য (*work_queue_func_t)(unsigned পূর্ণসংখ্যা n, unsigned পূর্ণসংখ্যা i, শূন্য *);

কাঠামো work_queue_s;
প্রকারলেখ কাঠামো work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t স্তূপ_আকার);
size_t       work_queue_alignof(শূন্য);
work_queue_t work_queue_init(শূন্য * ptr, uint32_t n_threads, uint32_t capacity, uint32_t স্তূপ_আকার);
শূন্য         work_queue_free(work_queue_t q);

শূন্য work_queue_wakeup(work_queue_t q);
শূন্য work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, শূন্য * তথ্য, unsigned পূর্ণসংখ্যা n);

স্থির অন্তঃরেখ bool work_queue_run(work_queue_t q, work_queue_func_t func, শূন্য * তথ্য, unsigned পূর্ণসংখ্যা n) {
    যদি (n <= 1) {
        func(n, 0, তথ্য);
        ফেরত true;
    }
    ফেরত work_queue_run_async(q, func, তথ্য, n);
}

// Legacy compatibility
প্রকারলেখ work_queue_func_t     worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
