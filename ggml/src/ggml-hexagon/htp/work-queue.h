#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stddef.h>

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ (*work_queue_func_t)(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ *);

ସଂରଚନା work_queue_s;
ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t ଗଦା_size);
size_t       work_queue_alignof(ଶୂନ୍ୟ);
work_queue_t work_queue_init(ଶୂନ୍ୟ * ptr, uint32_t n_threads, uint32_t capacity, uint32_t ଗଦା_size);
ଶୂନ୍ୟ         work_queue_free(work_queue_t q);

ଶୂନ୍ୟ work_queue_wakeup(work_queue_t q);
ଶୂନ୍ୟ work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, ଶୂନ୍ୟ * data, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n);

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool work_queue_run(work_queue_t q, work_queue_func_t func, ଶୂନ୍ୟ * data, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n) {
    ଯଦି (n <= 1) {
        func(n, 0, data);
        ଫେରନ୍ତୁ true;
    }
    ଫେରନ୍ତୁ work_queue_run_async(q, func, data, n);
}

// Legacy compatibility
ପ୍ରକାରବ୍ୟାଖ୍ୟା work_queue_func_t     worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
