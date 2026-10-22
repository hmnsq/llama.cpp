#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stddef.h>

પ્રકારવ્યાખ્યા શૂન્ય (*work_queue_func_t)(unsigned પૂર્ણાંક n, unsigned પૂર્ણાંક i, શૂન્ય *);

સંરચના work_queue_s;
પ્રકારવ્યાખ્યા સંરચના work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t ઢગલો_size);
size_t       work_queue_alignof(શૂન્ય);
work_queue_t work_queue_init(શૂન્ય * ptr, uint32_t n_threads, uint32_t capacity, uint32_t ઢગલો_size);
શૂન્ય         work_queue_free(work_queue_t q);

શૂન્ય work_queue_wakeup(work_queue_t q);
શૂન્ય work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, શૂન્ય * data, unsigned પૂર્ણાંક n);

સ્થિર અંતઃરેખ bool work_queue_run(work_queue_t q, work_queue_func_t func, શૂન્ય * data, unsigned પૂર્ણાંક n) {
    જો (n <= 1) {
        func(n, 0, data);
        પાછા true;
    }
    પાછા work_queue_run_async(q, func, data, n);
}

// Legacy compatibility
પ્રકારવ્યાખ્યા work_queue_func_t     worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
