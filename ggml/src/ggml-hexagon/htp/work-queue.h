#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <stddef.h>

തരനിർവചനം ശൂന്യം (*work_queue_func_t)(unsigned പൂർണ്ണസംഖ്യ n, unsigned പൂർണ്ണസംഖ്യ i, ശൂന്യം *);

ഘടന work_queue_s;
തരനിർവചനം ഘടന work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t അട്ടി_size);
size_t       work_queue_alignof(ശൂന്യം);
work_queue_t work_queue_init(ശൂന്യം * ptr, uint32_t n_threads, uint32_t capacity, uint32_t അട്ടി_size);
ശൂന്യം         work_queue_free(work_queue_t q);

ശൂന്യം work_queue_wakeup(work_queue_t q);
ശൂന്യം work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, ശൂന്യം * data, unsigned പൂർണ്ണസംഖ്യ n);

സ്ഥിതം വരിയിൽ bool work_queue_run(work_queue_t q, work_queue_func_t func, ശൂന്യം * data, unsigned പൂർണ്ണസംഖ്യ n) {
    എങ്കിൽ (n <= 1) {
        func(n, 0, data);
        തിരികെ true;
    }
    തിരികെ work_queue_run_async(q, func, data, n);
}

// Legacy compatibility
തരനിർവചനം work_queue_func_t     worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
