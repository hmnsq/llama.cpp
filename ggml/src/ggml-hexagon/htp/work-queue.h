#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>

प्रकारव्याख्या शून्य (*work_queue_func_t)(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य *);

संरचना work_queue_s;
प्रकारव्याख्या संरचना work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10

माप_प्रकार       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t चळत_size);
माप_प्रकार       work_queue_alignof(शून्य);
work_queue_t work_queue_init(शून्य * ptr, uint32_t n_threads, uint32_t capacity, uint32_t चळत_size);
शून्य         work_queue_free(work_queue_t q);

शून्य work_queue_wakeup(work_queue_t q);
शून्य work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, शून्य * data, अचिन्हित पूर्णांक n);

स्थिर ओळीत bool work_queue_run(work_queue_t q, work_queue_func_t func, शून्य * data, अचिन्हित पूर्णांक n) {
    जर (n <= 1) {
        func(n, 0, data);
        परत true;
    }
    परत work_queue_run_async(q, func, data, n);
}

// Legacy compatibility
प्रकारव्याख्या work_queue_func_t     worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
