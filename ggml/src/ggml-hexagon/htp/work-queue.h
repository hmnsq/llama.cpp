#अगर_अघोषित HTP_WORK_QUEUE_H
#घोषणा HTP_WORK_QUEUE_H

#समावेश <stdbool.h>
#समावेश <stdint.h>
#समावेश <stddef.h>

प्रकार व्योम (*work_queue_func_t)(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम *);

काष्ठा work_queue_s;
प्रकार काष्ठा work_queue_s * work_queue_t;

#घोषणा WORK_QUEUE_MAX_N_THREADS      10

माप_प्रकार       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t ढेर_आकार);
माप_प्रकार       work_queue_alignof(व्योम);
work_queue_t work_queue_init(व्योम * ptr, uint32_t n_threads, uint32_t capacity, uint32_t ढेर_आकार);
व्योम         work_queue_free(work_queue_t q);

व्योम work_queue_wakeup(work_queue_t q);
व्योम work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, व्योम * तथ्य, अचिन्हित पूर्णांक n);

अटल अंतरभूत bool work_queue_run(work_queue_t q, work_queue_func_t func, व्योम * तथ्य, अचिन्हित पूर्णांक n) {
    अगर (n <= 1) {
        func(n, 0, तथ्य);
        वापस true;
    }
    वापस work_queue_run_async(q, func, तथ्य, n);
}

// Legacy compatibility
प्रकार work_queue_func_t     worker_callback_t;
#घोषणा worker_pool_run_func  work_queue_run
#घोषणा worker_pool           work_queue

#पूर्ण_अगर  // #ifndef HTP_WORK_QUEUE_H
