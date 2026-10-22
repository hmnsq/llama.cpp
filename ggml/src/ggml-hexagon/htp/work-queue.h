#यदि_अपरिभाषितम् HTP_WORK_QUEUE_H
#परिभाषय HTP_WORK_QUEUE_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>

प्रकारनिर्देश शून्य (*work_queue_func_t)(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य *);

संरचना work_queue_s;
प्रकारनिर्देश संरचना work_queue_s * work_queue_t;

#परिभाषय WORK_QUEUE_MAX_N_तन्तवः      10

size_t       work_queue_sizeof(uint32_t n_तन्तवः, uint32_t क्षमता, uint32_t राशिः_आकार);
size_t       work_queue_alignof(शून्य);
work_queue_t work_queue_init(शून्य * ptr, uint32_t n_तन्तवः, uint32_t क्षमता, uint32_t राशिः_आकार);
शून्य         work_queue_free(work_queue_t q);

शून्य work_queue_wakeup(work_queue_t q);
शून्य work_queue_suspend(work_queue_t q);

bool work_queue_सततम्_async(work_queue_t q, work_queue_func_t func, शून्य * दत्त, अचिह्नित पूर्णाङ्क n);

स्थिर अन्तःस्थ bool work_queue_सततम्(work_queue_t q, work_queue_func_t func, शून्य * दत्त, अचिह्नित पूर्णाङ्क n) {
    यदि (n <= 1) {
        func(n, 0, दत्त);
        प्रत्यागम true;
    }
    प्रत्यागम work_queue_सततम्_async(q, func, दत्त, n);
}

// Legacy compatibility
प्रकारनिर्देश work_queue_func_t     कार्यकर्ता_callback_t;
#परिभाषय कार्यकर्ता_pool_सततम्_func  work_queue_सततम्
#परिभाषय कार्यकर्ता_pool           work_queue

#यदि_अन्तः  // #ifndef HTP_WORK_QUEUE_H
