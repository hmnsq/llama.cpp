#اگر_غیر_تعریف_شدہ HTP_WORK_قطار_H
#تعریف_کرو HTP_WORK_قطار_H

#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <stddef.h>

قسمتعریف خالی (*work_قطار_func_t)(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی *);

ساخت work_قطار_s;
قسمتعریف ساخت work_قطار_s * work_قطار_t;

#تعریف_کرو WORK_قطار_MAX_N_دھاگے      10

size_t       work_قطار_sizeof(uint32_t n_دھاگے, uint32_t استعداد, uint32_t ڈھیر_سائز);
size_t       work_قطار_alignof(خالی);
work_قطار_t work_قطار_init(خالی * ptr, uint32_t n_دھاگے, uint32_t استعداد, uint32_t ڈھیر_سائز);
خالی         work_قطار_free(work_قطار_t q);

خالی work_قطار_wakeup(work_قطار_t q);
خالی work_قطار_suspend(work_قطار_t q);

bool work_قطار_تسلسل_async(work_قطار_t q, work_قطار_func_t func, خالی * ڈیٹا, غیرعلامتی صحیح n);

ساکن اندرونی bool work_قطار_تسلسل(work_قطار_t q, work_قطار_func_t func, خالی * ڈیٹا, غیرعلامتی صحیح n) {
    اگر (n <= 1) {
        func(n, 0, ڈیٹا);
        واپس true;
    }
    واپس work_قطار_تسلسل_async(q, func, ڈیٹا, n);
}

// Legacy compatibility
قسمتعریف work_قطار_func_t     کارکن_callback_t;
#تعریف_کرو کارکن_pool_تسلسل_func  work_قطار_تسلسل
#تعریف_کرو کارکن_pool           work_قطار

#اگر_ختم  // #ifndef HTP_WORK_QUEUE_H
