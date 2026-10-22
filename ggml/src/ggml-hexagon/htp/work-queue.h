#که_نه_تعریف_شوی HTP_WORK_کتار_H
#تعریف_کړه HTP_WORK_کتار_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <stddef.h>

تعريفنوع فراغ (*work_کتار_func_t)(بینښه صحيح n, بینښه صحيح i, فراغ *);

بنية work_کتار_s;
تعريفنوع بنية work_کتار_s * work_کتار_t;

#تعریف_کړه WORK_کتار_MAX_N_تارونه      10

size_t       work_کتار_sizeof(uint32_t n_تارونه, uint32_t ظرفیت, uint32_t ډېری_اندازه);
size_t       work_کتار_alignof(فراغ);
work_کتار_t work_کتار_init(فراغ * ptr, uint32_t n_تارونه, uint32_t ظرفیت, uint32_t ډېری_اندازه);
فراغ         work_کتار_free(work_کتار_t q);

فراغ work_کتار_wakeup(work_کتار_t q);
فراغ work_کتار_suspend(work_کتار_t q);

bool work_کتار_پرلپسې_async(work_کتار_t q, work_کتار_func_t func, فراغ * معلومات, بینښه صحيح n);

ساكن مضمن bool work_کتار_پرلپسې(work_کتار_t q, work_کتار_func_t func, فراغ * معلومات, بینښه صحيح n) {
    إذا (n <= 1) {
        func(n, 0, معلومات);
        أرجع true;
    }
    أرجع work_کتار_پرلپسې_async(q, func, معلومات, n);
}

// Legacy compatibility
تعريفنوع work_کتار_func_t     کارګر_callback_t;
#تعریف_کړه کارګر_pool_پرلپسې_func  work_کتار_پرلپسې
#تعریف_کړه کارګر_pool           work_کتار

#د_که_پای  // #ifndef HTP_WORK_QUEUE_H
