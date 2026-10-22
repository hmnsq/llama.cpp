#اگر_تعریف_نشده HTP_WORK_صف_H
#تعریف_کن HTP_WORK_صف_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <stddef.h>

تعريفنوع فراغ (*work_صف_func_t)(بیعلامت صحيح n, بیعلامت صحيح i, فراغ *);

بنية work_صف_s;
تعريفنوع بنية work_صف_s * work_صف_t;

#تعریف_کن WORK_صف_MAX_N_نخها      10

size_t       work_صف_sizeof(uint32_t n_نخها, uint32_t ظرفیت, uint32_t پشته_اندازه);
size_t       work_صف_alignof(فراغ);
work_صف_t work_صف_init(فراغ * ptr, uint32_t n_نخها, uint32_t ظرفیت, uint32_t پشته_اندازه);
فراغ         work_صف_free(work_صف_t q);

فراغ work_صف_wakeup(work_صف_t q);
فراغ work_صف_suspend(work_صف_t q);

bool work_صف_پیاپی_async(work_صف_t q, work_صف_func_t func, فراغ * داده, بیعلامت صحيح n);

ساكن مضمن bool work_صف_پیاپی(work_صف_t q, work_صف_func_t func, فراغ * داده, بیعلامت صحيح n) {
    إذا (n <= 1) {
        func(n, 0, داده);
        أرجع true;
    }
    أرجع work_صف_پیاپی_async(q, func, داده, n);
}

// Legacy compatibility
تعريفنوع work_صف_func_t     کارگر_callback_t;
#تعریف_کن کارگر_pool_پیاپی_func  work_صف_پیاپی
#تعریف_کن کارگر_pool           work_صف

#پایان_اگر  // #ifndef HTP_WORK_QUEUE_H
