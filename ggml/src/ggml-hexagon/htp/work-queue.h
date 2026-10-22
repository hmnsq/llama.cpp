#جے_بنا_تعریف HTP_WORK_قطار_H
#تعریف_کرو HTP_WORK_قطار_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <stddef.h>

تعريفنوع فراغ (*work_قطار_func_t)(بنانشانی صحيح n, بنانشانی صحيح i, فراغ *);

بنية work_قطار_s;
تعريفنوع بنية work_قطار_s * work_قطار_t;

#تعریف_کرو WORK_قطار_MAX_N_دھاگے      10

size_t       work_قطار_sizeof(uint32_t n_دھاگے, uint32_t سمرتھا, uint32_t ڈھیر_ناپ);
size_t       work_قطار_alignof(فراغ);
work_قطار_t work_قطار_init(فراغ * ptr, uint32_t n_دھاگے, uint32_t سمرتھا, uint32_t ڈھیر_ناپ);
فراغ         work_قطار_free(work_قطار_t q);

فراغ work_قطار_wakeup(work_قطار_t q);
فراغ work_قطار_suspend(work_قطار_t q);

bool work_قطار_لگاتار_async(work_قطار_t q, work_قطار_func_t func, فراغ * انکڑے, بنانشانی صحيح n);

ساكن مضمن bool work_قطار_لگاتار(work_قطار_t q, work_قطار_func_t func, فراغ * انکڑے, بنانشانی صحيح n) {
    إذا (n <= 1) {
        func(n, 0, انکڑے);
        أرجع true;
    }
    أرجع work_قطار_لگاتار_async(q, func, انکڑے, n);
}

// Legacy compatibility
تعريفنوع work_قطار_func_t     کامہ_callback_t;
#تعریف_کرو کامہ_pool_لگاتار_func  work_قطار_لگاتار
#تعریف_کرو کامہ_pool           work_قطار

#جے_انت  // #ifndef HTP_WORK_QUEUE_H
