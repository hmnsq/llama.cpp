#جيڪڏهن_وصف_نه_ٿيل HTP_WORK_قطار_H
#وصف_ڪريو HTP_WORK_قطار_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <stddef.h>

تعريفنوع فراغ (*work_قطار_func_t)(بنانشان صحيح n, بنانشان صحيح i, فراغ *);

بنية work_قطار_s;
تعريفنوع بنية work_قطار_s * work_قطار_t;

#وصف_ڪريو WORK_قطار_MAX_N_ڌاڳا      10

size_t       work_قطار_sizeof(uint32_t n_ڌاڳا, uint32_t صلاحيت, uint32_t ڍير_سائيز);
size_t       work_قطار_alignof(فراغ);
work_قطار_t work_قطار_init(فراغ * ptr, uint32_t n_ڌاڳا, uint32_t صلاحيت, uint32_t ڍير_سائيز);
فراغ         work_قطار_free(work_قطار_t q);

فراغ work_قطار_wakeup(work_قطار_t q);
فراغ work_قطار_suspend(work_قطار_t q);

bool work_قطار_لاڳيتو_async(work_قطار_t q, work_قطار_func_t func, فراغ * معلومات, بنانشان صحيح n);

ساكن مضمن bool work_قطار_لاڳيتو(work_قطار_t q, work_قطار_func_t func, فراغ * معلومات, بنانشان صحيح n) {
    إذا (n <= 1) {
        func(n, 0, معلومات);
        أرجع true;
    }
    أرجع work_قطار_لاڳيتو_async(q, func, معلومات, n);
}

// Legacy compatibility
تعريفنوع work_قطار_func_t     ڪم_ڪندڙ_callback_t;
#وصف_ڪريو ڪم_ڪندڙ_pool_لاڳيتو_func  work_قطار_لاڳيتو
#وصف_ڪريو ڪم_ڪندڙ_pool           work_قطار

#جيڪڏهن_پورو  // #ifndef HTP_WORK_QUEUE_H
