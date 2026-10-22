#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_WORK_ਕਤਾਰ_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_WORK_ਕਤਾਰ_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*work_ਕਤਾਰ_func_t)(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ *);

ਬਣਤਰ work_ਕਤਾਰ_s;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ work_ਕਤਾਰ_s * work_ਕਤਾਰ_t;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WORK_ਕਤਾਰ_MAX_N_ਧਾਗੇ      10

size_t       work_ਕਤਾਰ_sizeof(uint32_t n_ਧਾਗੇ, uint32_t ਸਮਰੱਥਾ, uint32_t ਢੇਰ_ਮਾਪ);
size_t       work_ਕਤਾਰ_alignof(ਖਾਲੀ);
work_ਕਤਾਰ_t work_ਕਤਾਰ_init(ਖਾਲੀ * ptr, uint32_t n_ਧਾਗੇ, uint32_t ਸਮਰੱਥਾ, uint32_t ਢੇਰ_ਮਾਪ);
ਖਾਲੀ         work_ਕਤਾਰ_free(work_ਕਤਾਰ_t q);

ਖਾਲੀ work_ਕਤਾਰ_wakeup(work_ਕਤਾਰ_t q);
ਖਾਲੀ work_ਕਤਾਰ_suspend(work_ਕਤਾਰ_t q);

bool work_ਕਤਾਰ_ਲੜੀਵਾਰ_async(work_ਕਤਾਰ_t q, work_ਕਤਾਰ_func_t func, ਖਾਲੀ * ਅੰਕੜੇ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n);

ਸਥਿਰ ਸਤਰਅੰਦਰ bool work_ਕਤਾਰ_ਲੜੀਵਾਰ(work_ਕਤਾਰ_t q, work_ਕਤਾਰ_func_t func, ਖਾਲੀ * ਅੰਕੜੇ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n) {
    ਜੇ (n <= 1) {
        func(n, 0, ਅੰਕੜੇ);
        ਵਾਪਸ true;
    }
    ਵਾਪਸ work_ਕਤਾਰ_ਲੜੀਵਾਰ_async(q, func, ਅੰਕੜੇ, n);
}

// Legacy compatibility
ਕਿਸਮਪਰਿਭਾਸ਼ਾ work_ਕਤਾਰ_func_t     ਕਾਮਾ_callback_t;
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਕਾਮਾ_pool_ਲੜੀਵਾਰ_func  work_ਕਤਾਰ_ਲੜੀਵਾਰ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਕਾਮਾ_pool           work_ਕਤਾਰ

#ਜੇ_ਅੰਤ  // #ifndef HTP_WORK_QUEUE_H
