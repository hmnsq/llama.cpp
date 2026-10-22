#اگر_غیر_تعریف_شدہ HTP_DCACHE_H
#تعریف_کرو HTP_DCACHE_H

#شامل <stdint.h>
#شامل "htp-ops.h"

#تعریف_کرو HTP_MAX_DIRTY_RANGES 64

ساخت htp_dirty_range {
    uint32_t آغاز;
    uint32_t انتہا;
};

ساخت htp_context;

خالی htp_dcache_flush_dirty_ranges(ساخت htp_context * ctx);
خالی htp_dcache_flush_dirty_tensors(ساخت htp_context * ctx, ثابت ساخت htp_tensor * ثابت * tensors, uint32_t n);
خالی htp_dcache_mark_dirty_tensors(ساخت htp_context * ctx, ثابت ساخت htp_tensor * ثابت * tensors, uint32_t n);

#اگر_ختم // HTP_DCACHE_H
