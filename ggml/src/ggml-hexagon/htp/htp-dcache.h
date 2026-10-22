#اگر_تعریف_نشده HTP_DCACHE_H
#تعریف_کن HTP_DCACHE_H

#ضمن <stdint.h>
#ضمن "htp-ops.h"

#تعریف_کن HTP_MAX_DIRTY_RANGES 64

بنية htp_dirty_range {
    uint32_t شروع;
    uint32_t انتها;
};

بنية htp_context;

فراغ htp_dcache_flush_dirty_ranges(بنية htp_context * ctx);
فراغ htp_dcache_flush_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n);
فراغ htp_dcache_mark_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n);

#پایان_اگر // HTP_DCACHE_H
