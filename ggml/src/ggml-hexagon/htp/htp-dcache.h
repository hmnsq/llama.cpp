#که_نه_تعریف_شوی HTP_DCACHE_H
#تعریف_کړه HTP_DCACHE_H

#ضمن <stdint.h>
#ضمن "htp-ops.h"

#تعریف_کړه HTP_MAX_DIRTY_RANGES 64

بنية htp_dirty_range {
    uint32_t پیل;
    uint32_t پای_ټکی;
};

بنية htp_context;

فراغ htp_dcache_flush_dirty_ranges(بنية htp_context * ctx);
فراغ htp_dcache_flush_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n);
فراغ htp_dcache_mark_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n);

#د_که_پای // HTP_DCACHE_H
