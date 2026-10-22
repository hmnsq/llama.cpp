#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#ضمن <stdint.h>
#ضمن "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

بنية htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

بنية htp_context;

فراغ htp_dcache_flush_dirty_ranges(بنية htp_context * ctx);
فراغ htp_dcache_flush_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n);
فراغ htp_dcache_mark_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n);

#endif // HTP_DCACHE_H
