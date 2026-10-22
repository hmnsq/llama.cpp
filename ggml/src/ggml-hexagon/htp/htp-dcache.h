#جيڪڏهن_وصف_نه_ٿيل HTP_DCACHE_H
#وصف_ڪريو HTP_DCACHE_H

#ضمن <stdint.h>
#ضمن "htp-ops.h"

#وصف_ڪريو HTP_MAX_DIRTY_RANGES 64

بنية htp_dirty_range {
    uint32_t شروعات;
    uint32_t پڇاڙي;
};

بنية htp_context;

فراغ htp_dcache_flush_dirty_ranges(بنية htp_context * ctx);
فراغ htp_dcache_flush_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n);
فراغ htp_dcache_mark_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n);

#جيڪڏهن_پورو // HTP_DCACHE_H
