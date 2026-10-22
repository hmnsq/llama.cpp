#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

সংরচনা htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

সংরচনা htp_context;

শূন্য htp_dcache_flush_dirty_ranges(সংরচনা htp_context * ctx);
শূন্য htp_dcache_flush_dirty_tensors(সংরচনা htp_context * ctx, ধ্রুৱ সংরচনা htp_tensor * ধ্রুৱ * tensors, uint32_t n);
শূন্য htp_dcache_mark_dirty_tensors(সংরচনা htp_context * ctx, ধ্রুৱ সংরচনা htp_tensor * ধ্রুৱ * tensors, uint32_t n);

#endif // HTP_DCACHE_H
