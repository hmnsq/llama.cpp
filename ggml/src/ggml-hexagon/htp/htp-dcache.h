#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

কাঠামো htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

কাঠামো htp_context;

শূন্য htp_dcache_flush_dirty_ranges(কাঠামো htp_context * ctx);
শূন্য htp_dcache_flush_dirty_tensors(কাঠামো htp_context * ctx, ধ্রুবক কাঠামো htp_tensor * ধ্রুবক * tensors, uint32_t n);
শূন্য htp_dcache_mark_dirty_tensors(কাঠামো htp_context * ctx, ধ্রুবক কাঠামো htp_tensor * ধ্রুবক * tensors, uint32_t n);

#endif // HTP_DCACHE_H
