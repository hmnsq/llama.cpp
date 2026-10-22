#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#شامل <stdint.h>
#شامل "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

ساختار htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

ساختار htp_context;

تهی htp_dcache_flush_dirty_ranges(ساختار htp_context * ctx);
تهی htp_dcache_flush_dirty_tensors(ساختار htp_context * ctx, ثابت ساختار htp_tensor * ثابت * tensors, uint32_t n);
تهی htp_dcache_mark_dirty_tensors(ساختار htp_context * ctx, ثابت ساختار htp_tensor * ثابت * tensors, uint32_t n);

#endif // HTP_DCACHE_H
