#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

கட்டமைப்பு htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

கட்டமைப்பு htp_context;

வெறுமை htp_dcache_flush_dirty_ranges(கட்டமைப்பு htp_context * ctx);
வெறுமை htp_dcache_flush_dirty_tensors(கட்டமைப்பு htp_context * ctx, மாறிலி கட்டமைப்பு htp_tensor * மாறிலி * tensors, uint32_t n);
வெறுமை htp_dcache_mark_dirty_tensors(கட்டமைப்பு htp_context * ctx, மாறிலி கட்டமைப்பு htp_tensor * மாறிலி * tensors, uint32_t n);

#endif // HTP_DCACHE_H
