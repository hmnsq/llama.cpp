#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_DCACHE_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_DCACHE_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MAX_DIRTY_RANGES 64

ਬਣਤਰ htp_dirty_range {
    uint32_t ਸ਼ੁਰੂ;
    uint32_t ਸਿਰਾ;
};

ਬਣਤਰ htp_context;

ਖਾਲੀ htp_dcache_flush_dirty_ranges(ਬਣਤਰ htp_context * ctx);
ਖਾਲੀ htp_dcache_flush_dirty_tensors(ਬਣਤਰ htp_context * ctx, ਅਚਲ ਬਣਤਰ htp_tensor * ਅਚਲ * tensors, uint32_t n);
ਖਾਲੀ htp_dcache_mark_dirty_tensors(ਬਣਤਰ htp_context * ctx, ਅਚਲ ਬਣਤਰ htp_tensor * ਅਚਲ * tensors, uint32_t n);

#ਜੇ_ਅੰਤ // HTP_DCACHE_H
