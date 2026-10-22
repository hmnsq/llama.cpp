#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

ഘടന htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

ഘടന htp_context;

ശൂന്യം htp_dcache_flush_dirty_ranges(ഘടന htp_context * ctx);
ശൂന്യം htp_dcache_flush_dirty_tensors(ഘടന htp_context * ctx, സ്ഥിരം ഘടന htp_tensor * സ്ഥിരം * tensors, uint32_t n);
ശൂന്യം htp_dcache_mark_dirty_tensors(ഘടന htp_context * ctx, സ്ഥിരം ഘടന htp_tensor * സ്ഥിരം * tensors, uint32_t n);

#endif // HTP_DCACHE_H
