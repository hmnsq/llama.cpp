#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#include <stdint.h>
#include "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

sa_nracanaa htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

sa_nracanaa htp_context;

void htp_dcache_flush_dirty_ranges(sa_nracanaa htp_context * ctx);
void htp_dcache_flush_dirty_tensors(sa_nracanaa htp_context * ctx, const sa_nracanaa htp_tensor * const * tensors, uint32_t n);
void htp_dcache_mark_dirty_tensors(sa_nracanaa htp_context * ctx, const sa_nracanaa htp_tensor * const * tensors, uint32_t n);

#endif // HTP_DCACHE_H
