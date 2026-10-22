#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

ସଂରଚନା htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

ସଂରଚନା htp_context;

ଶୂନ୍ୟ htp_dcache_flush_dirty_ranges(ସଂରଚନା htp_context * ctx);
ଶୂନ୍ୟ htp_dcache_flush_dirty_tensors(ସଂରଚନା htp_context * ctx, ଅଚଳ ସଂରଚନା htp_tensor * ଅଚଳ * tensors, uint32_t n);
ଶୂନ୍ୟ htp_dcache_mark_dirty_tensors(ସଂରଚନା htp_context * ctx, ଅଚଳ ସଂରଚନା htp_tensor * ଅଚଳ * tensors, uint32_t n);

#endif // HTP_DCACHE_H
