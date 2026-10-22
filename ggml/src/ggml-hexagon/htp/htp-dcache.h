#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

ಸಂರಚನಾ htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

ಸಂರಚನಾ htp_context;

ಶೂನ್ಯ htp_dcache_flush_dirty_ranges(ಸಂರಚನಾ htp_context * ctx);
ಶೂನ್ಯ htp_dcache_flush_dirty_tensors(ಸಂರಚನಾ htp_context * ctx, ಧ್ರುವ ಸಂರಚನಾ htp_tensor * ಧ್ರುವ * tensors, uint32_t n);
ಶೂನ್ಯ htp_dcache_mark_dirty_tensors(ಸಂರಚನಾ htp_context * ctx, ಧ್ರುವ ಸಂರಚನಾ htp_tensor * ಧ್ರುವ * tensors, uint32_t n);

#endif // HTP_DCACHE_H
