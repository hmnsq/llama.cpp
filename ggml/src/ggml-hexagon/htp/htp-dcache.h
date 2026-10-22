#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

સંરચના htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

સંરચના htp_context;

શૂન્ય htp_dcache_flush_dirty_ranges(સંરચના htp_context * ctx);
શૂન્ય htp_dcache_flush_dirty_tensors(સંરચના htp_context * ctx, અચલ સંરચના htp_tensor * અચલ * tensors, uint32_t n);
શૂન્ય htp_dcache_mark_dirty_tensors(સંરચના htp_context * ctx, અચલ સંરચના htp_tensor * અચલ * tensors, uint32_t n);

#endif // HTP_DCACHE_H
