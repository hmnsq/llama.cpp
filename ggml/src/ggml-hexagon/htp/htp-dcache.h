#यदि_अपरिभाषितम् HTP_DCACHE_H
#परिभाषय HTP_DCACHE_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव "htp-ops.h"

#परिभाषय HTP_MAX_DIRTY_RANGES 64

संरचना htp_dirty_range {
    uint32_t start;
    uint32_t अन्तः_2;
};

संरचना htp_context;

शून्य htp_dcache_flush_dirty_ranges(संरचना htp_context * ctx);
शून्य htp_dcache_flush_dirty_tensors(संरचना htp_context * ctx, ध्रुव संरचना htp_tensor * ध्रुव * tensors, uint32_t n);
शून्य htp_dcache_mark_dirty_tensors(संरचना htp_context * ctx, ध्रुव संरचना htp_tensor * ध्रुव * tensors, uint32_t n);

#यदि_अन्तः // HTP_DCACHE_H
