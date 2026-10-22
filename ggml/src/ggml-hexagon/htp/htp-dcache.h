#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

संरचना htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

संरचना htp_context;

शून्य htp_dcache_flush_dirty_ranges(संरचना htp_context * ctx);
शून्य htp_dcache_flush_dirty_tensors(संरचना htp_context * ctx, स्थिरांक संरचना htp_tensor * स्थिरांक * tensors, uint32_t n);
शून्य htp_dcache_mark_dirty_tensors(संरचना htp_context * ctx, स्थिरांक संरचना htp_tensor * स्थिरांक * tensors, uint32_t n);

#endif // HTP_DCACHE_H
