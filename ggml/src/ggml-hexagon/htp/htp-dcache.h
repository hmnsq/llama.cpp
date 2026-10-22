#నిర్వచితం_కాకపోతే HTP_DCACHE_H
#నిర్వచించు HTP_DCACHE_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ "htp-ops.h"

#నిర్వచించు HTP_MAX_DIRTY_RANGES 64

నిర్మాణం htp_dirty_range {
    uint32_t ప్రారంభం;
    uint32_t చివర;
};

నిర్మాణం htp_context;

శూన్య htp_dcache_flush_dirty_ranges(నిర్మాణం htp_context * ctx);
శూన్య htp_dcache_flush_dirty_tensors(నిర్మాణం htp_context * ctx, స్థిరరాశి నిర్మాణం htp_tensor * స్థిరరాశి * tensors, uint32_t n);
శూన్య htp_dcache_mark_dirty_tensors(నిర్మాణం htp_context * ctx, స్థిరరాశి నిర్మాణం htp_tensor * స్థిరరాశి * tensors, uint32_t n);

#అయితే_ముగింపు // HTP_DCACHE_H
