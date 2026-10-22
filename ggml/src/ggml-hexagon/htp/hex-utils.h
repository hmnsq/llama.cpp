#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <qurt_memory.h>
#அந்தர்பாவ <qurt.h>

#அந்தர்பாவ "hexagon_types.h"
#அந்தர்பாவ "hexagon_protos.h"

#அந்தர்பாவ "hex-fastdiv.h"
#அந்தர்பாவ "hex-dump.h"
#அந்தர்பாவ "hex-common.h"

நிலையான உள்வரி uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm நிலையற்ற(" %0 = c15:14\n" : "=r"(cycles));
    திரும்பு cycles;
}

நிலையான உள்வரி uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm நிலையற்ற(" %0 = c19:18\n" : "=r"(pktcnt));
    திரும்பு pktcnt;
}

நிலையான உள்வரி வெறுமை hex_l2fetch(மாறிலி வெறுமை * p, uint32_t width, uint32_t stride, uint32_t height) {
    மாறிலி uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((வெறுமை *) p, control);
}

நிலையான உள்வரி வெறுமை hex_l2fetch_block(மாறிலி வெறுமை * addr, size_t size) {
    எனில் (size == 0) திரும்பு;
    மாறிலி uint32_t width = 16384; // 16KB rows
    மாறிலி uint32_t height = (size + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#define HEX_L2_வரி_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_வரி_SIZE * 4) // flush granularity (chunks per thread)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

நிலையான உள்வரி வெறுமை hex_l2flush(வெறுமை * addr, size_t size) {
    எனில் (size == 0) திரும்பு;
    மாறிலி uint32_t s = ((uint32_t) addr) & ~(HEX_L2_வரி_SIZE - 1);
    மாறிலி uint32_t e = (((uint32_t) addr) + size + HEX_L2_வரி_SIZE - 1) & ~(HEX_L2_வரி_SIZE - 1);
    க்கு (uint32_t i = s; i < e; i += HEX_L2_வரி_SIZE) {
        Q6_dccleaninva_A((வெறுமை *) i);
    }
}

நிலையான உள்வரி வெறுமை hex_pause() {
    asm நிலையற்ற(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
