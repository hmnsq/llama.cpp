#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <qurt_memory.h>
#ଅନ୍ତର୍ଭାଵ <qurt.h>

#ଅନ୍ତର୍ଭାଵ "hexagon_types.h"
#ଅନ୍ତର୍ଭାଵ "hexagon_protos.h"

#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"
#ଅନ୍ତର୍ଭାଵ "hex-dump.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm ଅସ୍ଥିର(" %0 = c15:14\n" : "=r"(cycles));
    ଫେରନ୍ତୁ cycles;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm ଅସ୍ଥିର(" %0 = c19:18\n" : "=r"(pktcnt));
    ଫେରନ୍ତୁ pktcnt;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_l2fetch(ଅଚଳ ଶୂନ୍ୟ * p, uint32_t width, uint32_t stride, uint32_t height) {
    ଅଚଳ uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((ଶୂନ୍ୟ *) p, control);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_l2fetch_block(ଅଚଳ ଶୂନ୍ୟ * addr, size_t size) {
    ଯଦି (size == 0) ଫେରନ୍ତୁ;
    ଅଚଳ uint32_t width = 16384; // 16KB rows
    ଅଚଳ uint32_t height = (size + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#define HEX_L2_ଧାଡ଼ି_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_ଧାଡ଼ି_SIZE * 4) // flush granularity (chunks per thread)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_l2flush(ଶୂନ୍ୟ * addr, size_t size) {
    ଯଦି (size == 0) ଫେରନ୍ତୁ;
    ଅଚଳ uint32_t s = ((uint32_t) addr) & ~(HEX_L2_ଧାଡ଼ି_SIZE - 1);
    ଅଚଳ uint32_t e = (((uint32_t) addr) + size + HEX_L2_ଧାଡ଼ି_SIZE - 1) & ~(HEX_L2_ଧାଡ଼ି_SIZE - 1);
    ପାଇଁ (uint32_t i = s; i < e; i += HEX_L2_ଧାଡ଼ି_SIZE) {
        Q6_dccleaninva_A((ଶୂନ୍ୟ *) i);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_pause() {
    asm ଅସ୍ଥିର(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
