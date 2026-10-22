#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <qurt_memory.h>
#અન્તર્ભાવ <qurt.h>

#અન્તર્ભાવ "hexagon_types.h"
#અન્તર્ભાવ "hexagon_protos.h"

#અન્તર્ભાવ "hex-fastdiv.h"
#અન્તર્ભાવ "hex-dump.h"
#અન્તર્ભાવ "hex-common.h"

સ્થિર અંતઃરેખ uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm અસ્થિર(" %0 = c15:14\n" : "=r"(cycles));
    પાછા cycles;
}

સ્થિર અંતઃરેખ uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm અસ્થિર(" %0 = c19:18\n" : "=r"(pktcnt));
    પાછા pktcnt;
}

સ્થિર અંતઃરેખ શૂન્ય hex_l2fetch(અચલ શૂન્ય * p, uint32_t width, uint32_t stride, uint32_t height) {
    અચલ uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((શૂન્ય *) p, control);
}

સ્થિર અંતઃરેખ શૂન્ય hex_l2fetch_block(અચલ શૂન્ય * addr, size_t size) {
    જો (size == 0) પાછા;
    અચલ uint32_t width = 16384; // 16KB rows
    અચલ uint32_t height = (size + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#define HEX_L2_લીટી_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_લીટી_SIZE * 4) // flush granularity (chunks per thread)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

સ્થિર અંતઃરેખ શૂન્ય hex_l2flush(શૂન્ય * addr, size_t size) {
    જો (size == 0) પાછા;
    અચલ uint32_t s = ((uint32_t) addr) & ~(HEX_L2_લીટી_SIZE - 1);
    અચલ uint32_t e = (((uint32_t) addr) + size + HEX_L2_લીટી_SIZE - 1) & ~(HEX_L2_લીટી_SIZE - 1);
    માટે (uint32_t i = s; i < e; i += HEX_L2_લીટી_SIZE) {
        Q6_dccleaninva_A((શૂન્ય *) i);
    }
}

સ્થિર અંતઃરેખ શૂન્ય hex_pause() {
    asm અસ્થિર(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
