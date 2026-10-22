#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <qurt_memory.h>
#അന്തര്ഭാവ <qurt.h>

#അന്തര്ഭാവ "hexagon_types.h"
#അന്തര്ഭാവ "hexagon_protos.h"

#അന്തര്ഭാവ "hex-fastdiv.h"
#അന്തര്ഭാവ "hex-dump.h"
#അന്തര്ഭാവ "hex-common.h"

സ്ഥിതം വരിയിൽ uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm അസ്ഥിര(" %0 = c15:14\n" : "=r"(cycles));
    തിരികെ cycles;
}

സ്ഥിതം വരിയിൽ uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm അസ്ഥിര(" %0 = c19:18\n" : "=r"(pktcnt));
    തിരികെ pktcnt;
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_l2fetch(സ്ഥിരം ശൂന്യം * p, uint32_t width, uint32_t stride, uint32_t height) {
    സ്ഥിരം uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((ശൂന്യം *) p, control);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_l2fetch_block(സ്ഥിരം ശൂന്യം * addr, size_t size) {
    എങ്കിൽ (size == 0) തിരികെ;
    സ്ഥിരം uint32_t width = 16384; // 16KB rows
    സ്ഥിരം uint32_t height = (size + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#define HEX_L2_വരി_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_വരി_SIZE * 4) // flush granularity (chunks per thread)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

സ്ഥിതം വരിയിൽ ശൂന്യം hex_l2flush(ശൂന്യം * addr, size_t size) {
    എങ്കിൽ (size == 0) തിരികെ;
    സ്ഥിരം uint32_t s = ((uint32_t) addr) & ~(HEX_L2_വരി_SIZE - 1);
    സ്ഥിരം uint32_t e = (((uint32_t) addr) + size + HEX_L2_വരി_SIZE - 1) & ~(HEX_L2_വരി_SIZE - 1);
    വേണ്ടി (uint32_t i = s; i < e; i += HEX_L2_വരി_SIZE) {
        Q6_dccleaninva_A((ശൂന്യം *) i);
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_pause() {
    asm അസ്ഥിര(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
