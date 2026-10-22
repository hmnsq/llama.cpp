#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HEX_UTILS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HEX_UTILS_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <qurt_memory.h>
#ಅನ್ತರ್ಭಾವ <qurt.h>

#ಅನ್ತರ್ಭಾವ "hexagon_types.h"
#ಅನ್ತರ್ಭಾವ "hexagon_protos.h"

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-dump.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t hex_get_ಚಕ್ರಗಳು() {
    uint64_t ಚಕ್ರಗಳು = 0;
    asm ಅಸ್ಥಿರ(" %0 = c15:14\n" : "=r"(ಚಕ್ರಗಳು));
    ಹಿಂತಿರುಗಿ ಚಕ್ರಗಳು;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm ಅಸ್ಥಿರ(" %0 = c19:18\n" : "=r"(pktcnt));
    ಹಿಂತಿರುಗಿ pktcnt;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_l2fetch(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * p, uint32_t ಅಗಲ, uint32_t stride, uint32_t ಎತ್ತರ) {
    ಸ್ಥಿರಾಂಕ uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(ಅಗಲ, ಎತ್ತರ));
    Q6_l2fetch_AP((ಶೂನ್ಯ *) p, control);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_l2fetch_ಖಂಡ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * addr, size_t ಗಾತ್ರ_2) {
    ಆದರೆ (ಗಾತ್ರ_2 == 0) ಹಿಂತಿರುಗಿ;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಗಲ = 16384; // 16KB rows
    ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ = (ಗಾತ್ರ_2 + ಅಗಲ - 1) / ಅಗಲ;
    hex_l2fetch(addr, ಅಗಲ, ಅಗಲ, ಎತ್ತರ);
}

#ವ್ಯಾಖ್ಯಾನಿಸು HEX_L2_ಸಾಲು_ಗಾತ್ರ           128
#ವ್ಯಾಖ್ಯಾನಿಸು HEX_L2_ಖಂಡ_ಗಾತ್ರ          (HEX_L2_ಸಾಲು_ಗಾತ್ರ * 4) // flush granularity (chunks per thread)
#ವ್ಯಾಖ್ಯಾನಿಸು HEX_L2_FLUSH_WQ_ಮಿತಿ  (4 * 1024)
#ವ್ಯಾಖ್ಯಾನಿಸು HEX_L2_FLUSH_ALL_ಮಿತಿ (4 * 1024 * 1024)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_l2flush(ಶೂನ್ಯ * addr, size_t ಗಾತ್ರ_2) {
    ಆದರೆ (ಗಾತ್ರ_2 == 0) ಹಿಂತಿರುಗಿ;
    ಸ್ಥಿರಾಂಕ uint32_t s = ((uint32_t) addr) & ~(HEX_L2_ಸಾಲು_ಗಾತ್ರ - 1);
    ಸ್ಥಿರಾಂಕ uint32_t e = (((uint32_t) addr) + ಗಾತ್ರ_2 + HEX_L2_ಸಾಲು_ಗಾತ್ರ - 1) & ~(HEX_L2_ಸಾಲು_ಗಾತ್ರ - 1);
    ಗಾಗಿ (uint32_t i = s; i < e; i += HEX_L2_ಸಾಲು_ಗಾತ್ರ) {
        Q6_dccleaninva_A((ಶೂನ್ಯ *) i);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_pause() {
    asm ಅಸ್ಥಿರ(" pause(#255)\n");
}

#ಆದರೆ_ಅಂತ್ಯ /* HEX_UTILS_H */
