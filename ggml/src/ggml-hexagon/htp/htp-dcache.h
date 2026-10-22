#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_DCACHE_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_DCACHE_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ "htp-ops.h"

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MAX_DIRTY_RANGES 64

ರಚನೆ htp_dirty_range {
    uint32_t ಆರಂಭ;
    uint32_t ಕೊನೆ;
};

ರಚನೆ htp_context;

ಶೂನ್ಯ htp_dcache_flush_dirty_ranges(ರಚನೆ htp_context * ctx);
ಶೂನ್ಯ htp_dcache_flush_dirty_tensors(ರಚನೆ htp_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸ್ಥಿರಾಂಕ * tensors, uint32_t n);
ಶೂನ್ಯ htp_dcache_mark_dirty_tensors(ರಚನೆ htp_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸ್ಥಿರಾಂಕ * tensors, uint32_t n);

#ಆದರೆ_ಅಂತ್ಯ // HTP_DCACHE_H
