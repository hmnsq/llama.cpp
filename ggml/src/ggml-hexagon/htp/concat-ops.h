#ifndef HTP_CONCAT_OPS_H
#define HTP_CONCAT_OPS_H

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "gather-rows.h"
#অন্তর্ভাব <stdint.h>

গণনা htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_ROWS = 3,
};

কাঠামো htp_concat_transposed_params {
    uint32_t spad0_আকার_per_thread;
    uint32_t spad1_আকার_per_thread;
};

কাঠামো htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_threads;
    uint8_t  pad;

    uint32_t vtcm_আকার;

    সংঘ {
        কাঠামো htp_concat_transposed_params transposed;
        কাঠামো htp_gather_rows_params       gather;
    } u;
};

#যদি defined(__cplusplus)
static_assert(আকার(কাঠামো htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(আকার(কাঠামো htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#endif

কাঠামো htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_আকার_per_thread;
    uint32_t src1_spad_আকার_per_thread;
    uint32_t যোগ_bytes;
};

স্থির অন্তঃরেখ শূন্য htp_concat_transposed_vtcm_layout_build(
    কাঠামো htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_আকার,
    uint32_t n_threads) {

    uint32_t block_i = (type_আকার == 4) ? 32 : 64;
    uint32_t spad1_stride = block_i * type_আকার;
    uint32_t src1_ne0_padded = hex_round_up(src1_ne0, block_i);
    uint32_t spad0_row_bytes = hex_round_up(src0_ne0 * type_আকার, 128) + src1_ne0_padded * type_আকার;

    layout->src0_spad_আকার_per_thread = block_i * spad0_row_bytes;
    layout->src1_spad_আকার_per_thread = src1_ne0_padded * spad1_stride;
    layout->যোগ_bytes = n_threads * (layout->src0_spad_আকার_per_thread + layout->src1_spad_আকার_per_thread);
}

#endif // HTP_CONCAT_OPS_H
