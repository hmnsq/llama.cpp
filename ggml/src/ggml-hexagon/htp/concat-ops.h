#ifndef HTP_CONCAT_OPS_H
#define HTP_CONCAT_OPS_H

#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "gather-rows.h"
#അന്തര്ഭാവ <stdint.h>

എണ്ണൽ htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_ROWS = 3,
};

ഘടന htp_concat_transposed_params {
    uint32_t spad0_size_per_thread;
    uint32_t spad1_size_per_thread;
};

ഘടന htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_threads;
    uint8_t  pad;

    uint32_t vtcm_size;

    സംഘം {
        ഘടന htp_concat_transposed_params transposed;
        ഘടന htp_gather_rows_params       gather;
    } u;
};

#എങ്കിൽ defined(__cplusplus)
static_assert(വലുപ്പം(ഘടന htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#അല്ലെങ്കിൽ
_Static_assert(വലുപ്പം(ഘടന htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#endif

ഘടന htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_size_per_thread;
    uint32_t src1_spad_size_per_thread;
    uint32_t total_bytes;
};

സ്ഥിതം വരിയിൽ ശൂന്യം htp_concat_transposed_vtcm_layout_build(
    ഘടന htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_size,
    uint32_t n_threads) {

    uint32_t block_i = (type_size == 4) ? 32 : 64;
    uint32_t spad1_stride = block_i * type_size;
    uint32_t src1_ne0_padded = hex_round_up(src1_ne0, block_i);
    uint32_t spad0_row_bytes = hex_round_up(src0_ne0 * type_size, 128) + src1_ne0_padded * type_size;

    layout->src0_spad_size_per_thread = block_i * spad0_row_bytes;
    layout->src1_spad_size_per_thread = src1_ne0_padded * spad1_stride;
    layout->total_bytes = n_threads * (layout->src0_spad_size_per_thread + layout->src1_spad_size_per_thread);
}

#endif // HTP_CONCAT_OPS_H
