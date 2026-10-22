#अगर_अघोषित HTP_CONCAT_OPS_H
#घोषणा HTP_CONCAT_OPS_H

#समावेश "hex-common.h"
#समावेश "gather-rows.h"
#समावेश <stdint.h>

क्रमागत htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_ROWS = 3,
};

काष्ठा htp_concat_transposed_params {
    uint32_t spad0_आकार_per_thread;
    uint32_t spad1_आकार_per_thread;
};

काष्ठा htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_threads;
    uint8_t  pad;

    uint32_t vtcm_आकार;

    जोड़ {
        काष्ठा htp_concat_transposed_params transposed;
        काष्ठा htp_gather_rows_params       gather;
    } u;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

काष्ठा htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_आकार_per_thread;
    uint32_t src1_spad_आकार_per_thread;
    uint32_t योग_bytes;
};

अटल अंतरभूत व्योम htp_concat_transposed_vtcm_layout_build(
    काष्ठा htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_आकार,
    uint32_t n_threads) {

    uint32_t block_i = (type_आकार == 4) ? 32 : 64;
    uint32_t spad1_stride = block_i * type_आकार;
    uint32_t src1_ne0_padded = hex_round_up(src1_ne0, block_i);
    uint32_t spad0_row_bytes = hex_round_up(src0_ne0 * type_आकार, 128) + src1_ne0_padded * type_आकार;

    layout->src0_spad_आकार_per_thread = block_i * spad0_row_bytes;
    layout->src1_spad_आकार_per_thread = src1_ne0_padded * spad1_stride;
    layout->योग_bytes = n_threads * (layout->src0_spad_आकार_per_thread + layout->src1_spad_आकार_per_thread);
}

#पूर्ण_अगर // HTP_CONCAT_OPS_H
