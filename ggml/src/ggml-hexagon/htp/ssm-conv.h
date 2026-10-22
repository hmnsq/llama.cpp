#अगर_अघोषित HTP_SSM_CONV_H
#घोषणा HTP_SSM_CONV_H

#समावेश <stdint.h>

#समावेश "hex-fastdiv.h"
#समावेश "htp-ops.h"

काष्ठा htp_ssm_conv_kernel_params {
    uint32_t n_threads;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_आकार_per_thread;
    uint32_t vtcm_src1_आकार_per_thread;
    uint32_t vtcm_dst_आकार_per_thread;

    uint32_t vtcm_src0_आकार;
    uint32_t vtcm_src1_आकार;
    uint32_t vtcm_dst_आकार;
    uint32_t vtcm_आकार;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

#पूर्ण_अगर // HTP_SSM_CONV_H
