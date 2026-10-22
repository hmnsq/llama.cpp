#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_METAL_IMPL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_METAL_IMPL

// kernel parameters for mat-mat threadgroups
//
// TODO: become function constants

#ವ್ಯಾಖ್ಯಾನಿಸು SZ_SIMDGROUP 16
#ವ್ಯಾಖ್ಯಾನಿಸು N_MM_NK 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_MM_NK_ಒಟ್ಟು (SZ_SIMDGROUP * N_MM_NK)

#ವ್ಯಾಖ್ಯಾನಿಸು N_MM_ಖಂಡ_X 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_MM_ಖಂಡ_Y 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_MM_SIMD_ಗುಂಪು_X 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_MM_SIMD_ಗುಂಪು_Y 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_MM_NPART_AMAX 256

#ವ್ಯಾಖ್ಯಾನಿಸು N_ARGMAX_NPART_MAX 256
#ವ್ಯಾಖ್ಯಾನಿಸು N_ARGMAX_CHUNK 4096

// kernel parameters for mat-vec threadgroups
//
// N_R0: number of src0 rows to process per simdgroup
// N_SG: number of simdgroups per threadgroup
//
// TODO: for optimal performance, become function of the device and work size

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q1_0 8
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q1_0 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q2_0 8
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q2_0 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q4_0 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q4_0 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q4_1 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q4_1 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q5_0 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q5_0 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q5_1 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q5_1 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q8_0 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q8_0 4

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_MXFP4 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_MXFP4 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q2_K 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q2_K 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q3_K 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q3_K 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q4_K 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q4_K 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q5_K 1
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q5_K 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_Q6_K 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_Q6_K 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ1_S 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_IQ1_S 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ1_S_ವಿಭಜಿಸು 8

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ1_M 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_IQ1_M 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ1_M_ವಿಭಜಿಸು 8

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ2_XXS 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_IQ2_XXS 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ2_XXS_ವಿಭಜಿಸು 8

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ2_XS 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_IQ2_XS 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ2_XS_ವಿಭಜಿಸು 8

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ2_S 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_IQ2_S 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ2_S_ವಿಭಜಿಸು 8

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ3_XXS 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_IQ3_XXS 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ3_XXS_ವಿಭಜಿಸು 8

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ3_S 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_IQ3_S 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ3_S_ವಿಭಜಿಸು 8

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ4_NL 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_IQ4_NL 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_IQ4_XS 2
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_IQ4_XS 2

#ವ್ಯಾಖ್ಯಾನಿಸು N_R0_TQ2_0 4
#ವ್ಯಾಖ್ಯಾನಿಸು N_SG_TQ2_0 2

// function constants offsets
#ವ್ಯಾಖ್ಯಾನಿಸು FC_FLASH_ATTN_EXT_PAD          100
#ವ್ಯಾಖ್ಯಾನಿಸು FC_FLASH_ATTN_EXT_BLK          200
#ವ್ಯಾಖ್ಯಾನಿಸು FC_FLASH_ATTN_EXT              300
#ವ್ಯಾಖ್ಯಾನಿಸು FC_FLASH_ATTN_EXT_VEC          400
#ವ್ಯಾಖ್ಯಾನಿಸು FC_FLASH_ATTN_EXT_VEC_REDUCE   500
#ವ್ಯಾಖ್ಯಾನಿಸು FC_MUL_MV                      600
#ವ್ಯಾಖ್ಯಾನಿಸು FC_MUL_MM                      700
#ವ್ಯಾಖ್ಯಾನಿಸು FC_ROPE                        800
#ವ್ಯಾಖ್ಯಾನಿಸು FC_SSM_CONV                    900
#ವ್ಯಾಖ್ಯಾನಿಸು FC_SOLVE_TRI                   1000
#ವ್ಯಾಖ್ಯಾನಿಸು FC_ಎಣಿಕೆ_EQUAL                 1100
#ವ್ಯಾಖ್ಯಾನಿಸು FC_UNARY                       1200
#ವ್ಯಾಖ್ಯಾನಿಸು FC_BIN                         1300
#ವ್ಯಾಖ್ಯಾನಿಸು FC_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು                    1400
#ವ್ಯಾಖ್ಯಾನಿಸು FC_UPSCALE                     1500
#ವ್ಯಾಖ್ಯಾನಿಸು FC_GATED_DELTA_NET             1600
#ವ್ಯಾಖ್ಯಾನಿಸು FC_NORM                        1700
#ವ್ಯಾಖ್ಯಾನಿಸು FC_TOPK_MOE                    1800
#ವ್ಯಾಖ್ಯಾನಿಸು FC_MOE_REDUCE                  1900
#ವ್ಯಾಖ್ಯಾನಿಸು FC_DSV4_HC                     2000
#ವ್ಯಾಖ್ಯಾನಿಸು FC_PAD                         2100
#ವ್ಯಾಖ್ಯಾನಿಸು FC_FLASH_ATTN_EXT_TENSOR       2200
#ವ್ಯಾಖ್ಯಾನಿಸು FC_LIGHTNING_INDEXER           2200
#ವ್ಯಾಖ್ಯಾನಿಸು FC_MUL_MV_MMA                  2300

// op-specific constants
#ವ್ಯಾಖ್ಯಾನಿಸು OP_FLASH_ATTN_EXT_NQPSG 8
#ವ್ಯಾಖ್ಯಾನಿಸು OP_FLASH_ATTN_EXT_NCPSG 64

#ವ್ಯಾಖ್ಯಾನಿಸು OP_FLASH_ATTN_EXT_TENSOR_NQPSG       32
#ವ್ಯಾಖ್ಯಾನಿಸು OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE 16
#ವ್ಯಾಖ್ಯಾನಿಸು OP_FLASH_ATTN_EXT_TENSOR_NCPSG       64
#ವ್ಯಾಖ್ಯಾನಿಸು OP_FLASH_ATTN_EXT_TENSOR_NSG         8

#ವ್ಯಾಖ್ಯಾನಿಸು OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#ವ್ಯಾಖ್ಯಾನಿಸು OP_FLASH_ATTN_EXT_VEC_NCPSG 32

#ವ್ಯಾಖ್ಯಾನಿಸು OP_LIGHTNING_INDEXER_DK    128
#ವ್ಯಾಖ್ಯಾನಿಸು OP_LIGHTNING_INDEXER_NHPTG   8
#ವ್ಯಾಖ್ಯಾನಿಸು OP_LIGHTNING_INDEXER_NKPSG   8
#ವ್ಯಾಖ್ಯಾನಿಸು OP_LIGHTNING_INDEXER_NSG     8
#ವ್ಯಾಖ್ಯಾನಿಸು OP_LIGHTNING_INDEXER_NBPTG   8

#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_SCALE      10
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_FILL       11
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_CLAMP      12
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_SQR        13
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_SQRT       14
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_SIN        15
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_COS        16
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_LOG        17
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_LEAKY_RELU 18

#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_TANH        100
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_RELU        101
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_SIGMOID     102
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_GELU        103
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_GELU_ERF    104
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_GELU_ಶೀಘ್ರ  105
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_SILU        106
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_ELU         107
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_NEG         108
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_ABS         109
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_SGN         110
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_ಹೆಜ್ಜೆ        111
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_HARDSWISH   112
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_HARDSIGMOID 113
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_EXP         114
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_SOFTPLUS    115
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_EXPM1       116
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_FLOOR       117
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_CEIL        118
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_ಸುತ್ತು       119
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_TRUNC       120
#ವ್ಯಾಖ್ಯಾನಿಸು OP_UNARY_ಅಂಕೆ_XIELU       121

#ವ್ಯಾಖ್ಯಾನಿಸು OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ಅಂಕೆ_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು 10
#ವ್ಯಾಖ್ಯಾನಿಸು OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ಅಂಕೆ_ಸರಾಸರಿ     11

#ವ್ಯಾಖ್ಯಾನಿಸು OP_SSM_ಪರಿಶೀಲನೆ_SSD_CS  64 // Metal-specific; Chunk Size; 64 is largest multiple of 8 (simdgroup tile) fitting into 32 KiB Metal threadgroup mem limit (~26.75 KiB shared mem; see smem layout comment in kernel_ssm_scan_ssd_mma_f32)
#ವ್ಯಾಖ್ಯಾನಿಸು OP_SSM_ಪರಿಶೀಲನೆ_SSD_HD  64 // Metal-specific; Head Dim the MMA kernel is specialized for (Mamba-2); use_mma gates on d_inner == this
#ವ್ಯಾಖ್ಯಾನಿಸು OP_SSM_ಪರಿಶೀಲನೆ_SSD_NSG 4  // Metal-specific; Number of SimdGroups per threadgroup; NSG*32 == threads dispatched per threadgroup

// kernel argument structs
//
// - element counters (e.g. ne00) typically use int32_t to reduce register usage
//   however, be careful from int overflows when using those in the kernel implementation
//
// - strides (e.g. nb00) use uint64_t

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  dim;
    int32_t  nc0;
} ggml_metal_kargs_concat;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    ದಶಮಾಂಶ    ಇಳಿಜಾರು;
    ದಶಮಾಂಶ    scale;
    ದಶಮಾಂಶ    bias;
    ದಶಮಾಂಶ    val;
    ದಶಮಾಂಶ    min;
    ದಶಮಾಂಶ    max;
} ggml_metal_kargs_unary;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    uint64_t o1[8];
} ggml_metal_kargs_bin;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t ne0;
    int64_t ne1;
    size_t nb01;
    size_t nb02;
    size_t nb11;
    size_t nb21;
} ggml_metal_kargs_ಕೂಡಿಸು_id;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_repeat;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_cpy;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne10;
    int64_t  ne11;
    int64_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    bool     inplace;
} ggml_metal_kargs_ಗಣ;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_past;
    int32_t  n_dims;
    int32_t  n_offs;
    int32_t  n_ctx_orig;
    ದಶಮಾಂಶ    freq_ತಳ;
    ದಶಮಾಂಶ    freq_scale;
    ದಶಮಾಂಶ    ext_ಗುಣಕ;
    ದಶಮಾಂಶ    attn_ಗುಣಕ;
    ದಶಮಾಂಶ    beta_fast;
    ದಶಮಾಂಶ    beta_slow;
    int32_t  sect_0;
    int32_t  sect_1;
    int32_t  sect_2;
    int32_t  sect_3;
    bool     src2;
    bool     inplace;
} ggml_metal_kargs_rope;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  nblocks;
} ggml_metal_kargs_flash_attn_ext_kv_f16;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne11;
    int32_t  ne_12_2; // assume K and V are same shape
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ggml_metal_kargs_flash_attn_ext_pad;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ggml_metal_kargs_flash_attn_ext_blk;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2; // assume K and V are same shape
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    ದಶಮಾಂಶ    scale;
    ದಶಮಾಂಶ    max_bias;
    ದಶಮಾಂಶ    m0;
    ದಶಮಾಂಶ    m1;
    int32_t  n_ತಲೆ_log2;
    ದಶಮಾಂಶ    logit_softcap;
} ggml_metal_kargs_flash_attn_ext;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2; // assume K and V are same shape
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    ದಶಮಾಂಶ    scale;
    ದಶಮಾಂಶ    max_bias;
    ದಶಮಾಂಶ    m0;
    ದಶಮಾಂಶ    m1;
    int32_t  n_ತಲೆ_log2;
    ದಶಮಾಂಶ    logit_softcap;
    int32_t  n_kv_max_padded;
} ggml_metal_kargs_flash_attn_ext_vec;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  n_kv_max;
    int32_t  n_kv_max_padded;
} ggml_metal_kargs_flash_attn_ext_vec_idx;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  nrows;
} ggml_metal_kargs_flash_attn_ext_vec_reduce;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mm;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mv;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mv_ext;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne10;
    int32_t  ne11;  // n_expert_used (bcast)
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21; // n_tokens
    int32_t  ne20;  // n_expert_used
    uint64_t nb21;
} ggml_metal_kargs_mul_mm_id_map0;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
} ggml_metal_kargs_mul_mm_id_amax;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mm_id;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
} ggml_metal_kargs_mul_mv_id;

// NORM
// RMS_NORM
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    ದಶಮಾಂಶ    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
    ದಶಮಾಂಶ    scale;
} ggml_metal_kargs_norm;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    ದಶಮಾಂಶ    eps;
} ggml_metal_kargs_l2_norm;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ngrp;
    ದಶಮಾಂಶ    eps;
} ggml_metal_kargs_ಗುಂಪು_norm;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  IC;
    int32_t  IL;
    int32_t  K;
    int32_t  s0;
    uint64_t nb0;
    uint64_t nb1;
} ggml_metal_kargs_conv_transpose_1d;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  T_ಒಳ;
    int32_t  T_ಹೊರಹೋಗು;
    int32_t  OC;
    int32_t  K;
    int32_t  K_OC;
    int32_t  s0;
    int32_t  p0;
} ggml_metal_kargs_col2im_1d;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t T;
    int32_t C;
} ggml_metal_kargs_snake;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  IC;
    int32_t  IH;
    int32_t  IW;
    int32_t  KH;
    int32_t  KW;
    int32_t  OC;
    int32_t  s0;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_conv_transpose_2d;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  IW;
    int32_t  IH;
    int32_t  KW;
    int32_t  KH;
    int32_t  IC;
    int32_t  OC;
    int32_t  OW;
    int32_t  OH;
    int32_t  N;
    int32_t  s0;
    int32_t  s1;
    int32_t  p0;
    int32_t  p1;
    int32_t  d0;
    int32_t  d1;
} ggml_metal_kargs_conv_2d;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    uint64_t nb00;  // kernel strides
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;  // input strides
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb0;   // output strides
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  IW;    // input width
    int32_t  IH;    // input height
    int32_t  KW;    // kernel width
    int32_t  KH;    // kernel height
    int32_t  C;     // channels (IC == OC for depthwise)
    int32_t  OW;    // output width
    int32_t  OH;    // output height
    int32_t  N;     // batch size
    int32_t  s0;    // stride x
    int32_t  s1;    // stride y
    int32_t  p0;    // padding x
    int32_t  p1;    // padding y
    int32_t  d0;    // dilation x
    int32_t  d1;    // dilation y
} ggml_metal_kargs_conv_2d_dw;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    uint64_t  ofs0;
    uint64_t  ofs1;
    int32_t  IW;
    int32_t  IH;
    int32_t  CHW;
    int32_t  s0;
    int32_t  s1;
    int32_t  p0;
    int32_t  p1;
    int32_t  d0;
    int32_t  d1;
    int32_t  N;
    int32_t  KH;
    int32_t  KW;
    int32_t  KHW; // KH * KW, pre-computed on CPU to save GPU resources
} ggml_metal_kargs_im2col;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  IW;
    int32_t  IH;
    int32_t  ID;
    int32_t  OW;
    int32_t  OH;
    int32_t  OD;
    int32_t  KW;
    int32_t  KH;
    int32_t  KD;
    int32_t  s0;
    int32_t  s1;
    int32_t  s2;
    int32_t  p0;
    int32_t  p1;
    int32_t  p2;
    int32_t  d0;
    int32_t  d1;
    int32_t  d2;
    int32_t  IC;
    int32_t  N;
    int32_t  OC;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_conv_3d;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ{
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    ದಶಮಾಂಶ    alpha;
    ದಶಮಾಂಶ    ಎಲ್ಲೆ;
} ggml_metal_kargs_glu;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    uint64_t np;
} ggml_metal_kargs_ಮೊತ್ತ;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  net0;
    int64_t  net1;
    int64_t  net2;
    int64_t  net3;
    uint64_t nbt0;
    uint64_t nbt1;
    uint64_t nbt2;
    uint64_t nbt3;
    bool     outb;
} ggml_metal_kargs_cumsum_blk;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  net0;
    int64_t  net1;
    int64_t  net2;
    int64_t  net3;
    uint64_t nbt0;
    uint64_t nbt1;
    uint64_t nbt2;
    uint64_t nbt3;
} ggml_metal_kargs_cumsum_ಕೂಡಿಸು;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    ದಶಮಾಂಶ    scale;
    ದಶಮಾಂಶ    max_bias;
    ದಶಮಾಂಶ    m0;
    ದಶಮಾಂಶ    m1;
    int32_t  n_ತಲೆ_log2;
} ggml_metal_kargs_soft_max;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
} ggml_metal_kargs_ssm_conv;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  d_state;
    int64_t  d_inner;
    int64_t  n_ತಲೆ;
    int64_t  n_ಗುಂಪು;
    int64_t  n_seq_ಟೋಕನುಗಳು;
    int64_t  n_seq_ಟೋಕನುಗಳು_ಒಟ್ಟು;
    int64_t  token_ಸ್ಥಾನಾಂತರ;
    int64_t  n_seqs;
    int64_t  K;
    uint64_t s_off;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t ns12;
    uint64_t nb13;
    uint64_t nb20;
    uint64_t nb21;
    uint64_t ns21;
    uint64_t nb22;
    int64_t  ne30;
    uint64_t nb31;
    uint64_t nb41;
    uint64_t nb42;
    uint64_t ns42;
    uint64_t nb43;
    uint64_t nb51;
    uint64_t nb52;
    uint64_t ns52;
    uint64_t nb53;
    uint64_t nb0;
} ggml_metal_kargs_ssm_ಪರಿಶೀಲನೆ;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne22;
    int32_t  ne23;
    uint64_t nb20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ns02;
    int32_t  ns12;
    int32_t  ns22;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t nb_ಹೊರಹೋಗು; // 0 => snapshots are appended after the attn scores (unfused)
} ggml_metal_kargs_gated_delta_net;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_solve_tri;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00t;
    int32_t  ne00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_get_ಅಡ್ಡಸಾಲುಗಳು;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  nk0;
    int32_t  ne01;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_diag;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    ದಶಮಾಂಶ    sf0;
    ದಶಮಾಂಶ    sf1;
    ದಶಮಾಂಶ    sf2;
    ದಶಮಾಂಶ    sf3;
    ದಶಮಾಂಶ    poffs;
} ggml_metal_kargs_upscale;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  lp0;
    int32_t  lp1;
    int32_t  lp2;
    int32_t  lp3;
} ggml_metal_kargs_pad;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  p0;
    int32_t  p1;
} ggml_metal_kargs_pad_reflect_1d;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  s0;
    int32_t  s1;
    int32_t  s2;
    int32_t  s3;
} ggml_metal_kargs_roll;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    uint64_t nb1;
    ಪೂರ್ಣಾಂಕ      dim;
    ಪೂರ್ಣಾಂಕ      max_period;
} ggml_metal_kargs_timestep_embedding;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_tri;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  ತುದಿ_k;
} ggml_metal_kargs_argsort;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  ತುದಿ_k;
    int32_t  len;
} ggml_metal_kargs_argsort_ವಿಲೀನ;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;   // number of columns (elements per row)
    int32_t  ne01;   //ಅಡ್ಡಸಾಲುಗಳು
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;   // row stride in src0
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ತುದಿ_k;  // k
} ggml_metal_kargs_ತುದಿ_k;

// widths at or above this use the threadgroup FWHT kernel, one row per threadgroup
// with GGML_METAL_FWHT_TG_NT threads, instead of one row per simdgroup
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_METAL_FWHT_TG_MIN_N 512
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_METAL_FWHT_TG_NT    256

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne01;      // n_tokens
    uint64_t nb01;      // logits row stride
    uint64_t nb1_ids;   // ids row stride
    ದಶಮಾಂಶ    clamp;
    ದಶಮಾಂಶ    scale;
} ggml_metal_kargs_topk_moe;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t ne00; // n_embd
    int32_t ne02; // n_tokens
} ggml_metal_kargs_moe_reduce;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t nrows;
} ggml_metal_kargs_fwht;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  ne0;
    ದಶಮಾಂಶ    ಆರಂಭ;
    ದಶಮಾಂಶ    ಹೆಜ್ಜೆ;
} ggml_metal_kargs_arange;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t val;
} ggml_metal_kargs_memset;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  n_kv;
    int32_t  n_batch;
    int32_t  ಮುಖವಾಡ_ne3;
    uint64_t nb1;
    uint64_t nb3;
    uint64_t nbq1;
    uint64_t nbq2;
    uint64_t nbq3;
    uint64_t nbk2;
    uint64_t nbk3;
    uint64_t nbw1;
    uint64_t nbw3;
    uint64_t nbm1;
    uint64_t nbm3;
} ggml_metal_kargs_lightning_indexer;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  n_ಟೋಕನುಗಳು;
    int32_t  n_iter;
    uint64_t nb_m0;
    uint64_t nb_m1;
    uint64_t nb_s0;
    uint64_t nb_b0;
    uint64_t nb_d0;
    uint64_t nb_d1;
    uint64_t nb_d2;
    ದಶಮಾಂಶ    eps;
} ggml_metal_kargs_dsv4_hc_comb;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  n_embd;
    int32_t  n_ಟೋಕನುಗಳು;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb_w2;
    uint64_t nb_d0;
    uint64_t nb_d1;
    ದಶಮಾಂಶ    scale;
} ggml_metal_kargs_dsv4_hc_pre;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  n_embd;
    int32_t  n_ಟೋಕನುಗಳು;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_r0;
    uint64_t nb_r1;
    uint64_t nb_r2;
    uint64_t nb_p0;
    uint64_t nb_p1;
    uint64_t nb_c0;
    uint64_t nb_c1;
    uint64_t nb_c2;
    uint64_t nb_d0;
    uint64_t nb_d1;
    uint64_t nb_d2;
} ggml_metal_kargs_dsv4_hc_post;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
} ggml_metal_kargs_ಎಣಿಕೆ_equal;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  k0;
    int32_t  k1;
    int32_t  s0;
    int32_t  s1;
    int32_t  p0;
    int32_t  p1;
    int64_t  IH;
    int64_t  IW;
    int64_t  OH;
    int64_t  OW;
    int64_t  np;
} ggml_metal_kargs_pool_2d;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t  k0;
    int32_t  s0;
    int32_t  p0;
    int64_t  IW;
    int64_t  OW;
    int64_t  np;
} ggml_metal_kargs_pool_1d;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t ne00;
    uint64_t nb01;
    int64_t npart;
    int64_t ne01;
} ggml_metal_kargs_argmax;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  np;
} ggml_metal_kargs_opt_ಹೆಜ್ಜೆ_adamw;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t  np;
} ggml_metal_kargs_opt_ಹೆಜ್ಜೆ_sgd;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int64_t ne;
} ggml_metal_kargs_silu_ಹಿಂಭಾಗ;

#ಆದರೆ_ಅಂತ್ಯ // GGML_METAL_IMPL
