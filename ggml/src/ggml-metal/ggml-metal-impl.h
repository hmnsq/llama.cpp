#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_METAL_IMPL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_METAL_IMPL

// kernel parameters for mat-mat threadgroups
//
// TODO: become function constants

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SZ_SIMDGROUP 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_MM_NK 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_MM_NK_ਕੁੱਲ (SZ_SIMDGROUP * N_MM_NK)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_MM_ਖੰਡ_X 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_MM_ਖੰਡ_Y 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_MM_SIMD_ਸਮੂਹ_X 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_MM_SIMD_ਸਮੂਹ_Y 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_MM_NPART_AMAX 256

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_ARGMAX_NPART_MAX 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_ARGMAX_CHUNK 4096

// kernel parameters for mat-vec threadgroups
//
// N_R0: number of src0 rows to process per simdgroup
// N_SG: number of simdgroups per threadgroup
//
// TODO: for optimal performance, become function of the device and work size

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q1_0 8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q1_0 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q2_0 8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q2_0 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q4_0 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q4_0 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q4_1 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q4_1 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q5_0 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q5_0 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q5_1 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q5_1 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q8_0 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q8_0 4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_MXFP4 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_MXFP4 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q2_K 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q2_K 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q3_K 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q3_K 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q4_K 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q4_K 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q5_K 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q5_K 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_Q6_K 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_Q6_K 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ1_S 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_IQ1_S 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ1_S_ਵੰਡੋ 8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ1_M 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_IQ1_M 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ1_M_ਵੰਡੋ 8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ2_XXS 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_IQ2_XXS 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ2_XXS_ਵੰਡੋ 8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ2_XS 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_IQ2_XS 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ2_XS_ਵੰਡੋ 8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ2_S 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_IQ2_S 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ2_S_ਵੰਡੋ 8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ3_XXS 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_IQ3_XXS 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ3_XXS_ਵੰਡੋ 8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ3_S 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_IQ3_S 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ3_S_ਵੰਡੋ 8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ4_NL 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_IQ4_NL 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_IQ4_XS 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_IQ4_XS 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_R0_TQ2_0 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SG_TQ2_0 2

// function constants offsets
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_FLASH_ATTN_EXT_PAD          100
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_FLASH_ATTN_EXT_BLK          200
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_FLASH_ATTN_EXT              300
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_FLASH_ATTN_EXT_VEC          400
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_FLASH_ATTN_EXT_VEC_REDUCE   500
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_MUL_MV                      600
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_MUL_MM                      700
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_ROPE                        800
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_SSM_CONV                    900
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_SOLVE_TRI                   1000
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_ਗਿਣਤੀ_EQUAL                 1100
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_UNARY                       1200
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_BIN                         1300
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_ਜੋੜ_ਪੰਕਤੀਆਂ                    1400
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_UPSCALE                     1500
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_GATED_DELTA_NET             1600
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_NORM                        1700
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_TOPK_MOE                    1800
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_MOE_REDUCE                  1900
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_DSV4_HC                     2000
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_PAD                         2100
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_FLASH_ATTN_EXT_TENSOR       2200
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_LIGHTNING_INDEXER           2200
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FC_MUL_MV_MMA                  2300

// op-specific constants
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_FLASH_ATTN_EXT_NQPSG 8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_FLASH_ATTN_EXT_NCPSG 64

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_FLASH_ATTN_EXT_TENSOR_NQPSG       32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_FLASH_ATTN_EXT_TENSOR_NCPSG       64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_FLASH_ATTN_EXT_TENSOR_NSG         8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_FLASH_ATTN_EXT_VEC_NCPSG 32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_LIGHTNING_INDEXER_DK    128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_LIGHTNING_INDEXER_NHPTG   8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_LIGHTNING_INDEXER_NKPSG   8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_LIGHTNING_INDEXER_NSG     8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_LIGHTNING_INDEXER_NBPTG   8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_SCALE      10
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_FILL       11
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_CLAMP      12
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_SQR        13
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_SQRT       14
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_SIN        15
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_COS        16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_LOG        17
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_LEAKY_RELU 18

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_TANH        100
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_RELU        101
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_SIGMOID     102
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_GELU        103
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_GELU_ERF    104
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_GELU_ਤੇਜ਼  105
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_SILU        106
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_ELU         107
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_NEG         108
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_ABS         109
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_SGN         110
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_ਕਦਮ        111
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_HARDSWISH   112
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_HARDSIGMOID 113
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_EXP         114
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_SOFTPLUS    115
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_EXPM1       116
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_FLOOR       117
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_CEIL        118
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_ਗੇੜਾ       119
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_TRUNC       120
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_UNARY_ਅੰਕ_XIELU       121

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_ਜੋੜ_ਪੰਕਤੀਆਂ_ਅੰਕ_ਜੋੜ_ਪੰਕਤੀਆਂ 10
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_ਜੋੜ_ਪੰਕਤੀਆਂ_ਅੰਕ_ਔਸਤ     11

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_SSM_ਜਾਂਚ_SSD_CS  64 // Metal-specific; Chunk Size; 64 is largest multiple of 8 (simdgroup tile) fitting into 32 KiB Metal threadgroup mem limit (~26.75 KiB shared mem; see smem layout comment in kernel_ssm_scan_ssd_mma_f32)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_SSM_ਜਾਂਚ_SSD_HD  64 // Metal-specific; Head Dim the MMA kernel is specialized for (Mamba-2); use_mma gates on d_inner == this
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OP_SSM_ਜਾਂਚ_SSD_NSG 4  // Metal-specific; Number of SimdGroups per threadgroup; NSG*32 == threads dispatched per threadgroup

// kernel argument structs
//
// - element counters (e.g. ne00) typically use int32_t to reduce register usage
//   however, be careful from int overflows when using those in the kernel implementation
//
// - strides (e.g. nb00) use uint64_t

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
    ਦਸ਼ਮਲਵ    ਢਲਾਨ;
    ਦਸ਼ਮਲਵ    scale;
    ਦਸ਼ਮਲਵ    bias;
    ਦਸ਼ਮਲਵ    val;
    ਦਸ਼ਮਲਵ    min;
    ਦਸ਼ਮਲਵ    max;
} ggml_metal_kargs_unary;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int64_t ne0;
    int64_t ne1;
    size_t nb01;
    size_t nb02;
    size_t nb11;
    size_t nb21;
} ggml_metal_kargs_ਜੋੜੋ_id;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
} ggml_metal_kargs_ਸੈੱਟ;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
    ਦਸ਼ਮਲਵ    freq_ਅਧਾਰ;
    ਦਸ਼ਮਲਵ    freq_scale;
    ਦਸ਼ਮਲਵ    ext_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ    attn_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ    beta_fast;
    ਦਸ਼ਮਲਵ    beta_slow;
    int32_t  sect_0;
    int32_t  sect_1;
    int32_t  sect_2;
    int32_t  sect_3;
    bool     src2;
    bool     inplace;
} ggml_metal_kargs_rope;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ggml_metal_kargs_flash_attn_ext_blk;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
    ਦਸ਼ਮਲਵ    scale;
    ਦਸ਼ਮਲਵ    max_bias;
    ਦਸ਼ਮਲਵ    m0;
    ਦਸ਼ਮਲਵ    m1;
    int32_t  n_ਸਿਰ_log2;
    ਦਸ਼ਮਲਵ    logit_softcap;
} ggml_metal_kargs_flash_attn_ext;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
    ਦਸ਼ਮਲਵ    scale;
    ਦਸ਼ਮਲਵ    max_bias;
    ਦਸ਼ਮਲਵ    m0;
    ਦਸ਼ਮਲਵ    m1;
    int32_t  n_ਸਿਰ_log2;
    ਦਸ਼ਮਲਵ    logit_softcap;
    int32_t  n_kv_max_padded;
} ggml_metal_kargs_flash_attn_ext_vec;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  nrows;
} ggml_metal_kargs_flash_attn_ext_vec_reduce;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  ne10;
    int32_t  ne11;  // n_expert_used (bcast)
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21; // n_tokens
    int32_t  ne20;  // n_expert_used
    uint64_t nb21;
} ggml_metal_kargs_mul_mm_id_map0;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
} ggml_metal_kargs_mul_mm_id_amax;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    ਦਸ਼ਮਲਵ    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
    ਦਸ਼ਮਲਵ    scale;
} ggml_metal_kargs_norm;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
    ਦਸ਼ਮਲਵ    eps;
} ggml_metal_kargs_l2_norm;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ngrp;
    ਦਸ਼ਮਲਵ    eps;
} ggml_metal_kargs_ਸਮੂਹ_norm;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  IC;
    int32_t  IL;
    int32_t  K;
    int32_t  s0;
    uint64_t nb0;
    uint64_t nb1;
} ggml_metal_kargs_conv_transpose_1d;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  T_ਵਿੱਚ;
    int32_t  T_ਬਾਹਰਲਾ;
    int32_t  OC;
    int32_t  K;
    int32_t  K_OC;
    int32_t  s0;
    int32_t  p0;
} ggml_metal_kargs_col2im_1d;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t T;
    int32_t C;
} ggml_metal_kargs_snake;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ{
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    ਦਸ਼ਮਲਵ    alpha;
    ਦਸ਼ਮਲਵ    ਹੱਦਬੰਦੀ;
} ggml_metal_kargs_glu;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    uint64_t np;
} ggml_metal_kargs_ਜੋੜ;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
} ggml_metal_kargs_ਜੋੜ_ਪੰਕਤੀਆਂ;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
} ggml_metal_kargs_cumsum_ਜੋੜੋ;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
    ਦਸ਼ਮਲਵ    scale;
    ਦਸ਼ਮਲਵ    max_bias;
    ਦਸ਼ਮਲਵ    m0;
    ਦਸ਼ਮਲਵ    m1;
    int32_t  n_ਸਿਰ_log2;
} ggml_metal_kargs_soft_max;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int64_t  d_state;
    int64_t  d_inner;
    int64_t  n_ਸਿਰ;
    int64_t  n_ਸਮੂਹ;
    int64_t  n_seq_ਟੋਕਨ;
    int64_t  n_seq_ਟੋਕਨ_ਕੁੱਲ;
    int64_t  token_ਵਿੱਥ;
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
} ggml_metal_kargs_ssm_ਜਾਂਚ;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
    uint64_t nb_ਬਾਹਰਲਾ; // 0 => snapshots are appended after the attn scores (unfused)
} ggml_metal_kargs_gated_delta_net;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
} ggml_metal_kargs_get_ਪੰਕਤੀਆਂ;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
} ggml_metal_kargs_ਸੈੱਟ_ਪੰਕਤੀਆਂ;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
    ਦਸ਼ਮਲਵ    sf0;
    ਦਸ਼ਮਲਵ    sf1;
    ਦਸ਼ਮਲਵ    sf2;
    ਦਸ਼ਮਲਵ    sf3;
    ਦਸ਼ਮਲਵ    poffs;
} ggml_metal_kargs_upscale;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    uint64_t nb1;
    ਪੂਰਨ      dim;
    ਪੂਰਨ      max_period;
} ggml_metal_kargs_timestep_embedding;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
    int32_t  ਸਿਖਰ_k;
} ggml_metal_kargs_argsort;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
    int32_t  ਸਿਖਰ_k;
    int32_t  len;
} ggml_metal_kargs_argsort_ਮਿਲਾਓ;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  ne00;   // number of columns (elements per row)
    int32_t  ne01;   //ਪੰਕਤੀਆਂ
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;   // row stride in src0
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ਸਿਖਰ_k;  // k
} ggml_metal_kargs_ਸਿਖਰ_k;

// widths at or above this use the threadgroup FWHT kernel, one row per threadgroup
// with GGML_METAL_FWHT_TG_NT threads, instead of one row per simdgroup
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_METAL_FWHT_TG_MIN_N 512
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_METAL_FWHT_TG_NT    256

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  ne01;      // n_tokens
    uint64_t nb01;      // logits row stride
    uint64_t nb1_ids;   // ids row stride
    ਦਸ਼ਮਲਵ    clamp;
    ਦਸ਼ਮਲਵ    scale;
} ggml_metal_kargs_topk_moe;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t ne00; // n_embd
    int32_t ne02; // n_tokens
} ggml_metal_kargs_moe_reduce;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t nrows;
} ggml_metal_kargs_fwht;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int64_t  ne0;
    ਦਸ਼ਮਲਵ    ਸ਼ੁਰੂ;
    ਦਸ਼ਮਲਵ    ਕਦਮ;
} ggml_metal_kargs_arange;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int64_t val;
} ggml_metal_kargs_memset;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  n_kv;
    int32_t  n_batch;
    int32_t  ਨਕਾਬ_ne3;
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  n_ਟੋਕਨ;
    int32_t  n_iter;
    uint64_t nb_m0;
    uint64_t nb_m1;
    uint64_t nb_s0;
    uint64_t nb_b0;
    uint64_t nb_d0;
    uint64_t nb_d1;
    uint64_t nb_d2;
    ਦਸ਼ਮਲਵ    eps;
} ggml_metal_kargs_dsv4_hc_comb;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  n_embd;
    int32_t  n_ਟੋਕਨ;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb_w2;
    uint64_t nb_d0;
    uint64_t nb_d1;
    ਦਸ਼ਮਲਵ    scale;
} ggml_metal_kargs_dsv4_hc_pre;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  n_embd;
    int32_t  n_ਟੋਕਨ;
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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
} ggml_metal_kargs_ਗਿਣਤੀ_equal;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t  k0;
    int32_t  s0;
    int32_t  p0;
    int64_t  IW;
    int64_t  OW;
    int64_t  np;
} ggml_metal_kargs_pool_1d;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int64_t ne00;
    uint64_t nb01;
    int64_t npart;
    int64_t ne01;
} ggml_metal_kargs_argmax;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int64_t  np;
} ggml_metal_kargs_opt_ਕਦਮ_adamw;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int64_t  np;
} ggml_metal_kargs_opt_ਕਦਮ_sgd;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int64_t ne;
} ggml_metal_kargs_silu_ਪਿੱਠ;

#ਜੇ_ਅੰਤ // GGML_METAL_IMPL
