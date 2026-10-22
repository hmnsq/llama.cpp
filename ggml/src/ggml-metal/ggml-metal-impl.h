#که_نه_تعریف_شوی GGML_METAL_IMPL
#تعریف_کړه GGML_METAL_IMPL

// kernel parameters for mat-mat threadgroups
//
// TODO: become function constants

#تعریف_کړه SZ_SIMDGROUP 16
#تعریف_کړه N_MM_NK 2
#تعریف_کړه N_MM_NK_ټول (SZ_SIMDGROUP * N_MM_NK)

#تعریف_کړه N_MM_بلاک_X 4
#تعریف_کړه N_MM_بلاک_Y 2
#تعریف_کړه N_MM_SIMD_ډله_X 2
#تعریف_کړه N_MM_SIMD_ډله_Y 2

#تعریف_کړه N_MM_NPART_AMAX 256

#تعریف_کړه N_ARGMAX_NPART_MAX 256
#تعریف_کړه N_ARGMAX_CHUNK 4096

// kernel parameters for mat-vec threadgroups
//
// N_R0: number of src0 rows to process per simdgroup
// N_SG: number of simdgroups per threadgroup
//
// TODO: for optimal performance, become function of the device and work size

#تعریف_کړه N_R0_Q1_0 8
#تعریف_کړه N_SG_Q1_0 2

#تعریف_کړه N_R0_Q2_0 8
#تعریف_کړه N_SG_Q2_0 2

#تعریف_کړه N_R0_Q4_0 4
#تعریف_کړه N_SG_Q4_0 2

#تعریف_کړه N_R0_Q4_1 4
#تعریف_کړه N_SG_Q4_1 2

#تعریف_کړه N_R0_Q5_0 4
#تعریف_کړه N_SG_Q5_0 2

#تعریف_کړه N_R0_Q5_1 4
#تعریف_کړه N_SG_Q5_1 2

#تعریف_کړه N_R0_Q8_0 2
#تعریف_کړه N_SG_Q8_0 4

#تعریف_کړه N_R0_MXFP4 2
#تعریف_کړه N_SG_MXFP4 2

#تعریف_کړه N_R0_Q2_K 4
#تعریف_کړه N_SG_Q2_K 2

#تعریف_کړه N_R0_Q3_K 2
#تعریف_کړه N_SG_Q3_K 2

#تعریف_کړه N_R0_Q4_K 2
#تعریف_کړه N_SG_Q4_K 2

#تعریف_کړه N_R0_Q5_K 1
#تعریف_کړه N_SG_Q5_K 2

#تعریف_کړه N_R0_Q6_K 2
#تعریف_کړه N_SG_Q6_K 2

#تعریف_کړه N_R0_IQ1_S 4
#تعریف_کړه N_SG_IQ1_S 2
#تعریف_کړه N_R0_IQ1_S_وېشل 8

#تعریف_کړه N_R0_IQ1_M 4
#تعریف_کړه N_SG_IQ1_M 2
#تعریف_کړه N_R0_IQ1_M_وېشل 8

#تعریف_کړه N_R0_IQ2_XXS 4
#تعریف_کړه N_SG_IQ2_XXS 2
#تعریف_کړه N_R0_IQ2_XXS_وېشل 8

#تعریف_کړه N_R0_IQ2_XS 4
#تعریف_کړه N_SG_IQ2_XS 2
#تعریف_کړه N_R0_IQ2_XS_وېشل 8

#تعریف_کړه N_R0_IQ2_S 4
#تعریف_کړه N_SG_IQ2_S 2
#تعریف_کړه N_R0_IQ2_S_وېشل 8

#تعریف_کړه N_R0_IQ3_XXS 4
#تعریف_کړه N_SG_IQ3_XXS 2
#تعریف_کړه N_R0_IQ3_XXS_وېشل 8

#تعریف_کړه N_R0_IQ3_S 4
#تعریف_کړه N_SG_IQ3_S 2
#تعریف_کړه N_R0_IQ3_S_وېشل 8

#تعریف_کړه N_R0_IQ4_NL 2
#تعریف_کړه N_SG_IQ4_NL 2

#تعریف_کړه N_R0_IQ4_XS 2
#تعریف_کړه N_SG_IQ4_XS 2

#تعریف_کړه N_R0_TQ2_0 4
#تعریف_کړه N_SG_TQ2_0 2

// function constants offsets
#تعریف_کړه FC_FLASH_ATTN_EXT_PAD          100
#تعریف_کړه FC_FLASH_ATTN_EXT_BLK          200
#تعریف_کړه FC_FLASH_ATTN_EXT              300
#تعریف_کړه FC_FLASH_ATTN_EXT_VEC          400
#تعریف_کړه FC_FLASH_ATTN_EXT_VEC_REDUCE   500
#تعریف_کړه FC_MUL_MV                      600
#تعریف_کړه FC_MUL_MM                      700
#تعریف_کړه FC_ROPE                        800
#تعریف_کړه FC_SSM_CONV                    900
#تعریف_کړه FC_SOLVE_TRI                   1000
#تعریف_کړه FC_شمېر_EQUAL                 1100
#تعریف_کړه FC_UNARY                       1200
#تعریف_کړه FC_BIN                         1300
#تعریف_کړه FC_مجموعه_لیکې                    1400
#تعریف_کړه FC_UPSCALE                     1500
#تعریف_کړه FC_GATED_DELTA_NET             1600
#تعریف_کړه FC_NORM                        1700
#تعریف_کړه FC_TOPK_MOE                    1800
#تعریف_کړه FC_MOE_REDUCE                  1900
#تعریف_کړه FC_DSV4_HC                     2000
#تعریف_کړه FC_PAD                         2100
#تعریف_کړه FC_FLASH_ATTN_EXT_TENSOR       2200
#تعریف_کړه FC_LIGHTNING_INDEXER           2200
#تعریف_کړه FC_MUL_MV_MMA                  2300

// op-specific constants
#تعریف_کړه OP_FLASH_ATTN_EXT_NQPSG 8
#تعریف_کړه OP_FLASH_ATTN_EXT_NCPSG 64

#تعریف_کړه OP_FLASH_ATTN_EXT_TENSOR_NQPSG       32
#تعریف_کړه OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE 16
#تعریف_کړه OP_FLASH_ATTN_EXT_TENSOR_NCPSG       64
#تعریف_کړه OP_FLASH_ATTN_EXT_TENSOR_NSG         8

#تعریف_کړه OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#تعریف_کړه OP_FLASH_ATTN_EXT_VEC_NCPSG 32

#تعریف_کړه OP_LIGHTNING_INDEXER_DK    128
#تعریف_کړه OP_LIGHTNING_INDEXER_NHPTG   8
#تعریف_کړه OP_LIGHTNING_INDEXER_NKPSG   8
#تعریف_کړه OP_LIGHTNING_INDEXER_NSG     8
#تعریف_کړه OP_LIGHTNING_INDEXER_NBPTG   8

#تعریف_کړه OP_UNARY_عدد_SCALE      10
#تعریف_کړه OP_UNARY_عدد_FILL       11
#تعریف_کړه OP_UNARY_عدد_CLAMP      12
#تعریف_کړه OP_UNARY_عدد_SQR        13
#تعریف_کړه OP_UNARY_عدد_SQRT       14
#تعریف_کړه OP_UNARY_عدد_SIN        15
#تعریف_کړه OP_UNARY_عدد_COS        16
#تعریف_کړه OP_UNARY_عدد_LOG        17
#تعریف_کړه OP_UNARY_عدد_LEAKY_RELU 18

#تعریف_کړه OP_UNARY_عدد_TANH        100
#تعریف_کړه OP_UNARY_عدد_RELU        101
#تعریف_کړه OP_UNARY_عدد_SIGMOID     102
#تعریف_کړه OP_UNARY_عدد_GELU        103
#تعریف_کړه OP_UNARY_عدد_GELU_ERF    104
#تعریف_کړه OP_UNARY_عدد_GELU_چټک  105
#تعریف_کړه OP_UNARY_عدد_SILU        106
#تعریف_کړه OP_UNARY_عدد_ELU         107
#تعریف_کړه OP_UNARY_عدد_NEG         108
#تعریف_کړه OP_UNARY_عدد_ABS         109
#تعریف_کړه OP_UNARY_عدد_SGN         110
#تعریف_کړه OP_UNARY_عدد_ګام        111
#تعریف_کړه OP_UNARY_عدد_HARDSWISH   112
#تعریف_کړه OP_UNARY_عدد_HARDSIGMOID 113
#تعریف_کړه OP_UNARY_عدد_EXP         114
#تعریف_کړه OP_UNARY_عدد_SOFTPLUS    115
#تعریف_کړه OP_UNARY_عدد_EXPM1       116
#تعریف_کړه OP_UNARY_عدد_FLOOR       117
#تعریف_کړه OP_UNARY_عدد_CEIL        118
#تعریف_کړه OP_UNARY_عدد_پړاو       119
#تعریف_کړه OP_UNARY_عدد_TRUNC       120
#تعریف_کړه OP_UNARY_عدد_XIELU       121

#تعریف_کړه OP_مجموعه_لیکې_عدد_مجموعه_لیکې 10
#تعریف_کړه OP_مجموعه_لیکې_عدد_منځنۍ     11

#تعریف_کړه OP_SSM_لټون_SSD_CS  64 // Metal-specific; Chunk Size; 64 is largest multiple of 8 (simdgroup tile) fitting into 32 KiB Metal threadgroup mem limit (~26.75 KiB shared mem; see smem layout comment in kernel_ssm_scan_ssd_mma_f32)
#تعریف_کړه OP_SSM_لټون_SSD_HD  64 // Metal-specific; Head Dim the MMA kernel is specialized for (Mamba-2); use_mma gates on d_inner == this
#تعریف_کړه OP_SSM_لټون_SSD_NSG 4  // Metal-specific; Number of SimdGroups per threadgroup; NSG*32 == threads dispatched per threadgroup

// kernel argument structs
//
// - element counters (e.g. ne00) typically use int32_t to reduce register usage
//   however, be careful from int overflows when using those in the kernel implementation
//
// - strides (e.g. nb00) use uint64_t

تعريفنوع بنية {
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

تعريفنوع بنية {
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
    عائم    میلان;
    عائم    scale;
    عائم    bias;
    عائم    val;
    عائم    min;
    عائم    max;
} ggml_metal_kargs_unary;

تعريفنوع بنية {
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

تعريفنوع بنية {
    int64_t ne0;
    int64_t ne1;
    size_t nb01;
    size_t nb02;
    size_t nb11;
    size_t nb21;
} ggml_metal_kargs_جمع_کړه_id;

تعريفنوع بنية {
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

تعريفنوع بنية {
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

تعريفنوع بنية {
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
} ggml_metal_kargs_سېټ;

تعريفنوع بنية {
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
    عائم    freq_بنسټ;
    عائم    freq_scale;
    عائم    ext_ضریب;
    عائم    attn_ضریب;
    عائم    beta_fast;
    عائم    beta_slow;
    int32_t  sect_0;
    int32_t  sect_1;
    int32_t  sect_2;
    int32_t  sect_3;
    bool     src2;
    bool     inplace;
} ggml_metal_kargs_rope;

تعريفنوع بنية {
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

تعريفنوع بنية {
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

تعريفنوع بنية {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ggml_metal_kargs_flash_attn_ext_blk;

تعريفنوع بنية {
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
    عائم    scale;
    عائم    max_bias;
    عائم    m0;
    عائم    m1;
    int32_t  n_سر_log2;
    عائم    logit_softcap;
} ggml_metal_kargs_flash_attn_ext;

تعريفنوع بنية {
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
    عائم    scale;
    عائم    max_bias;
    عائم    m0;
    عائم    m1;
    int32_t  n_سر_log2;
    عائم    logit_softcap;
    int32_t  n_kv_max_padded;
} ggml_metal_kargs_flash_attn_ext_vec;

تعريفنوع بنية {
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

تعريفنوع بنية {
    int32_t  nrows;
} ggml_metal_kargs_flash_attn_ext_vec_reduce;

تعريفنوع بنية {
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

تعريفنوع بنية {
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

تعريفنوع بنية {
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

تعريفنوع بنية {
    int32_t  ne10;
    int32_t  ne11;  // n_expert_used (bcast)
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21; // n_tokens
    int32_t  ne20;  // n_expert_used
    uint64_t nb21;
} ggml_metal_kargs_mul_mm_id_map0;

تعريفنوع بنية {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
} ggml_metal_kargs_mul_mm_id_amax;

تعريفنوع بنية {
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

تعريفنوع بنية {
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
تعريفنوع بنية {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    عائم    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
    عائم    scale;
} ggml_metal_kargs_norm;

تعريفنوع بنية {
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
    عائم    eps;
} ggml_metal_kargs_l2_norm;

تعريفنوع بنية {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ngrp;
    عائم    eps;
} ggml_metal_kargs_ډله_norm;

تعريفنوع بنية {
    int32_t  IC;
    int32_t  IL;
    int32_t  K;
    int32_t  s0;
    uint64_t nb0;
    uint64_t nb1;
} ggml_metal_kargs_conv_transpose_1d;

تعريفنوع بنية {
    int32_t  T_دننه;
    int32_t  T_بهر;
    int32_t  OC;
    int32_t  K;
    int32_t  K_OC;
    int32_t  s0;
    int32_t  p0;
} ggml_metal_kargs_col2im_1d;

تعريفنوع بنية {
    int32_t T;
    int32_t C;
} ggml_metal_kargs_snake;

تعريفنوع بنية {
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

تعريفنوع بنية {
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

تعريفنوع بنية {
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

تعريفنوع بنية {
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

تعريفنوع بنية {
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

تعريفنوع بنية{
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    عائم    alpha;
    عائم    حد;
} ggml_metal_kargs_glu;

تعريفنوع بنية {
    uint64_t np;
} ggml_metal_kargs_مجموعه;

تعريفنوع بنية {
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
} ggml_metal_kargs_مجموعه_لیکې;

تعريفنوع بنية {
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

تعريفنوع بنية {
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
} ggml_metal_kargs_cumsum_جمع_کړه;

تعريفنوع بنية {
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
    عائم    scale;
    عائم    max_bias;
    عائم    m0;
    عائم    m1;
    int32_t  n_سر_log2;
} ggml_metal_kargs_soft_max;

تعريفنوع بنية {
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

تعريفنوع بنية {
    int64_t  d_state;
    int64_t  d_inner;
    int64_t  n_سر;
    int64_t  n_ډله;
    int64_t  n_seq_نښې;
    int64_t  n_seq_نښې_ټول;
    int64_t  token_دننه_واټن;
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
} ggml_metal_kargs_ssm_لټون;

تعريفنوع بنية {
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
    uint64_t nb_بهر; // 0 => snapshots are appended after the attn scores (unfused)
} ggml_metal_kargs_gated_delta_net;

تعريفنوع بنية {
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

تعريفنوع بنية {
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
} ggml_metal_kargs_get_لیکې;

تعريفنوع بنية {
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
} ggml_metal_kargs_سېټ_لیکې;

تعريفنوع بنية {
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

تعريفنوع بنية {
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
    عائم    sf0;
    عائم    sf1;
    عائم    sf2;
    عائم    sf3;
    عائم    poffs;
} ggml_metal_kargs_upscale;

تعريفنوع بنية {
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

تعريفنوع بنية {
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

تعريفنوع بنية {
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

تعريفنوع بنية {
    uint64_t nb1;
    صحيح      dim;
    صحيح      max_period;
} ggml_metal_kargs_timestep_embedding;

تعريفنوع بنية {
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

تعريفنوع بنية {
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
    int32_t  سر_k;
} ggml_metal_kargs_argsort;

تعريفنوع بنية {
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
    int32_t  سر_k;
    int32_t  len;
} ggml_metal_kargs_argsort_یوځای;

تعريفنوع بنية {
    int32_t  ne00;   // number of columns (elements per row)
    int32_t  ne01;   //لیکې
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;   // row stride in src0
    uint64_t nb02;
    uint64_t nb03;
    int32_t  سر_k;  // k
} ggml_metal_kargs_سر_k;

// widths at or above this use the threadgroup FWHT kernel, one row per threadgroup
// with GGML_METAL_FWHT_TG_NT threads, instead of one row per simdgroup
#تعریف_کړه GGML_METAL_FWHT_TG_MIN_N 512
#تعریف_کړه GGML_METAL_FWHT_TG_NT    256

تعريفنوع بنية {
    int32_t  ne01;      // n_tokens
    uint64_t nb01;      // logits row stride
    uint64_t nb1_ids;   // ids row stride
    عائم    clamp;
    عائم    scale;
} ggml_metal_kargs_topk_moe;

تعريفنوع بنية {
    int32_t ne00; // n_embd
    int32_t ne02; // n_tokens
} ggml_metal_kargs_moe_reduce;

تعريفنوع بنية {
    int32_t nrows;
} ggml_metal_kargs_fwht;

تعريفنوع بنية {
    int64_t  ne0;
    عائم    پیل;
    عائم    ګام;
} ggml_metal_kargs_arange;

تعريفنوع بنية {
    int64_t val;
} ggml_metal_kargs_memset;

تعريفنوع بنية {
    int32_t  n_kv;
    int32_t  n_batch;
    int32_t  ماسک_ne3;
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

تعريفنوع بنية {
    int32_t  n_نښې;
    int32_t  n_iter;
    uint64_t nb_m0;
    uint64_t nb_m1;
    uint64_t nb_s0;
    uint64_t nb_b0;
    uint64_t nb_d0;
    uint64_t nb_d1;
    uint64_t nb_d2;
    عائم    eps;
} ggml_metal_kargs_dsv4_hc_comb;

تعريفنوع بنية {
    int32_t  n_embd;
    int32_t  n_نښې;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb_w2;
    uint64_t nb_d0;
    uint64_t nb_d1;
    عائم    scale;
} ggml_metal_kargs_dsv4_hc_pre;

تعريفنوع بنية {
    int32_t  n_embd;
    int32_t  n_نښې;
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

تعريفنوع بنية {
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
} ggml_metal_kargs_شمېر_equal;

تعريفنوع بنية {
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

تعريفنوع بنية {
    int32_t  k0;
    int32_t  s0;
    int32_t  p0;
    int64_t  IW;
    int64_t  OW;
    int64_t  np;
} ggml_metal_kargs_pool_1d;

تعريفنوع بنية {
    int64_t ne00;
    uint64_t nb01;
    int64_t npart;
    int64_t ne01;
} ggml_metal_kargs_argmax;

تعريفنوع بنية {
    int64_t  np;
} ggml_metal_kargs_opt_ګام_adamw;

تعريفنوع بنية {
    int64_t  np;
} ggml_metal_kargs_opt_ګام_sgd;

تعريفنوع بنية {
    int64_t ne;
} ggml_metal_kargs_silu_شا;

#د_که_پای // GGML_METAL_IMPL
