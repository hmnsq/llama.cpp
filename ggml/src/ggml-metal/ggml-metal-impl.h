#నిర్వచితం_కాకపోతే GGML_METAL_IMPL
#నిర్వచించు GGML_METAL_IMPL

// kernel parameters for mat-mat threadgroups
//
// TODO: become function constants

#నిర్వచించు SZ_SIMDGROUP 16
#నిర్వచించు N_MM_NK 2
#నిర్వచించు N_MM_NK_మొత్తం (SZ_SIMDGROUP * N_MM_NK)

#నిర్వచించు N_MM_ఖండం_X 4
#నిర్వచించు N_MM_ఖండం_Y 2
#నిర్వచించు N_MM_SIMD_సమూహం_X 2
#నిర్వచించు N_MM_SIMD_సమూహం_Y 2

#నిర్వచించు N_MM_NPART_AMAX 256

#నిర్వచించు N_ARGMAX_NPART_MAX 256
#నిర్వచించు N_ARGMAX_CHUNK 4096

// kernel parameters for mat-vec threadgroups
//
// N_R0: number of src0 rows to process per simdgroup
// N_SG: number of simdgroups per threadgroup
//
// TODO: for optimal performance, become function of the device and work size

#నిర్వచించు N_R0_Q1_0 8
#నిర్వచించు N_SG_Q1_0 2

#నిర్వచించు N_R0_Q2_0 8
#నిర్వచించు N_SG_Q2_0 2

#నిర్వచించు N_R0_Q4_0 4
#నిర్వచించు N_SG_Q4_0 2

#నిర్వచించు N_R0_Q4_1 4
#నిర్వచించు N_SG_Q4_1 2

#నిర్వచించు N_R0_Q5_0 4
#నిర్వచించు N_SG_Q5_0 2

#నిర్వచించు N_R0_Q5_1 4
#నిర్వచించు N_SG_Q5_1 2

#నిర్వచించు N_R0_Q8_0 2
#నిర్వచించు N_SG_Q8_0 4

#నిర్వచించు N_R0_MXFP4 2
#నిర్వచించు N_SG_MXFP4 2

#నిర్వచించు N_R0_Q2_K 4
#నిర్వచించు N_SG_Q2_K 2

#నిర్వచించు N_R0_Q3_K 2
#నిర్వచించు N_SG_Q3_K 2

#నిర్వచించు N_R0_Q4_K 2
#నిర్వచించు N_SG_Q4_K 2

#నిర్వచించు N_R0_Q5_K 1
#నిర్వచించు N_SG_Q5_K 2

#నిర్వచించు N_R0_Q6_K 2
#నిర్వచించు N_SG_Q6_K 2

#నిర్వచించు N_R0_IQ1_S 4
#నిర్వచించు N_SG_IQ1_S 2
#నిర్వచించు N_R0_IQ1_S_విభజించు 8

#నిర్వచించు N_R0_IQ1_M 4
#నిర్వచించు N_SG_IQ1_M 2
#నిర్వచించు N_R0_IQ1_M_విభజించు 8

#నిర్వచించు N_R0_IQ2_XXS 4
#నిర్వచించు N_SG_IQ2_XXS 2
#నిర్వచించు N_R0_IQ2_XXS_విభజించు 8

#నిర్వచించు N_R0_IQ2_XS 4
#నిర్వచించు N_SG_IQ2_XS 2
#నిర్వచించు N_R0_IQ2_XS_విభజించు 8

#నిర్వచించు N_R0_IQ2_S 4
#నిర్వచించు N_SG_IQ2_S 2
#నిర్వచించు N_R0_IQ2_S_విభజించు 8

#నిర్వచించు N_R0_IQ3_XXS 4
#నిర్వచించు N_SG_IQ3_XXS 2
#నిర్వచించు N_R0_IQ3_XXS_విభజించు 8

#నిర్వచించు N_R0_IQ3_S 4
#నిర్వచించు N_SG_IQ3_S 2
#నిర్వచించు N_R0_IQ3_S_విభజించు 8

#నిర్వచించు N_R0_IQ4_NL 2
#నిర్వచించు N_SG_IQ4_NL 2

#నిర్వచించు N_R0_IQ4_XS 2
#నిర్వచించు N_SG_IQ4_XS 2

#నిర్వచించు N_R0_TQ2_0 4
#నిర్వచించు N_SG_TQ2_0 2

// function constants offsets
#నిర్వచించు FC_FLASH_ATTN_EXT_PAD          100
#నిర్వచించు FC_FLASH_ATTN_EXT_BLK          200
#నిర్వచించు FC_FLASH_ATTN_EXT              300
#నిర్వచించు FC_FLASH_ATTN_EXT_VEC          400
#నిర్వచించు FC_FLASH_ATTN_EXT_VEC_REDUCE   500
#నిర్వచించు FC_MUL_MV                      600
#నిర్వచించు FC_MUL_MM                      700
#నిర్వచించు FC_ROPE                        800
#నిర్వచించు FC_SSM_CONV                    900
#నిర్వచించు FC_SOLVE_TRI                   1000
#నిర్వచించు FC_లెక్క_EQUAL                 1100
#నిర్వచించు FC_UNARY                       1200
#నిర్వచించు FC_BIN                         1300
#నిర్వచించు FC_కూడిక_అడ్డువరుసలు                    1400
#నిర్వచించు FC_UPSCALE                     1500
#నిర్వచించు FC_GATED_DELTA_NET             1600
#నిర్వచించు FC_NORM                        1700
#నిర్వచించు FC_TOPK_MOE                    1800
#నిర్వచించు FC_MOE_REDUCE                  1900
#నిర్వచించు FC_DSV4_HC                     2000
#నిర్వచించు FC_PAD                         2100
#నిర్వచించు FC_FLASH_ATTN_EXT_TENSOR       2200
#నిర్వచించు FC_LIGHTNING_INDEXER           2200
#నిర్వచించు FC_MUL_MV_MMA                  2300

// op-specific constants
#నిర్వచించు OP_FLASH_ATTN_EXT_NQPSG 8
#నిర్వచించు OP_FLASH_ATTN_EXT_NCPSG 64

#నిర్వచించు OP_FLASH_ATTN_EXT_TENSOR_NQPSG       32
#నిర్వచించు OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE 16
#నిర్వచించు OP_FLASH_ATTN_EXT_TENSOR_NCPSG       64
#నిర్వచించు OP_FLASH_ATTN_EXT_TENSOR_NSG         8

#నిర్వచించు OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#నిర్వచించు OP_FLASH_ATTN_EXT_VEC_NCPSG 32

#నిర్వచించు OP_LIGHTNING_INDEXER_DK    128
#నిర్వచించు OP_LIGHTNING_INDEXER_NHPTG   8
#నిర్వచించు OP_LIGHTNING_INDEXER_NKPSG   8
#నిర్వచించు OP_LIGHTNING_INDEXER_NSG     8
#నిర్వచించు OP_LIGHTNING_INDEXER_NBPTG   8

#నిర్వచించు OP_UNARY_అంకె_SCALE      10
#నిర్వచించు OP_UNARY_అంకె_FILL       11
#నిర్వచించు OP_UNARY_అంకె_CLAMP      12
#నిర్వచించు OP_UNARY_అంకె_SQR        13
#నిర్వచించు OP_UNARY_అంకె_SQRT       14
#నిర్వచించు OP_UNARY_అంకె_SIN        15
#నిర్వచించు OP_UNARY_అంకె_COS        16
#నిర్వచించు OP_UNARY_అంకె_LOG        17
#నిర్వచించు OP_UNARY_అంకె_LEAKY_RELU 18

#నిర్వచించు OP_UNARY_అంకె_TANH        100
#నిర్వచించు OP_UNARY_అంకె_RELU        101
#నిర్వచించు OP_UNARY_అంకె_SIGMOID     102
#నిర్వచించు OP_UNARY_అంకె_GELU        103
#నిర్వచించు OP_UNARY_అంకె_GELU_ERF    104
#నిర్వచించు OP_UNARY_అంకె_GELU_శీఘ్ర  105
#నిర్వచించు OP_UNARY_అంకె_SILU        106
#నిర్వచించు OP_UNARY_అంకె_ELU         107
#నిర్వచించు OP_UNARY_అంకె_NEG         108
#నిర్వచించు OP_UNARY_అంకె_ABS         109
#నిర్వచించు OP_UNARY_అంకె_SGN         110
#నిర్వచించు OP_UNARY_అంకె_అడుగు        111
#నిర్వచించు OP_UNARY_అంకె_HARDSWISH   112
#నిర్వచించు OP_UNARY_అంకె_HARDSIGMOID 113
#నిర్వచించు OP_UNARY_అంకె_EXP         114
#నిర్వచించు OP_UNARY_అంకె_SOFTPLUS    115
#నిర్వచించు OP_UNARY_అంకె_EXPM1       116
#నిర్వచించు OP_UNARY_అంకె_FLOOR       117
#నిర్వచించు OP_UNARY_అంకె_CEIL        118
#నిర్వచించు OP_UNARY_అంకె_విడత       119
#నిర్వచించు OP_UNARY_అంకె_TRUNC       120
#నిర్వచించు OP_UNARY_అంకె_XIELU       121

#నిర్వచించు OP_కూడిక_అడ్డువరుసలు_అంకె_కూడిక_అడ్డువరుసలు 10
#నిర్వచించు OP_కూడిక_అడ్డువరుసలు_అంకె_సగటు     11

#నిర్వచించు OP_SSM_పరిశీలన_SSD_CS  64 // Metal-specific; Chunk Size; 64 is largest multiple of 8 (simdgroup tile) fitting into 32 KiB Metal threadgroup mem limit (~26.75 KiB shared mem; see smem layout comment in kernel_ssm_scan_ssd_mma_f32)
#నిర్వచించు OP_SSM_పరిశీలన_SSD_HD  64 // Metal-specific; Head Dim the MMA kernel is specialized for (Mamba-2); use_mma gates on d_inner == this
#నిర్వచించు OP_SSM_పరిశీలన_SSD_NSG 4  // Metal-specific; Number of SimdGroups per threadgroup; NSG*32 == threads dispatched per threadgroup

// kernel argument structs
//
// - element counters (e.g. ne00) typically use int32_t to reduce register usage
//   however, be careful from int overflows when using those in the kernel implementation
//
// - strides (e.g. nb00) use uint64_t

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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
    దశాంశం    వాలు;
    దశాంశం    scale;
    దశాంశం    bias;
    దశాంశం    val;
    దశాంశం    min;
    దశాంశం    max;
} ggml_metal_kargs_unary;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
    int64_t ne0;
    int64_t ne1;
    size_t nb01;
    size_t nb02;
    size_t nb11;
    size_t nb21;
} ggml_metal_kargs_కూడు_id;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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
} ggml_metal_kargs_సమితి;

రకనిర్వచనం నిర్మాణం {
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
    దశాంశం    freq_పునాది;
    దశాంశం    freq_scale;
    దశాంశం    ext_గుణకం;
    దశాంశం    attn_గుణకం;
    దశాంశం    beta_fast;
    దశాంశం    beta_slow;
    int32_t  sect_0;
    int32_t  sect_1;
    int32_t  sect_2;
    int32_t  sect_3;
    bool     src2;
    bool     inplace;
} ggml_metal_kargs_rope;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ggml_metal_kargs_flash_attn_ext_blk;

రకనిర్వచనం నిర్మాణం {
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
    దశాంశం    scale;
    దశాంశం    max_bias;
    దశాంశం    m0;
    దశాంశం    m1;
    int32_t  n_తల_log2;
    దశాంశం    logit_softcap;
} ggml_metal_kargs_flash_attn_ext;

రకనిర్వచనం నిర్మాణం {
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
    దశాంశం    scale;
    దశాంశం    max_bias;
    దశాంశం    m0;
    దశాంశం    m1;
    int32_t  n_తల_log2;
    దశాంశం    logit_softcap;
    int32_t  n_kv_max_padded;
} ggml_metal_kargs_flash_attn_ext_vec;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
    int32_t  nrows;
} ggml_metal_kargs_flash_attn_ext_vec_reduce;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
    int32_t  ne10;
    int32_t  ne11;  // n_expert_used (bcast)
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21; // n_tokens
    int32_t  ne20;  // n_expert_used
    uint64_t nb21;
} ggml_metal_kargs_mul_mm_id_map0;

రకనిర్వచనం నిర్మాణం {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
} ggml_metal_kargs_mul_mm_id_amax;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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
రకనిర్వచనం నిర్మాణం {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    దశాంశం    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
    దశాంశం    scale;
} ggml_metal_kargs_norm;

రకనిర్వచనం నిర్మాణం {
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
    దశాంశం    eps;
} ggml_metal_kargs_l2_norm;

రకనిర్వచనం నిర్మాణం {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ngrp;
    దశాంశం    eps;
} ggml_metal_kargs_సమూహం_norm;

రకనిర్వచనం నిర్మాణం {
    int32_t  IC;
    int32_t  IL;
    int32_t  K;
    int32_t  s0;
    uint64_t nb0;
    uint64_t nb1;
} ggml_metal_kargs_conv_transpose_1d;

రకనిర్వచనం నిర్మాణం {
    int32_t  T_లోపల;
    int32_t  T_బహిర్గతం;
    int32_t  OC;
    int32_t  K;
    int32_t  K_OC;
    int32_t  s0;
    int32_t  p0;
} ggml_metal_kargs_col2im_1d;

రకనిర్వచనం నిర్మాణం {
    int32_t T;
    int32_t C;
} ggml_metal_kargs_snake;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం{
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    దశాంశం    alpha;
    దశాంశం    హద్దు;
} ggml_metal_kargs_glu;

రకనిర్వచనం నిర్మాణం {
    uint64_t np;
} ggml_metal_kargs_కూడిక;

రకనిర్వచనం నిర్మాణం {
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
} ggml_metal_kargs_కూడిక_అడ్డువరుసలు;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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
} ggml_metal_kargs_cumsum_కూడు;

రకనిర్వచనం నిర్మాణం {
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
    దశాంశం    scale;
    దశాంశం    max_bias;
    దశాంశం    m0;
    దశాంశం    m1;
    int32_t  n_తల_log2;
} ggml_metal_kargs_soft_max;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
    int64_t  d_state;
    int64_t  d_inner;
    int64_t  n_తల;
    int64_t  n_సమూహం;
    int64_t  n_seq_టోకెన్లు;
    int64_t  n_seq_టోకెన్లు_మొత్తం;
    int64_t  token_స్థానభ్రంశం;
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
} ggml_metal_kargs_ssm_పరిశీలన;

రకనిర్వచనం నిర్మాణం {
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
    uint64_t nb_బహిర్గతం; // 0 => snapshots are appended after the attn scores (unfused)
} ggml_metal_kargs_gated_delta_net;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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
} ggml_metal_kargs_get_అడ్డువరుసలు;

రకనిర్వచనం నిర్మాణం {
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
} ggml_metal_kargs_సమితి_అడ్డువరుసలు;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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
    దశాంశం    sf0;
    దశాంశం    sf1;
    దశాంశం    sf2;
    దశాంశం    sf3;
    దశాంశం    poffs;
} ggml_metal_kargs_upscale;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
    uint64_t nb1;
    పూర్ణాంకం      dim;
    పూర్ణాంకం      max_period;
} ggml_metal_kargs_timestep_embedding;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
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
    int32_t  శిఖరం_k;
} ggml_metal_kargs_argsort;

రకనిర్వచనం నిర్మాణం {
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
    int32_t  శిఖరం_k;
    int32_t  len;
} ggml_metal_kargs_argsort_విలీనం;

రకనిర్వచనం నిర్మాణం {
    int32_t  ne00;   // number of columns (elements per row)
    int32_t  ne01;   //అడ్డువరుసలు
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;   // row stride in src0
    uint64_t nb02;
    uint64_t nb03;
    int32_t  శిఖరం_k;  // k
} ggml_metal_kargs_శిఖరం_k;

// widths at or above this use the threadgroup FWHT kernel, one row per threadgroup
// with GGML_METAL_FWHT_TG_NT threads, instead of one row per simdgroup
#నిర్వచించు GGML_METAL_FWHT_TG_MIN_N 512
#నిర్వచించు GGML_METAL_FWHT_TG_NT    256

రకనిర్వచనం నిర్మాణం {
    int32_t  ne01;      // n_tokens
    uint64_t nb01;      // logits row stride
    uint64_t nb1_ids;   // ids row stride
    దశాంశం    clamp;
    దశాంశం    scale;
} ggml_metal_kargs_topk_moe;

రకనిర్వచనం నిర్మాణం {
    int32_t ne00; // n_embd
    int32_t ne02; // n_tokens
} ggml_metal_kargs_moe_reduce;

రకనిర్వచనం నిర్మాణం {
    int32_t nrows;
} ggml_metal_kargs_fwht;

రకనిర్వచనం నిర్మాణం {
    int64_t  ne0;
    దశాంశం    ప్రారంభం;
    దశాంశం    అడుగు;
} ggml_metal_kargs_arange;

రకనిర్వచనం నిర్మాణం {
    int64_t val;
} ggml_metal_kargs_memset;

రకనిర్వచనం నిర్మాణం {
    int32_t  n_kv;
    int32_t  n_batch;
    int32_t  ముసుగు_ne3;
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

రకనిర్వచనం నిర్మాణం {
    int32_t  n_టోకెన్లు;
    int32_t  n_iter;
    uint64_t nb_m0;
    uint64_t nb_m1;
    uint64_t nb_s0;
    uint64_t nb_b0;
    uint64_t nb_d0;
    uint64_t nb_d1;
    uint64_t nb_d2;
    దశాంశం    eps;
} ggml_metal_kargs_dsv4_hc_comb;

రకనిర్వచనం నిర్మాణం {
    int32_t  n_embd;
    int32_t  n_టోకెన్లు;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb_w2;
    uint64_t nb_d0;
    uint64_t nb_d1;
    దశాంశం    scale;
} ggml_metal_kargs_dsv4_hc_pre;

రకనిర్వచనం నిర్మాణం {
    int32_t  n_embd;
    int32_t  n_టోకెన్లు;
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

రకనిర్వచనం నిర్మాణం {
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
} ggml_metal_kargs_లెక్క_equal;

రకనిర్వచనం నిర్మాణం {
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

రకనిర్వచనం నిర్మాణం {
    int32_t  k0;
    int32_t  s0;
    int32_t  p0;
    int64_t  IW;
    int64_t  OW;
    int64_t  np;
} ggml_metal_kargs_pool_1d;

రకనిర్వచనం నిర్మాణం {
    int64_t ne00;
    uint64_t nb01;
    int64_t npart;
    int64_t ne01;
} ggml_metal_kargs_argmax;

రకనిర్వచనం నిర్మాణం {
    int64_t  np;
} ggml_metal_kargs_opt_అడుగు_adamw;

రకనిర్వచనం నిర్మాణం {
    int64_t  np;
} ggml_metal_kargs_opt_అడుగు_sgd;

రకనిర్వచనం నిర్మాణం {
    int64_t ne;
} ggml_metal_kargs_silu_వెనుక;

#అయితే_ముగింపు // GGML_METAL_IMPL
