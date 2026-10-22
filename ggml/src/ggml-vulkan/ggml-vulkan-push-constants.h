#ಸೂಚನೆ once
#ಅನ್ತರ್ಭಾವ "ggml-vulkan-types.h"

uint32_t get_misalign_ಬೈಟುಗಳು(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * t);

uint32_t ggml_vk_concat_unit_ಗಾತ್ರ(ggml_type type);

ರಚನೆ vk_mat_mat_ತಳ್ಳು_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t ತಳ_work_ಗುಂಪು_z; uint32_t ಅಂಕೆ_batches;
    uint32_t k_ವಿಭಜಿಸು;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
    uint32_t padded_N;
};

ರಚನೆ vk_mat_vec_ತಳ್ಳು_constants {
    uint32_t ncols;
    uint32_t stride_a;
    uint32_t stride_b;
    uint32_t stride_d;
    uint32_t batch_stride_a;
    uint32_t batch_stride_b;
    uint32_t batch_stride_d;
    uint32_t fusion_flags;
    uint32_t ತಳ_work_ಗುಂಪು_y;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t broadcast2;
    uint32_t broadcast3;
};

ರಚನೆ vk_mat_vec_p021_ತಳ್ಳು_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t nchannels_x;
    uint32_t nchannels_y;
    uint32_t b_ಸ್ಥಾನಾಂತರ;
    uint32_t d_ಸ್ಥಾನಾಂತರ;
    uint32_t fusion_flags;
};

ರಚನೆ vk_mat_vec_nc_ತಳ್ಳು_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t ಅಡ್ಡಸಾಲು_stride_x;
    uint32_t channel_stride_x;
    uint32_t channel_stride_y;
    uint32_t channel_x_ಭಾಜಕ;
    uint32_t ne12;
    uint32_t b_ಸ್ಥಾನಾಂತರ;
    uint32_t d_ಸ್ಥಾನಾಂತರ;
    uint32_t nb03;
    uint32_t nb13;
    uint32_t nb23;
    uint32_t fusion_flags;
};

ರಚನೆ vk_mat_mat_id_ತಳ್ಳು_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t nei1; uint32_t nbi1; uint32_t ne11;
    uint32_t n_experts;
    uint32_t ಅಡ್ಡಸಾಲು_ids_ಸ್ಥಾನಾಂತರ;
};

ರಚನೆ vk_mat_vec_id_ತಳ್ಳು_constants {
    uint32_t ncols;
    uint32_t stride_a;
    uint32_t stride_b;
    uint32_t stride_d;
    uint32_t batch_stride_a;
    uint32_t batch_stride_b;
    uint32_t batch_stride_d;
    uint32_t fusion_flags;
    uint32_t nei0;
    uint32_t ne11;
    uint32_t expert_i1;
    uint32_t nbi1;
};

ರಚನೆ vk_flash_attn_ತಳ್ಳು_constants {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    ದಶಮಾಂಶ scale;
    ದಶಮಾಂಶ max_bias;
    ದಶಮಾಂಶ logit_softcap;

    uint32_t ಮುಖವಾಡ_n_ತಲೆ_log2;
    ದಶಮಾಂಶ m0;
    ದಶಮಾಂಶ m1;

    uint32_t gqa_ಅನುಪಾತ;
    uint32_t ವಿಭಜಿಸು_kv;
    uint32_t k_ಅಂಕೆ;
};

static_assert(ಗಾತ್ರ(vk_flash_attn_ತಳ್ಳು_constants) <= 128, "sizeof(vk_flash_attn_push_constants) must be <= 128");

ರಚನೆ vk_fa_xe_opt_ತಳ್ಳು_constants {
    uint32_t kv_seq_len;
    uint32_t activation_ಉದ್ದಳತೆ;
    uint32_t q_ತಲೆ;
    uint32_t kv_ತಲೆ;
    uint32_t qk_ಅನುಪಾತ;
    uint32_t qk_sub_groups;
    uint32_t flag;
    uint32_t nbkv_tok;
    uint32_t nbkv_ತಲೆ;
    uint32_t batch_stride_q;
    uint32_t batch_stride_k;
    uint32_t batch_stride_v;
    uint32_t batch_stride_m;
    uint32_t batch_stride_o;
    ದಶಮಾಂಶ softmax_scale;
};

ರಚನೆ vk_op_ತಳ್ಳು_constants {
    uint32_t KX;
    uint32_t KY;
    ದಶಮಾಂಶ param1;
    ದಶಮಾಂಶ param2;
    ದಶಮಾಂಶ param3;
    ದಶಮಾಂಶ param4;
};

ರಚನೆ vk_op_fwht_ತಳ್ಳು_constants {
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t src_ಸ್ಥಾನಾಂತರ;
    uint32_t dst_ಸ್ಥಾನಾಂತರ;
    ದಶಮಾಂಶ scale;
};

ರಚನೆ vk_op_dsv4_hc_comb_ತಳ್ಳು_constants {
    uint32_t n_ಟೋಕನುಗಳು;

    uint32_t nbm0; uint32_t nbm1;
    uint32_t nbs0;
    uint32_t nbb0;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t m_ಸ್ಥಾನಾಂತರ;
    uint32_t s_ಸ್ಥಾನಾಂತರ;
    uint32_t b_ಸ್ಥಾನಾಂತರ;
    uint32_t d_ಸ್ಥಾನಾಂತರ;

    ದಶಮಾಂಶ eps;
    uint32_t n_iter;
};

ರಚನೆ vk_op_dsv4_hc_pre_ತಳ್ಳು_constants {
    uint32_t n_embd;
    uint32_t n_ಟೋಕನುಗಳು;

    uint32_t nbx0; uint32_t nbx1; uint32_t nbx2;
    uint32_t nbw0; uint32_t nbw1; uint32_t nbw2;
    uint32_t nbd0; uint32_t nbd1;

    uint32_t x_ಸ್ಥಾನಾಂತರ;
    uint32_t w_ಸ್ಥಾನಾಂತರ;
    uint32_t d_ಸ್ಥಾನಾಂತರ;

    ದಶಮಾಂಶ scale;
};

ರಚನೆ vk_op_dsv4_hc_post_ತಳ್ಳು_constants {
    uint32_t n_embd;
    uint32_t n_ಟೋಕನುಗಳು;

    uint32_t nbx0; uint32_t nbx1;
    uint32_t nbr0; uint32_t nbr1; uint32_t nbr2;
    uint32_t nbp0; uint32_t nbp1;
    uint32_t nbc0; uint32_t nbc1; uint32_t nbc2;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t x_ಸ್ಥಾನಾಂತರ;
    uint32_t r_ಸ್ಥಾನಾಂತರ;
    uint32_t p_ಸ್ಥಾನಾಂತರ;
    uint32_t c_ಸ್ಥಾನಾಂತರ;
    uint32_t d_ಸ್ಥಾನಾಂತರ;

    uint32_t gate;
    ದಶಮಾಂಶ    gate_scale_ಒಳ;
    ದಶಮಾಂಶ    gate_scale_ಹೊರಹೋಗು;
};

ರಚನೆ vk_op_ಎಣಿಕೆ_experts_ತಳ್ಳು_constants {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t nb00;
    uint32_t nb01;
    uint32_t a_ಸ್ಥಾನಾಂತರ;
    uint32_t n_experts;
    uint32_t ne00mp;
    uint32_t ne00L;
    uint32_t ಅಡ್ಡಸಾಲು_tile_ಗಾತ್ರ;
    uint32_t ಅಡ್ಡಸಾಲು_ids_ಸ್ಥಾನಾಂತರ;
};

ರಚನೆ vk_op_glu_ತಳ್ಳು_constants {
    uint32_t N;
    uint32_t ne00;
    uint32_t ne20;
    uint32_t ಬಹುಲಕ;  // 0: default, 1: swapped, 2: split
    ದಶಮಾಂಶ alpha; // for swiglu_oai
    ದಶಮಾಂಶ ಎಲ್ಲೆ;
    uint32_t nb00;
    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb10;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb20;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;
    uint32_t ne21;
    uint32_t ne22;
    uint32_t misalign_offsets;
    uint32_t ne2_012mp; uint32_t ne2_012L;
    uint32_t ne2_01mp;  uint32_t ne2_01L;
    uint32_t ne2_0mp;   uint32_t ne2_0L;
};

static_assert(ಗಾತ್ರ(vk_op_glu_ತಳ್ಳು_constants) <= 128, "sizeof(vk_op_glu_push_constants) must be <= 128");

ರಚನೆ vk_op_unary_ತಳ್ಳು_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t misalign_offsets;
    ದಶಮಾಂಶ param1; ದಶಮಾಂಶ param2; ದಶಮಾಂಶ param3; ದಶಮಾಂಶ param4;
    uint32_t ne0_012mp; uint32_t ne0_01mp; uint32_t ne0_0mp; uint32_t ne0_Ls;
    uint32_t ne1_012mp; uint32_t ne1_01mp; uint32_t ne1_0mp; uint32_t ne1_Ls;
};

static_assert(ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants) <= 128, "sizeof(vk_op_unary_push_constants) must be <= 128");

ಸ್ಥಿರ vk_op_unary_ತಳ್ಳು_constants vk_op_unary_ತಳ್ಳು_constants_init(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * dst, int64_t ne = 0) {
    GGML_ASSERT(ne != 0 || (ggml_nelements(src0) == ggml_nelements(dst)));
    ne = ne != 0 ? ne : ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_unary_ತಳ್ಳು_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_ಗಾತ್ರ(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_ಗಾತ್ರ(dst->type);
    p.ne10 = (uint32_t)dst->ne[0];
    p.ne11 = (uint32_t)dst->ne[1];
    p.ne12 = (uint32_t)dst->ne[2];
    p.ne13 = (uint32_t)dst->ne[3];
    p.nb10 = (uint32_t)(dst->nb[0] / dst_tsize);
    p.nb11 = (uint32_t)(dst->nb[1] / dst_tsize);
    p.nb12 = (uint32_t)(dst->nb[2] / dst_tsize);
    p.nb13 = (uint32_t)(dst->nb[3] / dst_tsize);

    ಹಿಂತಿರುಗಿ p; // offsets are initialized later in ggml_vk_op
}

ರಚನೆ vk_op_pad_ತಳ್ಳು_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t misalign_offsets;
    uint32_t circular;

    uint32_t lp0; uint32_t rp0;
    uint32_t lp1; uint32_t rp1;
    uint32_t lp2; uint32_t rp2;
    uint32_t lp3; uint32_t rp3;
};

ಸ್ಥಿರ vk_op_pad_ತಳ್ಳು_constants vk_op_pad_ತಳ್ಳು_constants_init(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    int64_t ne = ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_pad_ತಳ್ಳು_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_ಗಾತ್ರ(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_ಗಾತ್ರ(dst->type);
    p.ne10 = (uint32_t)dst->ne[0];
    p.ne11 = (uint32_t)dst->ne[1];
    p.ne12 = (uint32_t)dst->ne[2];
    p.ne13 = (uint32_t)dst->ne[3];
    p.nb10 = (uint32_t)(dst->nb[0] / dst_tsize);
    p.nb11 = (uint32_t)(dst->nb[1] / dst_tsize);
    p.nb12 = (uint32_t)(dst->nb[2] / dst_tsize);
    p.nb13 = (uint32_t)(dst->nb[3] / dst_tsize);

    p.lp0 = dst->op_params[0];
    p.rp0 = dst->op_params[1];
    p.lp1 = dst->op_params[2];
    p.rp1 = dst->op_params[3];
    p.lp2 = dst->op_params[4];
    p.rp2 = dst->op_params[5];
    p.lp3 = dst->op_params[6];
    p.rp3 = dst->op_params[7];
    p.circular = dst->op_params[8];

    ಹಿಂತಿರುಗಿ p; // fastdiv values and offsets are initialized later in ggml_vk_op
}

ಸ್ಥಿರ ಶೂನ್ಯ init_fastdiv_ಮೌಲ್ಯಗಳು(uint32_t d, uint32_t &mp, uint32_t &L)
{
    // compute L = ceil(log2(d));
    L = 0;
    ಆಗ (L < 32 && (uint32_t{1} << L) < d) {
        L++;
    }

    mp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
}

ಸ್ಥಿರ uint32_t pack_fastdiv_L(uint32_t L0, uint32_t L1, uint32_t L2) {
    ಹಿಂತಿರುಗಿ L0 | (L1 << 8) | (L2 << 16);
}

ಮಾದರಿ <typename T> ಶೂನ್ಯ init_pushconst_fastdiv(T &p) {
    GGML_ಬಳಸದ(p);
    static_assert(!std::is_const<T>::ಮೌಲ್ಯ, "unexpected type");
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_unary_ತಳ್ಳು_constants &p) {
    // Compute magic values to divide by these six numbers.
    uint32_t ne0_012L;
    uint32_t ne0_01L;
    uint32_t ne0_0L;
    uint32_t ne1_012L;
    uint32_t ne1_01L;
    uint32_t ne1_0L;

    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne02*p.ne01*p.ne00,  p.ne0_012mp,    ne0_012L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne01*p.ne00,         p.ne0_01mp,     ne0_01L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne00,                p.ne0_0mp,      ne0_0L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne12*p.ne11*p.ne10,  p.ne1_012mp,    ne1_012L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne11*p.ne10,         p.ne1_01mp,     ne1_01L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne10,                p.ne1_0mp,      ne1_0L);

    p.ne0_Ls = pack_fastdiv_L(ne0_012L, ne0_01L, ne0_0L);
    p.ne1_Ls = pack_fastdiv_L(ne1_012L, ne1_01L, ne1_0L);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_glu_ತಳ್ಳು_constants &p) {
    // GLU linearizes over dst, then uses dst coordinates for src0/src1.
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne22*p.ne21*p.ne20,  p.ne2_012mp,    p.ne2_012L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne21*p.ne20,         p.ne2_01mp,     p.ne2_01L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne20,                p.ne2_0mp,      p.ne2_0L);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_ಎಣಿಕೆ_experts_ತಳ್ಳು_constants &p) {
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne00, p.ne00mp, p.ne00L);
}

ರಚನೆ vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23;
    uint32_t misalign_offsets;
    ದಶಮಾಂಶ param1; ದಶಮಾಂಶ param2; int32_t param3;
};

ರಚನೆ vk_op_concat_ತಳ್ಳು_constants : vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants {};

static_assert(ಗಾತ್ರ(vk_op_concat_ತಳ್ಳು_constants) == ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants));

static_assert(std::is_standard_layout_v<vk_op_concat_ತಳ್ಳು_constants>);

ರಚನೆ vk_op_multi_ಕೂಡಿಸು_ತಳ್ಳು_constants {
    // shape for dst
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23;

    // strides for srcs+dst
    uint32_t nb[MAX_PARAMETER_ಎಣಿಕೆ][4];

    uint32_t rms_partials;
};

static_assert(MAX_PARAMETER_ಎಣಿಕೆ == 12);

static_assert(ಗಾತ್ರ(vk_op_multi_ಕೂಡಿಸು_ತಳ್ಳು_constants) <= 256);

ರಚನೆ vk_op_topk_moe_ತಳ್ಳು_constants {
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t n_experts_ತಳ್ಳು;
    uint32_t n_expert_ಬಳಸಿದ;
    ದಶಮಾಂಶ clamp_min;
    ದಶಮಾಂಶ clamp_max;
    uint32_t gating_func;
    uint32_t has_bias;
    uint32_t with_norm;
    ದಶಮಾಂಶ output_scale;
    ದಶಮಾಂಶ output_bias;
};

ರಚನೆ vk_op_ಕೂಡಿಸು_id_ತಳ್ಳು_constants {
    uint32_t ne0;
    uint32_t ne1;
    uint32_t s01;
    uint32_t s02;
    uint32_t s11;
    uint32_t s21;
};

ರಚನೆ vk_op_diag_ಮುಖವಾಡ_ತಳ್ಳು_constants {
    uint32_t ncols;
    uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_channel;
    int32_t n_past;
};

ರಚನೆ vk_op_rope_ತಳ್ಳು_constants {
    uint32_t rope_ಬಹುಲಕ;
    uint32_t nrows;
    uint32_t n_dims;
    uint32_t n_offs;
    ದಶಮಾಂಶ freq_scale;
    ದಶಮಾಂಶ freq_ತಳ;
    ದಶಮಾಂಶ ext_ಗುಣಕ;
    ದಶಮಾಂಶ attn_ಗುಣಕ;
    ದಶಮಾಂಶ corr_dims[2];
    ದಶಮಾಂಶ theta_scale;
    uint32_t has_ff;
    int32_t sections[4];
    uint32_t is_imrope;
    uint32_t is_ಹಿಂಭಾಗ;
    uint32_t ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t a_ಸ್ಥಾನಾಂತರ;
    uint32_t d_ಸ್ಥಾನಾಂತರ;
};

static_assert(ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants) <= 128, "sizeof(vk_op_rope_push_constants) must be <= 128");

ರಚನೆ vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants {
    vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants bin;
    vk_op_rope_ತಳ್ಳು_constants rope;
};

ರಚನೆ vk_op_soft_max_ತಳ್ಳು_constants {
    uint32_t KX;
    uint32_t KY;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t ne13;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    ದಶಮಾಂಶ scale;
    ದಶಮಾಂಶ max_bias;
    ದಶಮಾಂಶ m0;
    ದಶಮಾಂಶ m1;
    uint32_t n_ತಲೆ_log2;
    uint32_t nrows_x;
    uint32_t has_sinks;
};

ರಚನೆ vk_op_argsort_ತಳ್ಳು_constants {
    uint32_t ncols;
    uint32_t ncols_padded;
    uint32_t ncols_padded_log2;
    uint32_t nrows;
    uint32_t ಕ್ರಮ;
    uint32_t outer_ಆರಂಭ;
    uint32_t outer_ಕೊನೆ;
    uint32_t inner_ಆರಂಭ;
    uint32_t inner_ಕೊನೆ;
};

ರಚನೆ vk_op_topk_ತಳ್ಳು_constants {
    uint32_t orig_ncols;
    uint32_t ncols_ಒಳಹರಿವು;
    uint32_t ncols_output;
    uint32_t k;
    uint32_t nrows;
    uint32_t ಮೊದಲ_pass;
    uint32_t ಕೊನೆಯ_pass;
};

ರಚನೆ vk_op_topk_radix_ತಳ್ಳು_constants {
    uint32_t ncols;
    uint32_t k;
    uint32_t nrows;
    uint32_t n_tps;    // QSA only
    uint32_t n_ಖಂಡಗಳು; // QSA only
    uint32_t n_ಹರಿವು; // QSA only
};

ರಚನೆ vk_op_im2col_ತಳ್ಳು_constants {
    uint64_t dst_addr;
    uint32_t batch_ಸ್ಥಾನಾಂತರ; uint32_t ಸ್ಥಾನಾಂತರ_delta;
    uint32_t IC;
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t KW; uint32_t KH;
    uint32_t OH_batch;
    uint32_t CHW;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
    int32_t d0; int32_t d1;
    uint32_t batch_IC;
};

ರಚನೆ vk_op_im2col_3d_ತಳ್ಳು_constants {
    uint64_t dst_addr;
    uint32_t nb10;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t s0;
    uint32_t s1;
    uint32_t s2;
    uint32_t p0;
    uint32_t p1;
    uint32_t p2;
    uint32_t d0;
    uint32_t d1;
    uint32_t d2;
    uint32_t IW;
    uint32_t IH;
    uint32_t ID;
    uint32_t IC;
    uint32_t KW;
    uint32_t OH;
    uint32_t KD_KH_KW;
    uint32_t KH_KW;
    uint32_t IC_KD_KH_KW;
    uint32_t N_OD_OH;
    uint32_t OD_OH;
    uint32_t OD_OH_OW_IC_KD_KH_KW;
    uint32_t OH_OW_IC_KD_KH_KW;
    uint32_t OW_IC_KD_KH_KW;
    uint32_t misalign_offsets;
};

ರಚನೆ vk_op_timestep_embedding_ತಳ್ಳು_constants {
    uint32_t nb1;
    uint32_t dim;
    uint32_t max_period;
};

ರಚನೆ vk_op_col2im_1d_ತಳ್ಳು_constants {
    uint32_t T_ಹೊರಹೋಗು;
    uint32_t OC;
    uint32_t K_OC;
    uint32_t T_ಒಳ;
    uint32_t K;
    int32_t  stride;
    int32_t  p0;
};

ರಚನೆ vk_op_conv_transpose_1d_ತಳ್ಳು_constants {
    uint32_t Cout;
    uint32_t Cin;
    uint32_t K;
    uint32_t L;
    uint32_t KL;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb11;
    uint32_t nb1;

    int32_t s0;
};

ರಚನೆ vk_op_snake_ತಳ್ಳು_constants {
    uint32_t ne0;
    uint32_t ne1;
};

ರಚನೆ vk_op_pool1d_ತಳ್ಳು_constants {
    uint32_t IL;
    uint32_t OL;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0;
    int32_t s0;
    int32_t p0;
};

ರಚನೆ vk_op_pool2d_ತಳ್ಳು_constants {
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0; int32_t k1;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
};

ರಚನೆ vk_op_rwkv_wkv6_ತಳ್ಳು_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

ರಚನೆ vk_op_rwkv_wkv7_ತಳ್ಳು_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

ರಚನೆ vk_op_gated_ರೇಖೀಯ_attn_ತಳ್ಳು_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
    ದಶಮಾಂಶ scale;
};

ರಚನೆ vk_op_lightning_indexer_ತಳ್ಳು_constants {
    uint32_t n_kv;
    uint32_t n_heads;
    uint32_t n_ಟೋಕನುಗಳು;
    uint32_t n_masks;
    uint32_t q_nb1;
    uint32_t q_nb2;
    uint32_t q_nb3;
    uint32_t k_nb2;
    uint32_t k_nb3;
    uint32_t w_nb1;
    uint32_t w_nb3;
    uint32_t m_nb1;
    uint32_t m_nb3;
    uint32_t d_nb1;
    uint32_t d_nb3;
};

static_assert(ಗಾತ್ರ(vk_op_lightning_indexer_ತಳ್ಳು_constants) <= 128);

ರಚನೆ vk_op_gated_delta_net_ತಳ್ಳು_constants {
    uint32_t H;
    uint32_t n_ಟೋಕನುಗಳು;
    uint32_t n_seqs;
    uint32_t s_off;
    uint32_t sq1, sq2, sq3;
    uint32_t sv1, sv2, sv3;
    uint32_t sb1, sb2, sb3;
    uint32_t neq1, rq3;
    ದಶಮಾಂಶ scale;
    uint32_t K;
};

ರಚನೆ vk_op_ssm_ಪರಿಶೀಲನೆ_ತಳ್ಳು_constants {
    uint32_t nb02, nb03, nb12, nb13;
    uint32_t nb21, nb22, nb31;
    uint32_t nb42, nb43, nb52, nb53;
    uint32_t s_off;
    uint32_t n_ತಲೆ, d_ತಲೆ, n_ಗುಂಪು, n_tok;
    uint32_t n_seq, K;
};

ರಚನೆ vk_op_ssm_conv_ತಳ್ಳು_constants {
    uint32_t nb01, nb02;
    uint32_t nb11;
    uint32_t dst_nb0, dst_nb1, dst_nb2;
    uint32_t nc, ncs, nr, n_t, n_s;
};

ರಚನೆ vk_op_conv2d_ತಳ್ಳು_constants {
    uint32_t Cout;
    uint32_t Cin;
    uint32_t N;

    uint32_t W;
    uint32_t H;
    uint32_t OW;
    uint32_t OH;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;

    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;

    uint32_t nb1;
    uint32_t nb2;
    uint32_t nb3;

    // init_fastdiv_values constants for dividing by OW, OW*OH
    uint32_t OWmp;   uint32_t OWL;
    uint32_t OWOHmp; uint32_t OWOHL;

    uint32_t knl_ಸ್ಥಾನಾಂತರ;
    uint32_t src_ಸ್ಥಾನಾಂತರ;
    uint32_t dst_ಸ್ಥಾನಾಂತರ;
};

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_conv2d_ತಳ್ಳು_constants &p) {
    // Compute magic values to divide by OW, OW*OH
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.OW,       p.OWmp,    p.OWL);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.OW*p.OH,  p.OWOHmp,  p.OWOHL);
}

ರಚನೆ vk_op_conv3d_ತಳ್ಳು_constants {
    uint32_t OC;
    uint32_t IC;
    uint32_t N;

    uint32_t IW;
    uint32_t IH;
    uint32_t ID;
    uint32_t OW;
    uint32_t OH;
    uint32_t OD;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;

    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;

    uint32_t nb1;
    uint32_t nb2;
    uint32_t nb3;

    uint32_t OWmp;     uint32_t OWL;
    uint32_t OWOHmp;   uint32_t OWOHL;
    uint32_t OWOHODmp; uint32_t OWOHODL;

    uint32_t knl_ಸ್ಥಾನಾಂತರ;
    uint32_t src_ಸ್ಥಾನಾಂತರ;
    uint32_t dst_ಸ್ಥಾನಾಂತರ;
};

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_conv3d_ತಳ್ಳು_constants &p) {
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.OW,             p.OWmp,     p.OWL);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.OW*p.OH,        p.OWOHmp,   p.OWOHL);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.OW*p.OH*p.OD,   p.OWOHODmp, p.OWOHODL);
}

ರಚನೆ vk_op_conv2d_dw_ತಳ್ಳು_constants {
    uint32_t ne;
    uint32_t batches;
    uint32_t channels;
    uint32_t dst_w;
    uint32_t dst_h;
    uint32_t src_w;
    uint32_t src_h;
    uint32_t knl_w;
    uint32_t knl_h;
    int32_t stride_x;
    int32_t stride_y;
    int32_t pad_x;
    int32_t pad_y;
    int32_t dilation_x;
    int32_t dilation_y;
};

ರಚನೆ vk_op_upscale_ತಳ್ಳು_constants {
    uint32_t ne; uint32_t a_ಸ್ಥಾನಾಂತರ; uint32_t d_ಸ್ಥಾನಾಂತರ;
    uint32_t ne00; uint32_t ne01;
    uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13;
    ದಶಮಾಂಶ sf0; ದಶಮಾಂಶ sf1; ದಶಮಾಂಶ sf2; ದಶಮಾಂಶ sf3;
    ದಶಮಾಂಶ pixel_ಸ್ಥಾನಾಂತರ;
};

ರಚನೆ vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants
{
    uint32_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    uint32_t ne01, ne02;
    uint32_t nb01, nb02, nb03;
    uint32_t nb11, nb12, nb13;
    ದಶಮಾಂಶ ತೂಕ;
    uint32_t misalign_offsets;
    uint32_t ne0_12mp, ne0_12L;
    uint32_t ne0_1mp, ne0_1L;
};

ಸ್ಥಿರ vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants_init(ಸ್ಥಿರಾಂಕ ggml_tensor * src, ಸ್ಥಿರಾಂಕ ggml_tensor * dst, int64_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) {
    uint32_t type_ಗಾತ್ರ = (uint32_t)ggml_type_ಗಾತ್ರ(src->type);
    vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants p = {};
    p.n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = (uint32_t)n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    p.ne01 = (uint32_t)src->ne[1];
    p.ne02 = (uint32_t)src->ne[2];
    p.nb01 = (uint32_t)src->nb[1] / type_ಗಾತ್ರ;
    p.nb02 = (uint32_t)src->nb[2] / type_ಗಾತ್ರ;
    p.nb03 = (uint32_t)src->nb[3] / type_ಗಾತ್ರ;
    p.nb11 = (uint32_t)dst->nb[1] / type_ಗಾತ್ರ;
    p.nb12 = (uint32_t)dst->nb[2] / type_ಗಾತ್ರ;
    p.nb13 = (uint32_t)dst->nb[3] / type_ಗಾತ್ರ;
    p.ತೂಕ = 1.0f;
    ಹಿಂತಿರುಗಿ p;
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants &p) {
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne01*p.ne02, p.ne0_12mp, p.ne0_12L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne01,        p.ne0_1mp,  p.ne0_1L);
}

ರಚನೆ vk_quantize_q8_1_ತಳ್ಳು_constants {
    uint32_t ne;
    uint32_t ಅಂಕೆ_ಖಂಡಗಳು;
};

ರಚನೆ vk_op_flash_attn_ವಿಭಜಿಸು_k_reduce_ತಳ್ಳು_constants {
    uint32_t D;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t k_ಅಂಕೆ;
    uint32_t sinks;
};

ರಚನೆ vk_op_flash_attn_ಮುಖವಾಡ_opt_ತಳ್ಳು_constants {
    uint32_t nem0;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nbm1;
    uint32_t nbm2;
    uint32_t nbm3;
    uint32_t nbd1;
    uint32_t nbd2;
    uint32_t nbd3;
};

ರಚನೆ vk_op_flash_attn_sparse_compact_ತಳ್ಳು_constants {
    uint32_t KV;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nbm1;
    uint32_t nbm2;
    uint32_t nbm3;
    uint32_t n_kv_max;
};

ಮಾದರಿ <typename T> ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, T &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    GGML_ಬಳಸದ(p);
    GGML_ಬಳಸದ(src0);
    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
    GGML_ಬಳಸದ(dst);
    static_assert(!std::is_const<T>::ಮೌಲ್ಯ, "unexpected type");
    GGML_ASSERT(!src0 || get_misalign_ಬೈಟುಗಳು(ctx, src0) == 0);
    GGML_ASSERT(!src1 || get_misalign_ಬೈಟುಗಳು(ctx, src1) == 0);
    GGML_ASSERT(!src2 || get_misalign_ಬೈಟುಗಳು(ctx, src2) == 0);
    GGML_ASSERT(!src3 || get_misalign_ಬೈಟುಗಳು(ctx, src3) == 0);
    GGML_ASSERT(!dst  || get_misalign_ಬೈಟುಗಳು(ctx, dst) == 0);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_p021_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t b_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type);

    p.b_ಸ್ಥಾನಾಂತರ = b_ಸ್ಥಾನಾಂತರ;
    p.d_ಸ್ಥಾನಾಂತರ = d_ಸ್ಥಾನಾಂತರ;

    GGML_ಬಳಸದ(src0);
    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_nc_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t b_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type);

    p.b_ಸ್ಥಾನಾಂತರ = b_ಸ್ಥಾನಾಂತರ;
    p.d_ಸ್ಥಾನಾಂತರ = d_ಸ್ಥಾನಾಂತರ;

    GGML_ಬಳಸದ(src0);
    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_fwht_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    p.src_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    p.dst_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst)  / ggml_type_ಗಾತ್ರ(dst->type);

    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_comb_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    p.m_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    p.s_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type);
    p.b_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src2) / ggml_type_ಗಾತ್ರ(src2->type);
    p.d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst)  / ggml_type_ಗಾತ್ರ(dst->type);

    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_pre_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    p.x_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    p.w_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type);
    p.d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst)  / ggml_type_ಗಾತ್ರ(dst->type);

    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_post_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    p.x_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    p.r_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type);
    p.p_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src2) / ggml_type_ಗಾತ್ರ(src2->type);
    p.c_ಸ್ಥಾನಾಂತರ = src3 ? get_misalign_ಬೈಟುಗಳು(ctx, src3) / ggml_type_ಗಾತ್ರ(src3->type) : 0;
    p.d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst)  / ggml_type_ಗಾತ್ರ(dst->type);
}

ಮಾದರಿ <typename T> size_t ತಳ್ಳು_constant_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ T &t) {
    static_assert(std::is_class<T>::ಮೌಲ್ಯ, "T must be a struct/class");
    GGML_ಬಳಸದ(t);
    ಹಿಂತಿರುಗಿ ಗಾತ್ರ(T);
}

ಮಾದರಿ <typename T> size_t ತಳ್ಳು_constant_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ std::ಸದಿಶ<T> &t) {
    GGML_ಬಳಸದ(t);
    ಹಿಂತಿರುಗಿ ಗಾತ್ರ(T) * t.ಗಾತ್ರ_2();
}

ಮಾದರಿ <typename T, uint32_t N> size_t ತಳ್ಳು_constant_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ std::array<T, N> &t) {
    GGML_ಬಳಸದ(t);
    ಹಿಂತಿರುಗಿ ಗಾತ್ರ(T) * N;
}

ಮಾದರಿ <typename T> ಸ್ಥಿರಾಂಕ T *ತಳ್ಳು_constant_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ T &t) {
    static_assert(std::is_class<T>::ಮೌಲ್ಯ, "T must be a struct/class");
    ಹಿಂತಿರುಗಿ &t;
}

ಮಾದರಿ <typename T> ಸ್ಥಿರಾಂಕ T *ತಳ್ಳು_constant_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ std::ಸದಿಶ<T> &t) {
    ಹಿಂತಿರುಗಿ t.ದತ್ತಾಂಶ();
}

ಮಾದರಿ <typename T, uint32_t N> ಸ್ಥಿರಾಂಕ T *ತಳ್ಳು_constant_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ std::array<T, N> &t) {
    ಹಿಂತಿರುಗಿ t.ದತ್ತಾಂಶ();
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_unary_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t a_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type);

    p.misalign_offsets = (a_ಸ್ಥಾನಾಂತರ << 16) | d_ಸ್ಥಾನಾಂತರ;

    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_glu_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t a_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t b_ಸ್ಥಾನಾಂತರ = src1 ? get_misalign_ಬೈಟುಗಳು(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type) : a_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint32_t d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type);

    GGML_ASSERT(a_ಸ್ಥಾನಾಂತರ < (1u << 8));
    GGML_ASSERT(b_ಸ್ಥಾನಾಂತರ < (1u << 8));
    GGML_ASSERT(d_ಸ್ಥಾನಾಂತರ < (1u << 8));

    p.misalign_offsets = (a_ಸ್ಥಾನಾಂತರ << 16) | (b_ಸ್ಥಾನಾಂತರ << 8) | d_ಸ್ಥಾನಾಂತರ;

    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t a_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type);

    p.misalign_offsets = (a_ಸ್ಥಾನಾಂತರ << 16) | d_ಸ್ಥಾನಾಂತರ;

    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_pad_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t a_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type);

    p.misalign_offsets = (a_ಸ್ಥಾನಾಂತರ << 16) | d_ಸ್ಥಾನಾಂತರ;

    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv2d_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    p.src_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type);
    p.dst_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst)  / ggml_type_ಗಾತ್ರ(dst->type);

    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv3d_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    p.src_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type);
    p.dst_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst)  / ggml_type_ಗಾತ್ರ(dst->type);

    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_im2col_3d_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t a_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type);

    p.misalign_offsets = (a_ಸ್ಥಾನಾಂತರ << 16) | d_ಸ್ಥಾನಾಂತರ;

    GGML_ಬಳಸದ(src0);
    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t a_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t b_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type);

    GGML_ASSERT(a_ಸ್ಥಾನಾಂತರ <= 0xFFFF);
    GGML_ASSERT(b_ಸ್ಥಾನಾಂತರ <= 0xFF);
    GGML_ASSERT(d_ಸ್ಥಾನಾಂತರ <= 0xFF);

    p.misalign_offsets = (a_ಸ್ಥಾನಾಂತರ << 16) | (b_ಸ್ಥಾನಾಂತರ << 8) | d_ಸ್ಥಾನಾಂತರ;

    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_concat_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t unit_ಗಾತ್ರ = ggml_vk_concat_unit_ಗಾತ್ರ(dst->type);
    ಸ್ಥಿರಾಂಕ uint32_t a_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / unit_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t b_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src1) / unit_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst) / unit_ಗಾತ್ರ;

    p.misalign_offsets = (a_ಸ್ಥಾನಾಂತರ << 16) | (b_ಸ್ಥಾನಾಂತರ << 8) | d_ಸ್ಥಾನಾಂತರ;

    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_upscale_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t a_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type);

    p.a_ಸ್ಥಾನಾಂತರ = a_ಸ್ಥಾನಾಂತರ;
    p.d_ಸ್ಥಾನಾಂತರ = d_ಸ್ಥಾನಾಂತರ;

    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_rope_ತಳ್ಳು_constants &p, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst) {
    p.a_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type);
    p.d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, dst)  / ggml_type_ಗಾತ್ರ(dst->type);

    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(src2);
    GGML_ಬಳಸದ(src3);
}

ಸ್ಥಿರ vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants ggml_vk_rms_norm_ತಳ್ಳು_constants(
        ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * dst,
        ದಶಮಾಂಶ eps, uint32_t ಅಂಕೆ_partials) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ಹಿಂತಿರುಗಿ {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        eps, 0.0f, (int32_t)ಅಂಕೆ_partials,
    };
}

