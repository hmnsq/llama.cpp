#pragma once
#ಅನ್ತರ್ಭಾವ "ggml-vulkan-types.h"

uint32_t get_misalign_bytes(ಧ್ರುವ ggml_backend_vk_context * ctx, ಧ್ರುವ ggml_tensor * t);

uint32_t ggml_vk_concat_unit_size(ggml_type type);

ಸಂರಚನಾ vk_mat_mat_ತಳ್ಳು_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t base_work_group_z; uint32_t ಅಂಕೆ_batches;
    uint32_t k_split;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
    uint32_t padded_N;
};

ಸಂರಚನಾ vk_mat_vec_ತಳ್ಳು_constants {
    uint32_t ncols;
    uint32_t stride_a;
    uint32_t stride_b;
    uint32_t stride_d;
    uint32_t batch_stride_a;
    uint32_t batch_stride_b;
    uint32_t batch_stride_d;
    uint32_t fusion_flags;
    uint32_t base_work_group_y;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t broadcast2;
    uint32_t broadcast3;
};

ಸಂರಚನಾ vk_mat_vec_p021_ತಳ್ಳು_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t nchannels_x;
    uint32_t nchannels_y;
    uint32_t b_offset;
    uint32_t d_offset;
    uint32_t fusion_flags;
};

ಸಂರಚನಾ vk_mat_vec_nc_ತಳ್ಳು_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t row_stride_x;
    uint32_t channel_stride_x;
    uint32_t channel_stride_y;
    uint32_t channel_x_ಭಾಜಕ;
    uint32_t ne12;
    uint32_t b_offset;
    uint32_t d_offset;
    uint32_t nb03;
    uint32_t nb13;
    uint32_t nb23;
    uint32_t fusion_flags;
};

ಸಂರಚನಾ vk_mat_mat_id_ತಳ್ಳು_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t nei1; uint32_t nbi1; uint32_t ne11;
    uint32_t n_experts;
    uint32_t row_ids_offset;
};

ಸಂರಚನಾ vk_mat_vec_id_ತಳ್ಳು_constants {
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

ಸಂರಚನಾ vk_flash_attn_ತಳ್ಳು_constants {
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

    ಭಿನ್ನ scale;
    ಭಿನ್ನ max_bias;
    ಭಿನ್ನ logit_softcap;

    uint32_t mask_n_head_log2;
    ಭಿನ್ನ m0;
    ಭಿನ್ನ m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_ಅಂಕೆ;
};

static_assert(ಪರಿಮಾಣ(vk_flash_attn_ತಳ್ಳು_constants) <= 128, "sizeof(vk_flash_attn_push_constants) must be <= 128");

ಸಂರಚನಾ vk_fa_xe_opt_ತಳ್ಳು_constants {
    uint32_t kv_seq_len;
    uint32_t activation_length;
    uint32_t q_head;
    uint32_t kv_head;
    uint32_t qk_ratio;
    uint32_t qk_sub_groups;
    uint32_t flag;
    uint32_t nbkv_tok;
    uint32_t nbkv_head;
    uint32_t batch_stride_q;
    uint32_t batch_stride_k;
    uint32_t batch_stride_v;
    uint32_t batch_stride_m;
    uint32_t batch_stride_o;
    ಭಿನ್ನ softmax_scale;
};

ಸಂರಚನಾ vk_op_ತಳ್ಳು_constants {
    uint32_t KX;
    uint32_t KY;
    ಭಿನ್ನ param1;
    ಭಿನ್ನ param2;
    ಭಿನ್ನ param3;
    ಭಿನ್ನ param4;
};

ಸಂರಚನಾ vk_op_fwht_ತಳ್ಳು_constants {
    uint32_t n_rows;
    uint32_t src_offset;
    uint32_t dst_offset;
    ಭಿನ್ನ scale;
};

ಸಂರಚನಾ vk_op_dsv4_hc_comb_ತಳ್ಳು_constants {
    uint32_t n_tokens;

    uint32_t nbm0; uint32_t nbm1;
    uint32_t nbs0;
    uint32_t nbb0;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t m_offset;
    uint32_t s_offset;
    uint32_t b_offset;
    uint32_t d_offset;

    ಭಿನ್ನ eps;
    uint32_t n_iter;
};

ಸಂರಚನಾ vk_op_dsv4_hc_pre_ತಳ್ಳು_constants {
    uint32_t n_embd;
    uint32_t n_tokens;

    uint32_t nbx0; uint32_t nbx1; uint32_t nbx2;
    uint32_t nbw0; uint32_t nbw1; uint32_t nbw2;
    uint32_t nbd0; uint32_t nbd1;

    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t d_offset;

    ಭಿನ್ನ scale;
};

ಸಂರಚನಾ vk_op_dsv4_hc_post_ತಳ್ಳು_constants {
    uint32_t n_embd;
    uint32_t n_tokens;

    uint32_t nbx0; uint32_t nbx1;
    uint32_t nbr0; uint32_t nbr1; uint32_t nbr2;
    uint32_t nbp0; uint32_t nbp1;
    uint32_t nbc0; uint32_t nbc1; uint32_t nbc2;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t x_offset;
    uint32_t r_offset;
    uint32_t p_offset;
    uint32_t c_offset;
    uint32_t d_offset;

    uint32_t gate;
    ಭಿನ್ನ    gate_scale_in;
    ಭಿನ್ನ    gate_scale_out;
};

ಸಂರಚನಾ vk_op_ಎಣಿಕೆ_experts_ತಳ್ಳು_constants {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t nb00;
    uint32_t nb01;
    uint32_t a_offset;
    uint32_t n_experts;
    uint32_t ne00mp;
    uint32_t ne00L;
    uint32_t row_tile_size;
    uint32_t row_ids_offset;
};

ಸಂರಚನಾ vk_op_glu_ತಳ್ಳು_constants {
    uint32_t N;
    uint32_t ne00;
    uint32_t ne20;
    uint32_t mode;  // 0: default, 1: swapped, 2: split
    ಭಿನ್ನ alpha; // for swiglu_oai
    ಭಿನ್ನ limit;
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

static_assert(ಪರಿಮಾಣ(vk_op_glu_ತಳ್ಳು_constants) <= 128, "sizeof(vk_op_glu_push_constants) must be <= 128");

ಸಂರಚನಾ vk_op_unary_ತಳ್ಳು_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t misalign_offsets;
    ಭಿನ್ನ param1; ಭಿನ್ನ param2; ಭಿನ್ನ param3; ಭಿನ್ನ param4;
    uint32_t ne0_012mp; uint32_t ne0_01mp; uint32_t ne0_0mp; uint32_t ne0_Ls;
    uint32_t ne1_012mp; uint32_t ne1_01mp; uint32_t ne1_0mp; uint32_t ne1_Ls;
};

static_assert(ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants) <= 128, "sizeof(vk_op_unary_push_constants) must be <= 128");

ಸ್ಥಿರ vk_op_unary_ತಳ್ಳು_constants vk_op_unary_ತಳ್ಳು_constants_init(ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * dst, int64_t ne = 0) {
    GGML_ASSERT(ne != 0 || (ggml_nelements(src0) == ggml_nelements(dst)));
    ne = ne != 0 ? ne : ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_unary_ತಳ್ಳು_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_size(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_size(dst->type);
    p.ne10 = (uint32_t)dst->ne[0];
    p.ne11 = (uint32_t)dst->ne[1];
    p.ne12 = (uint32_t)dst->ne[2];
    p.ne13 = (uint32_t)dst->ne[3];
    p.nb10 = (uint32_t)(dst->nb[0] / dst_tsize);
    p.nb11 = (uint32_t)(dst->nb[1] / dst_tsize);
    p.nb12 = (uint32_t)(dst->nb[2] / dst_tsize);
    p.nb13 = (uint32_t)(dst->nb[3] / dst_tsize);

    ಪ್ರತ್ಯಾಗಮ p; // offsets are initialized later in ggml_vk_op
}

ಸಂರಚನಾ vk_op_pad_ತಳ್ಳು_constants {
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

ಸ್ಥಿರ vk_op_pad_ತಳ್ಳು_constants vk_op_pad_ತಳ್ಳು_constants_init(ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * dst) {
    int64_t ne = ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_pad_ತಳ್ಳು_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_size(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_size(dst->type);
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

    ಪ್ರತ್ಯಾಗಮ p; // fastdiv values and offsets are initialized later in ggml_vk_op
}

ಸ್ಥಿರ ಶೂನ್ಯ init_fastdiv_ಮೌಲ್ಯಗಳು(uint32_t d, uint32_t &mp, uint32_t &L)
{
    // compute L = ceil(log2(d));
    L = 0;
    ಯಾವತ್ (L < 32 && (uint32_t{1} << L) < d) {
        L++;
    }

    mp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
}

ಸ್ಥಿರ uint32_t pack_fastdiv_L(uint32_t L0, uint32_t L1, uint32_t L2) {
    ಪ್ರತ್ಯಾಗಮ L0 | (L1 << 8) | (L2 << 16);
}

template <typename T> ಶೂನ್ಯ init_pushconst_fastdiv(T &p) {
    GGML_UNUSED(p);
    static_assert(!std::is_const<T>::ಮೌಲ್ಯ, "unexpected type");
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_unary_ತಳ್ಳು_constants &p) {
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

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_glu_ತಳ್ಳು_constants &p) {
    // GLU linearizes over dst, then uses dst coordinates for src0/src1.
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne22*p.ne21*p.ne20,  p.ne2_012mp,    p.ne2_012L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne21*p.ne20,         p.ne2_01mp,     p.ne2_01L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne20,                p.ne2_0mp,      p.ne2_0L);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_ಎಣಿಕೆ_experts_ತಳ್ಳು_constants &p) {
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne00, p.ne00mp, p.ne00L);
}

ಸಂರಚನಾ vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23;
    uint32_t misalign_offsets;
    ಭಿನ್ನ param1; ಭಿನ್ನ param2; int32_t param3;
};

ಸಂರಚನಾ vk_op_concat_ತಳ್ಳು_constants : vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants {};

static_assert(ಪರಿಮಾಣ(vk_op_concat_ತಳ್ಳು_constants) == ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants));

static_assert(std::is_standard_layout_v<vk_op_concat_ತಳ್ಳು_constants>);

ಸಂರಚನಾ vk_op_multi_add_ತಳ್ಳು_constants {
    // shape for dst
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23;

    // strides for srcs+dst
    uint32_t nb[MAX_PARAMETER_ಎಣಿಕೆ][4];

    uint32_t rms_partials;
};

static_assert(MAX_PARAMETER_ಎಣಿಕೆ == 12);

static_assert(ಪರಿಮಾಣ(vk_op_multi_add_ತಳ್ಳು_constants) <= 256);

ಸಂರಚನಾ vk_op_topk_moe_ತಳ್ಳು_constants {
    uint32_t n_rows;
    uint32_t n_experts_ತಳ್ಳು;
    uint32_t n_expert_used;
    ಭಿನ್ನ clamp_min;
    ಭಿನ್ನ clamp_max;
    uint32_t gating_func;
    uint32_t has_bias;
    uint32_t with_norm;
    ಭಿನ್ನ output_scale;
    ಭಿನ್ನ output_bias;
};

ಸಂರಚನಾ vk_op_add_id_ತಳ್ಳು_constants {
    uint32_t ne0;
    uint32_t ne1;
    uint32_t s01;
    uint32_t s02;
    uint32_t s11;
    uint32_t s21;
};

ಸಂರಚನಾ vk_op_diag_mask_ತಳ್ಳು_constants {
    uint32_t ncols;
    uint32_t rows_per_channel;
    int32_t n_past;
};

ಸಂರಚನಾ vk_op_rope_ತಳ್ಳು_constants {
    uint32_t rope_mode;
    uint32_t nrows;
    uint32_t n_dims;
    uint32_t n_offs;
    ಭಿನ್ನ freq_scale;
    ಭಿನ್ನ freq_base;
    ಭಿನ್ನ ext_factor;
    ಭಿನ್ನ attn_factor;
    ಭಿನ್ನ corr_dims[2];
    ಭಿನ್ನ theta_scale;
    uint32_t has_ff;
    int32_t sections[4];
    uint32_t is_imrope;
    uint32_t is_back;
    uint32_t set_rows_stride;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t a_offset;
    uint32_t d_offset;
};

static_assert(ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants) <= 128, "sizeof(vk_op_rope_push_constants) must be <= 128");

ಸಂರಚನಾ vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants {
    vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants bin;
    vk_op_rope_ತಳ್ಳು_constants rope;
};

ಸಂರಚನಾ vk_op_soft_max_ತಳ್ಳು_constants {
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
    ಭಿನ್ನ scale;
    ಭಿನ್ನ max_bias;
    ಭಿನ್ನ m0;
    ಭಿನ್ನ m1;
    uint32_t n_head_log2;
    uint32_t nrows_x;
    uint32_t has_sinks;
};

ಸಂರಚನಾ vk_op_argsort_ತಳ್ಳು_constants {
    uint32_t ncols;
    uint32_t ncols_padded;
    uint32_t ncols_padded_log2;
    uint32_t nrows;
    uint32_t order;
    uint32_t outer_start;
    uint32_t outer_end;
    uint32_t inner_start;
    uint32_t inner_end;
};

ಸಂರಚನಾ vk_op_topk_ತಳ್ಳು_constants {
    uint32_t orig_ncols;
    uint32_t ncols_ಒಳಹರಿವು;
    uint32_t ncols_output;
    uint32_t k;
    uint32_t nrows;
    uint32_t first_pass;
    uint32_t last_pass;
};

ಸಂರಚನಾ vk_op_topk_radix_ತಳ್ಳು_constants {
    uint32_t ncols;
    uint32_t k;
    uint32_t nrows;
    uint32_t n_tps;    // QSA only
    uint32_t n_blocks; // QSA only
    uint32_t n_stream; // QSA only
};

ಸಂರಚನಾ vk_op_im2col_ತಳ್ಳು_constants {
    uint64_t dst_addr;
    uint32_t batch_offset; uint32_t offset_delta;
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

ಸಂರಚನಾ vk_op_im2col_3d_ತಳ್ಳು_constants {
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

ಸಂರಚನಾ vk_op_timestep_embedding_ತಳ್ಳು_constants {
    uint32_t nb1;
    uint32_t dim;
    uint32_t max_period;
};

ಸಂರಚನಾ vk_op_col2im_1d_ತಳ್ಳು_constants {
    uint32_t T_out;
    uint32_t OC;
    uint32_t K_OC;
    uint32_t T_in;
    uint32_t K;
    int32_t  stride;
    int32_t  p0;
};

ಸಂರಚನಾ vk_op_conv_transpose_1d_ತಳ್ಳು_constants {
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

ಸಂರಚನಾ vk_op_snake_ತಳ್ಳು_constants {
    uint32_t ne0;
    uint32_t ne1;
};

ಸಂರಚನಾ vk_op_pool1d_ತಳ್ಳು_constants {
    uint32_t IL;
    uint32_t OL;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0;
    int32_t s0;
    int32_t p0;
};

ಸಂರಚನಾ vk_op_pool2d_ತಳ್ಳು_constants {
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0; int32_t k1;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
};

ಸಂರಚನಾ vk_op_rwkv_wkv6_ತಳ್ಳು_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

ಸಂರಚನಾ vk_op_rwkv_wkv7_ತಳ್ಳು_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

ಸಂರಚನಾ vk_op_gated_linear_attn_ತಳ್ಳು_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
    ಭಿನ್ನ scale;
};

ಸಂರಚನಾ vk_op_lightning_indexer_ತಳ್ಳು_constants {
    uint32_t n_kv;
    uint32_t n_heads;
    uint32_t n_tokens;
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

static_assert(ಪರಿಮಾಣ(vk_op_lightning_indexer_ತಳ್ಳು_constants) <= 128);

ಸಂರಚನಾ vk_op_gated_delta_net_ತಳ್ಳು_constants {
    uint32_t H;
    uint32_t n_tokens;
    uint32_t n_seqs;
    uint32_t s_off;
    uint32_t sq1, sq2, sq3;
    uint32_t sv1, sv2, sv3;
    uint32_t sb1, sb2, sb3;
    uint32_t neq1, rq3;
    ಭಿನ್ನ scale;
    uint32_t K;
};

ಸಂರಚನಾ vk_op_ssm_scan_ತಳ್ಳು_constants {
    uint32_t nb02, nb03, nb12, nb13;
    uint32_t nb21, nb22, nb31;
    uint32_t nb42, nb43, nb52, nb53;
    uint32_t s_off;
    uint32_t n_head, d_head, n_group, n_tok;
    uint32_t n_seq, K;
};

ಸಂರಚನಾ vk_op_ssm_conv_ತಳ್ಳು_constants {
    uint32_t nb01, nb02;
    uint32_t nb11;
    uint32_t dst_nb0, dst_nb1, dst_nb2;
    uint32_t nc, ncs, nr, n_t, n_s;
};

ಸಂರಚನಾ vk_op_conv2d_ತಳ್ಳು_constants {
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

    uint32_t knl_offset;
    uint32_t src_offset;
    uint32_t dst_offset;
};

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_conv2d_ತಳ್ಳು_constants &p) {
    // Compute magic values to divide by OW, OW*OH
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.OW,       p.OWmp,    p.OWL);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.OW*p.OH,  p.OWOHmp,  p.OWOHL);
}

ಸಂರಚನಾ vk_op_conv3d_ತಳ್ಳು_constants {
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

    uint32_t knl_offset;
    uint32_t src_offset;
    uint32_t dst_offset;
};

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_conv3d_ತಳ್ಳು_constants &p) {
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.OW,             p.OWmp,     p.OWL);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.OW*p.OH,        p.OWOHmp,   p.OWOHL);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.OW*p.OH*p.OD,   p.OWOHODmp, p.OWOHODL);
}

ಸಂರಚನಾ vk_op_conv2d_dw_ತಳ್ಳು_constants {
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

ಸಂರಚನಾ vk_op_upscale_ತಳ್ಳು_constants {
    uint32_t ne; uint32_t a_offset; uint32_t d_offset;
    uint32_t ne00; uint32_t ne01;
    uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13;
    ಭಿನ್ನ sf0; ಭಿನ್ನ sf1; ಭಿನ್ನ sf2; ಭಿನ್ನ sf3;
    ಭಿನ್ನ pixel_offset;
};

ಸಂರಚನಾ vk_op_sum_rows_ತಳ್ಳು_constants
{
    uint32_t n_cols;
    uint32_t ne01, ne02;
    uint32_t nb01, nb02, nb03;
    uint32_t nb11, nb12, nb13;
    ಭಿನ್ನ weight;
    uint32_t misalign_offsets;
    uint32_t ne0_12mp, ne0_12L;
    uint32_t ne0_1mp, ne0_1L;
};

ಸ್ಥಿರ vk_op_sum_rows_ತಳ್ಳು_constants vk_op_sum_rows_ತಳ್ಳು_constants_init(ಧ್ರುವ ggml_tensor * src, ಧ್ರುವ ggml_tensor * dst, int64_t n_cols) {
    uint32_t type_size = (uint32_t)ggml_type_size(src->type);
    vk_op_sum_rows_ತಳ್ಳು_constants p = {};
    p.n_cols = (uint32_t)n_cols;
    p.ne01 = (uint32_t)src->ne[1];
    p.ne02 = (uint32_t)src->ne[2];
    p.nb01 = (uint32_t)src->nb[1] / type_size;
    p.nb02 = (uint32_t)src->nb[2] / type_size;
    p.nb03 = (uint32_t)src->nb[3] / type_size;
    p.nb11 = (uint32_t)dst->nb[1] / type_size;
    p.nb12 = (uint32_t)dst->nb[2] / type_size;
    p.nb13 = (uint32_t)dst->nb[3] / type_size;
    p.weight = 1.0f;
    ಪ್ರತ್ಯಾಗಮ p;
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_fastdiv(vk_op_sum_rows_ತಳ್ಳು_constants &p) {
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne01*p.ne02, p.ne0_12mp, p.ne0_12L);
    init_fastdiv_ಮೌಲ್ಯಗಳು(p.ne01,        p.ne0_1mp,  p.ne0_1L);
}

ಸಂರಚನಾ vk_quantize_q8_1_ತಳ್ಳು_constants {
    uint32_t ne;
    uint32_t ಅಂಕೆ_blocks;
};

ಸಂರಚನಾ vk_op_flash_attn_split_k_reduce_ತಳ್ಳು_constants {
    uint32_t D;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t k_ಅಂಕೆ;
    uint32_t sinks;
};

ಸಂರಚನಾ vk_op_flash_attn_mask_opt_ತಳ್ಳು_constants {
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

ಸಂರಚನಾ vk_op_flash_attn_sparse_compact_ತಳ್ಳು_constants {
    uint32_t KV;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nbm1;
    uint32_t nbm2;
    uint32_t nbm3;
    uint32_t n_kv_max;
};

template <typename T> ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, T &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    GGML_UNUSED(p);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
    GGML_UNUSED(dst);
    static_assert(!std::is_const<T>::ಮೌಲ್ಯ, "unexpected type");
    GGML_ASSERT(!src0 || get_misalign_bytes(ctx, src0) == 0);
    GGML_ASSERT(!src1 || get_misalign_bytes(ctx, src1) == 0);
    GGML_ASSERT(!src2 || get_misalign_bytes(ctx, src2) == 0);
    GGML_ASSERT(!src3 || get_misalign_bytes(ctx, src3) == 0);
    GGML_ASSERT(!dst  || get_misalign_bytes(ctx, dst) == 0);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_p021_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t b_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.b_offset = b_offset;
    p.d_offset = d_offset;

    GGML_UNUSED(src0);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_nc_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t b_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.b_offset = b_offset;
    p.d_offset = d_offset;

    GGML_UNUSED(src0);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_fwht_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    p.src_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.dst_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_comb_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    p.m_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.s_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    p.b_offset = get_misalign_bytes(ctx, src2) / ggml_type_size(src2->type);
    p.d_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_pre_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    p.x_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.w_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    p.d_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_post_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    p.x_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.r_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    p.p_offset = get_misalign_bytes(ctx, src2) / ggml_type_size(src2->type);
    p.c_offset = src3 ? get_misalign_bytes(ctx, src3) / ggml_type_size(src3->type) : 0;
    p.d_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);
}

template <typename T> size_t ತಳ್ಳು_constant_size(ಧ್ರುವ T &t) {
    static_assert(std::is_class<T>::ಮೌಲ್ಯ, "T must be a struct/class");
    GGML_UNUSED(t);
    ಪ್ರತ್ಯಾಗಮ ಪರಿಮಾಣ(T);
}

template <typename T> size_t ತಳ್ಳು_constant_size(ಧ್ರುವ std::vector<T> &t) {
    GGML_UNUSED(t);
    ಪ್ರತ್ಯಾಗಮ ಪರಿಮಾಣ(T) * t.size();
}

template <typename T, uint32_t N> size_t ತಳ್ಳು_constant_size(ಧ್ರುವ std::array<T, N> &t) {
    GGML_UNUSED(t);
    ಪ್ರತ್ಯಾಗಮ ಪರಿಮಾಣ(T) * N;
}

template <typename T> ಧ್ರುವ T *ತಳ್ಳು_constant_data(ಧ್ರುವ T &t) {
    static_assert(std::is_class<T>::ಮೌಲ್ಯ, "T must be a struct/class");
    ಪ್ರತ್ಯಾಗಮ &t;
}

template <typename T> ಧ್ರುವ T *ತಳ್ಳು_constant_data(ಧ್ರುವ std::vector<T> &t) {
    ಪ್ರತ್ಯಾಗಮ t.data();
}

template <typename T, uint32_t N> ಧ್ರುವ T *ತಳ್ಳು_constant_data(ಧ್ರುವ std::array<T, N> &t) {
    ಪ್ರತ್ಯಾಗಮ t.data();
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_unary_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_glu_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t b_offset = src1 ? get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type) : a_offset;
    ಧ್ರುವ uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    GGML_ASSERT(a_offset < (1u << 8));
    GGML_ASSERT(b_offset < (1u << 8));
    GGML_ASSERT(d_offset < (1u << 8));

    p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_sum_rows_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_pad_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv2d_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.src_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    p.dst_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv3d_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.src_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    p.dst_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_im2col_3d_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t a_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src0);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t b_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    GGML_ASSERT(a_offset <= 0xFFFF);
    GGML_ASSERT(b_offset <= 0xFF);
    GGML_ASSERT(d_offset <= 0xFF);

    p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_concat_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t unit_size = ggml_vk_concat_unit_size(dst->type);
    ಧ್ರುವ uint32_t a_offset = get_misalign_bytes(ctx, src0) / unit_size;
    ಧ್ರುವ uint32_t b_offset = get_misalign_bytes(ctx, src1) / unit_size;
    ಧ್ರುವ uint32_t d_offset = get_misalign_bytes(ctx, dst) / unit_size;

    p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_upscale_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.a_offset = a_offset;
    p.d_offset = d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_rope_ತಳ್ಳು_constants &p, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst) {
    p.a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.d_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

ಸ್ಥಿರ vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants ggml_vk_rms_norm_ತಳ್ಳು_constants(
        ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * dst,
        ಭಿನ್ನ eps, uint32_t ಅಂಕೆ_partials) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ಪ್ರತ್ಯಾಗಮ {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        eps, 0.0f, (int32_t)ಅಂಕೆ_partials,
    };
}

