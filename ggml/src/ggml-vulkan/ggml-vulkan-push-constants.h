#ਹਦਾਇਤ once
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-vulkan-types.h"

uint32_t get_misalign_ਬਾਈਟਾਂ(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * t);

uint32_t ggml_vk_concat_unit_ਮਾਪ(ggml_type type);

ਬਣਤਰ vk_mat_mat_ਧੱਕੋ_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t ਅਧਾਰ_work_ਸਮੂਹ_z; uint32_t ਅੰਕ_batches;
    uint32_t k_ਵੰਡੋ;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
    uint32_t padded_N;
};

ਬਣਤਰ vk_mat_vec_ਧੱਕੋ_constants {
    uint32_t ncols;
    uint32_t stride_a;
    uint32_t stride_b;
    uint32_t stride_d;
    uint32_t batch_stride_a;
    uint32_t batch_stride_b;
    uint32_t batch_stride_d;
    uint32_t fusion_flags;
    uint32_t ਅਧਾਰ_work_ਸਮੂਹ_y;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t broadcast2;
    uint32_t broadcast3;
};

ਬਣਤਰ vk_mat_vec_p021_ਧੱਕੋ_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t nchannels_x;
    uint32_t nchannels_y;
    uint32_t b_ਵਿੱਥ;
    uint32_t d_ਵਿੱਥ;
    uint32_t fusion_flags;
};

ਬਣਤਰ vk_mat_vec_nc_ਧੱਕੋ_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t ਪੰਕਤੀ_stride_x;
    uint32_t channel_stride_x;
    uint32_t channel_stride_y;
    uint32_t channel_x_ਭਾਜਕ;
    uint32_t ne12;
    uint32_t b_ਵਿੱਥ;
    uint32_t d_ਵਿੱਥ;
    uint32_t nb03;
    uint32_t nb13;
    uint32_t nb23;
    uint32_t fusion_flags;
};

ਬਣਤਰ vk_mat_mat_id_ਧੱਕੋ_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t nei1; uint32_t nbi1; uint32_t ne11;
    uint32_t n_experts;
    uint32_t ਪੰਕਤੀ_ids_ਵਿੱਥ;
};

ਬਣਤਰ vk_mat_vec_id_ਧੱਕੋ_constants {
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

ਬਣਤਰ vk_flash_attn_ਧੱਕੋ_constants {
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

    ਦਸ਼ਮਲਵ scale;
    ਦਸ਼ਮਲਵ max_bias;
    ਦਸ਼ਮਲਵ logit_softcap;

    uint32_t ਨਕਾਬ_n_ਸਿਰ_log2;
    ਦਸ਼ਮਲਵ m0;
    ਦਸ਼ਮਲਵ m1;

    uint32_t gqa_ਅਨੁਪਾਤ;
    uint32_t ਵੰਡੋ_kv;
    uint32_t k_ਅੰਕ;
};

static_assert(ਆਕਾਰ(vk_flash_attn_ਧੱਕੋ_constants) <= 128, "sizeof(vk_flash_attn_push_constants) must be <= 128");

ਬਣਤਰ vk_fa_xe_opt_ਧੱਕੋ_constants {
    uint32_t kv_seq_len;
    uint32_t activation_ਲੰਮਾਈ;
    uint32_t q_ਸਿਰ;
    uint32_t kv_ਸਿਰ;
    uint32_t qk_ਅਨੁਪਾਤ;
    uint32_t qk_sub_groups;
    uint32_t flag;
    uint32_t nbkv_tok;
    uint32_t nbkv_ਸਿਰ;
    uint32_t batch_stride_q;
    uint32_t batch_stride_k;
    uint32_t batch_stride_v;
    uint32_t batch_stride_m;
    uint32_t batch_stride_o;
    ਦਸ਼ਮਲਵ softmax_scale;
};

ਬਣਤਰ vk_op_ਧੱਕੋ_constants {
    uint32_t KX;
    uint32_t KY;
    ਦਸ਼ਮਲਵ param1;
    ਦਸ਼ਮਲਵ param2;
    ਦਸ਼ਮਲਵ param3;
    ਦਸ਼ਮਲਵ param4;
};

ਬਣਤਰ vk_op_fwht_ਧੱਕੋ_constants {
    uint32_t n_ਪੰਕਤੀਆਂ;
    uint32_t src_ਵਿੱਥ;
    uint32_t dst_ਵਿੱਥ;
    ਦਸ਼ਮਲਵ scale;
};

ਬਣਤਰ vk_op_dsv4_hc_comb_ਧੱਕੋ_constants {
    uint32_t n_ਟੋਕਨ;

    uint32_t nbm0; uint32_t nbm1;
    uint32_t nbs0;
    uint32_t nbb0;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t m_ਵਿੱਥ;
    uint32_t s_ਵਿੱਥ;
    uint32_t b_ਵਿੱਥ;
    uint32_t d_ਵਿੱਥ;

    ਦਸ਼ਮਲਵ eps;
    uint32_t n_iter;
};

ਬਣਤਰ vk_op_dsv4_hc_pre_ਧੱਕੋ_constants {
    uint32_t n_embd;
    uint32_t n_ਟੋਕਨ;

    uint32_t nbx0; uint32_t nbx1; uint32_t nbx2;
    uint32_t nbw0; uint32_t nbw1; uint32_t nbw2;
    uint32_t nbd0; uint32_t nbd1;

    uint32_t x_ਵਿੱਥ;
    uint32_t w_ਵਿੱਥ;
    uint32_t d_ਵਿੱਥ;

    ਦਸ਼ਮਲਵ scale;
};

ਬਣਤਰ vk_op_dsv4_hc_post_ਧੱਕੋ_constants {
    uint32_t n_embd;
    uint32_t n_ਟੋਕਨ;

    uint32_t nbx0; uint32_t nbx1;
    uint32_t nbr0; uint32_t nbr1; uint32_t nbr2;
    uint32_t nbp0; uint32_t nbp1;
    uint32_t nbc0; uint32_t nbc1; uint32_t nbc2;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t x_ਵਿੱਥ;
    uint32_t r_ਵਿੱਥ;
    uint32_t p_ਵਿੱਥ;
    uint32_t c_ਵਿੱਥ;
    uint32_t d_ਵਿੱਥ;

    uint32_t gate;
    ਦਸ਼ਮਲਵ    gate_scale_ਵਿੱਚ;
    ਦਸ਼ਮਲਵ    gate_scale_ਬਾਹਰਲਾ;
};

ਬਣਤਰ vk_op_ਗਿਣਤੀ_experts_ਧੱਕੋ_constants {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t nb00;
    uint32_t nb01;
    uint32_t a_ਵਿੱਥ;
    uint32_t n_experts;
    uint32_t ne00mp;
    uint32_t ne00L;
    uint32_t ਪੰਕਤੀ_tile_ਮਾਪ;
    uint32_t ਪੰਕਤੀ_ids_ਵਿੱਥ;
};

ਬਣਤਰ vk_op_glu_ਧੱਕੋ_constants {
    uint32_t N;
    uint32_t ne00;
    uint32_t ne20;
    uint32_t ਬਹੁਲਕ;  // 0: default, 1: swapped, 2: split
    ਦਸ਼ਮਲਵ alpha; // for swiglu_oai
    ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ;
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

static_assert(ਆਕਾਰ(vk_op_glu_ਧੱਕੋ_constants) <= 128, "sizeof(vk_op_glu_push_constants) must be <= 128");

ਬਣਤਰ vk_op_unary_ਧੱਕੋ_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t misalign_offsets;
    ਦਸ਼ਮਲਵ param1; ਦਸ਼ਮਲਵ param2; ਦਸ਼ਮਲਵ param3; ਦਸ਼ਮਲਵ param4;
    uint32_t ne0_012mp; uint32_t ne0_01mp; uint32_t ne0_0mp; uint32_t ne0_Ls;
    uint32_t ne1_012mp; uint32_t ne1_01mp; uint32_t ne1_0mp; uint32_t ne1_Ls;
};

static_assert(ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants) <= 128, "sizeof(vk_op_unary_push_constants) must be <= 128");

ਸਥਿਰ vk_op_unary_ਧੱਕੋ_constants vk_op_unary_ਧੱਕੋ_constants_init(ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * dst, int64_t ne = 0) {
    GGML_ASSERT(ne != 0 || (ggml_nelements(src0) == ggml_nelements(dst)));
    ne = ne != 0 ? ne : ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_unary_ਧੱਕੋ_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_ਮਾਪ(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_ਮਾਪ(dst->type);
    p.ne10 = (uint32_t)dst->ne[0];
    p.ne11 = (uint32_t)dst->ne[1];
    p.ne12 = (uint32_t)dst->ne[2];
    p.ne13 = (uint32_t)dst->ne[3];
    p.nb10 = (uint32_t)(dst->nb[0] / dst_tsize);
    p.nb11 = (uint32_t)(dst->nb[1] / dst_tsize);
    p.nb12 = (uint32_t)(dst->nb[2] / dst_tsize);
    p.nb13 = (uint32_t)(dst->nb[3] / dst_tsize);

    ਵਾਪਸ p; // offsets are initialized later in ggml_vk_op
}

ਬਣਤਰ vk_op_pad_ਧੱਕੋ_constants {
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

ਸਥਿਰ vk_op_pad_ਧੱਕੋ_constants vk_op_pad_ਧੱਕੋ_constants_init(ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * dst) {
    int64_t ne = ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_pad_ਧੱਕੋ_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_ਮਾਪ(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_ਮਾਪ(dst->type);
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

    ਵਾਪਸ p; // fastdiv values and offsets are initialized later in ggml_vk_op
}

ਸਥਿਰ ਖਾਲੀ init_fastdiv_ਮੁੱਲਾਂ(uint32_t d, uint32_t &mp, uint32_t &L)
{
    // compute L = ceil(log2(d));
    L = 0;
    ਜਦੋਂ (L < 32 && (uint32_t{1} << L) < d) {
        L++;
    }

    mp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
}

ਸਥਿਰ uint32_t pack_fastdiv_L(uint32_t L0, uint32_t L1, uint32_t L2) {
    ਵਾਪਸ L0 | (L1 << 8) | (L2 << 16);
}

ਸਾਂਚਾ <typename T> ਖਾਲੀ init_pushconst_fastdiv(T &p) {
    GGML_ਅਣਵਰਤਿਆ(p);
    static_assert(!std::is_const<T>::ਮੁੱਲ, "unexpected type");
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_fastdiv(vk_op_unary_ਧੱਕੋ_constants &p) {
    // Compute magic values to divide by these six numbers.
    uint32_t ne0_012L;
    uint32_t ne0_01L;
    uint32_t ne0_0L;
    uint32_t ne1_012L;
    uint32_t ne1_01L;
    uint32_t ne1_0L;

    init_fastdiv_ਮੁੱਲਾਂ(p.ne02*p.ne01*p.ne00,  p.ne0_012mp,    ne0_012L);
    init_fastdiv_ਮੁੱਲਾਂ(p.ne01*p.ne00,         p.ne0_01mp,     ne0_01L);
    init_fastdiv_ਮੁੱਲਾਂ(p.ne00,                p.ne0_0mp,      ne0_0L);
    init_fastdiv_ਮੁੱਲਾਂ(p.ne12*p.ne11*p.ne10,  p.ne1_012mp,    ne1_012L);
    init_fastdiv_ਮੁੱਲਾਂ(p.ne11*p.ne10,         p.ne1_01mp,     ne1_01L);
    init_fastdiv_ਮੁੱਲਾਂ(p.ne10,                p.ne1_0mp,      ne1_0L);

    p.ne0_Ls = pack_fastdiv_L(ne0_012L, ne0_01L, ne0_0L);
    p.ne1_Ls = pack_fastdiv_L(ne1_012L, ne1_01L, ne1_0L);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_fastdiv(vk_op_glu_ਧੱਕੋ_constants &p) {
    // GLU linearizes over dst, then uses dst coordinates for src0/src1.
    init_fastdiv_ਮੁੱਲਾਂ(p.ne22*p.ne21*p.ne20,  p.ne2_012mp,    p.ne2_012L);
    init_fastdiv_ਮੁੱਲਾਂ(p.ne21*p.ne20,         p.ne2_01mp,     p.ne2_01L);
    init_fastdiv_ਮੁੱਲਾਂ(p.ne20,                p.ne2_0mp,      p.ne2_0L);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_fastdiv(vk_op_ਗਿਣਤੀ_experts_ਧੱਕੋ_constants &p) {
    init_fastdiv_ਮੁੱਲਾਂ(p.ne00, p.ne00mp, p.ne00L);
}

ਬਣਤਰ vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23;
    uint32_t misalign_offsets;
    ਦਸ਼ਮਲਵ param1; ਦਸ਼ਮਲਵ param2; int32_t param3;
};

ਬਣਤਰ vk_op_concat_ਧੱਕੋ_constants : vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants {};

static_assert(ਆਕਾਰ(vk_op_concat_ਧੱਕੋ_constants) == ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants));

static_assert(std::is_standard_layout_v<vk_op_concat_ਧੱਕੋ_constants>);

ਬਣਤਰ vk_op_multi_ਜੋੜੋ_ਧੱਕੋ_constants {
    // shape for dst
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23;

    // strides for srcs+dst
    uint32_t nb[MAX_PARAMETER_ਗਿਣਤੀ][4];

    uint32_t rms_partials;
};

static_assert(MAX_PARAMETER_ਗਿਣਤੀ == 12);

static_assert(ਆਕਾਰ(vk_op_multi_ਜੋੜੋ_ਧੱਕੋ_constants) <= 256);

ਬਣਤਰ vk_op_topk_moe_ਧੱਕੋ_constants {
    uint32_t n_ਪੰਕਤੀਆਂ;
    uint32_t n_experts_ਧੱਕੋ;
    uint32_t n_expert_ਵਰਤਿਆ;
    ਦਸ਼ਮਲਵ clamp_min;
    ਦਸ਼ਮਲਵ clamp_max;
    uint32_t gating_func;
    uint32_t has_bias;
    uint32_t with_norm;
    ਦਸ਼ਮਲਵ output_scale;
    ਦਸ਼ਮਲਵ output_bias;
};

ਬਣਤਰ vk_op_ਜੋੜੋ_id_ਧੱਕੋ_constants {
    uint32_t ne0;
    uint32_t ne1;
    uint32_t s01;
    uint32_t s02;
    uint32_t s11;
    uint32_t s21;
};

ਬਣਤਰ vk_op_diag_ਨਕਾਬ_ਧੱਕੋ_constants {
    uint32_t ncols;
    uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_channel;
    int32_t n_past;
};

ਬਣਤਰ vk_op_rope_ਧੱਕੋ_constants {
    uint32_t rope_ਬਹੁਲਕ;
    uint32_t nrows;
    uint32_t n_dims;
    uint32_t n_offs;
    ਦਸ਼ਮਲਵ freq_scale;
    ਦਸ਼ਮਲਵ freq_ਅਧਾਰ;
    ਦਸ਼ਮਲਵ ext_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ attn_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ corr_dims[2];
    ਦਸ਼ਮਲਵ theta_scale;
    uint32_t has_ff;
    int32_t sections[4];
    uint32_t is_imrope;
    uint32_t is_ਪਿੱਠ;
    uint32_t ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t a_ਵਿੱਥ;
    uint32_t d_ਵਿੱਥ;
};

static_assert(ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants) <= 128, "sizeof(vk_op_rope_push_constants) must be <= 128");

ਬਣਤਰ vk_op_rms_norm_mul_rope_ਧੱਕੋ_constants {
    vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants bin;
    vk_op_rope_ਧੱਕੋ_constants rope;
};

ਬਣਤਰ vk_op_soft_max_ਧੱਕੋ_constants {
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
    ਦਸ਼ਮਲਵ scale;
    ਦਸ਼ਮਲਵ max_bias;
    ਦਸ਼ਮਲਵ m0;
    ਦਸ਼ਮਲਵ m1;
    uint32_t n_ਸਿਰ_log2;
    uint32_t nrows_x;
    uint32_t has_sinks;
};

ਬਣਤਰ vk_op_argsort_ਧੱਕੋ_constants {
    uint32_t ncols;
    uint32_t ncols_padded;
    uint32_t ncols_padded_log2;
    uint32_t nrows;
    uint32_t ਕ੍ਰਮ;
    uint32_t outer_ਸ਼ੁਰੂ;
    uint32_t outer_ਸਿਰਾ;
    uint32_t inner_ਸ਼ੁਰੂ;
    uint32_t inner_ਸਿਰਾ;
};

ਬਣਤਰ vk_op_topk_ਧੱਕੋ_constants {
    uint32_t orig_ncols;
    uint32_t ncols_ਨਿਵੇਸ਼;
    uint32_t ncols_output;
    uint32_t k;
    uint32_t nrows;
    uint32_t ਪਹਿਲਾ_pass;
    uint32_t ਆਖਰੀ_pass;
};

ਬਣਤਰ vk_op_topk_radix_ਧੱਕੋ_constants {
    uint32_t ncols;
    uint32_t k;
    uint32_t nrows;
    uint32_t n_tps;    // QSA only
    uint32_t n_ਖੰਡਾਂ; // QSA only
    uint32_t n_ਧਾਰਾ; // QSA only
};

ਬਣਤਰ vk_op_im2col_ਧੱਕੋ_constants {
    uint64_t dst_addr;
    uint32_t batch_ਵਿੱਥ; uint32_t ਵਿੱਥ_delta;
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

ਬਣਤਰ vk_op_im2col_3d_ਧੱਕੋ_constants {
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

ਬਣਤਰ vk_op_timestep_embedding_ਧੱਕੋ_constants {
    uint32_t nb1;
    uint32_t dim;
    uint32_t max_period;
};

ਬਣਤਰ vk_op_col2im_1d_ਧੱਕੋ_constants {
    uint32_t T_ਬਾਹਰਲਾ;
    uint32_t OC;
    uint32_t K_OC;
    uint32_t T_ਵਿੱਚ;
    uint32_t K;
    int32_t  stride;
    int32_t  p0;
};

ਬਣਤਰ vk_op_conv_transpose_1d_ਧੱਕੋ_constants {
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

ਬਣਤਰ vk_op_snake_ਧੱਕੋ_constants {
    uint32_t ne0;
    uint32_t ne1;
};

ਬਣਤਰ vk_op_pool1d_ਧੱਕੋ_constants {
    uint32_t IL;
    uint32_t OL;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0;
    int32_t s0;
    int32_t p0;
};

ਬਣਤਰ vk_op_pool2d_ਧੱਕੋ_constants {
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0; int32_t k1;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
};

ਬਣਤਰ vk_op_rwkv_wkv6_ਧੱਕੋ_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

ਬਣਤਰ vk_op_rwkv_wkv7_ਧੱਕੋ_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

ਬਣਤਰ vk_op_gated_ਰੇਖਿਕ_attn_ਧੱਕੋ_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
    ਦਸ਼ਮਲਵ scale;
};

ਬਣਤਰ vk_op_lightning_indexer_ਧੱਕੋ_constants {
    uint32_t n_kv;
    uint32_t n_heads;
    uint32_t n_ਟੋਕਨ;
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

static_assert(ਆਕਾਰ(vk_op_lightning_indexer_ਧੱਕੋ_constants) <= 128);

ਬਣਤਰ vk_op_gated_delta_net_ਧੱਕੋ_constants {
    uint32_t H;
    uint32_t n_ਟੋਕਨ;
    uint32_t n_seqs;
    uint32_t s_off;
    uint32_t sq1, sq2, sq3;
    uint32_t sv1, sv2, sv3;
    uint32_t sb1, sb2, sb3;
    uint32_t neq1, rq3;
    ਦਸ਼ਮਲਵ scale;
    uint32_t K;
};

ਬਣਤਰ vk_op_ssm_ਜਾਂਚ_ਧੱਕੋ_constants {
    uint32_t nb02, nb03, nb12, nb13;
    uint32_t nb21, nb22, nb31;
    uint32_t nb42, nb43, nb52, nb53;
    uint32_t s_off;
    uint32_t n_ਸਿਰ, d_ਸਿਰ, n_ਸਮੂਹ, n_tok;
    uint32_t n_seq, K;
};

ਬਣਤਰ vk_op_ssm_conv_ਧੱਕੋ_constants {
    uint32_t nb01, nb02;
    uint32_t nb11;
    uint32_t dst_nb0, dst_nb1, dst_nb2;
    uint32_t nc, ncs, nr, n_t, n_s;
};

ਬਣਤਰ vk_op_conv2d_ਧੱਕੋ_constants {
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

    uint32_t knl_ਵਿੱਥ;
    uint32_t src_ਵਿੱਥ;
    uint32_t dst_ਵਿੱਥ;
};

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_fastdiv(vk_op_conv2d_ਧੱਕੋ_constants &p) {
    // Compute magic values to divide by OW, OW*OH
    init_fastdiv_ਮੁੱਲਾਂ(p.OW,       p.OWmp,    p.OWL);
    init_fastdiv_ਮੁੱਲਾਂ(p.OW*p.OH,  p.OWOHmp,  p.OWOHL);
}

ਬਣਤਰ vk_op_conv3d_ਧੱਕੋ_constants {
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

    uint32_t knl_ਵਿੱਥ;
    uint32_t src_ਵਿੱਥ;
    uint32_t dst_ਵਿੱਥ;
};

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_fastdiv(vk_op_conv3d_ਧੱਕੋ_constants &p) {
    init_fastdiv_ਮੁੱਲਾਂ(p.OW,             p.OWmp,     p.OWL);
    init_fastdiv_ਮੁੱਲਾਂ(p.OW*p.OH,        p.OWOHmp,   p.OWOHL);
    init_fastdiv_ਮੁੱਲਾਂ(p.OW*p.OH*p.OD,   p.OWOHODmp, p.OWOHODL);
}

ਬਣਤਰ vk_op_conv2d_dw_ਧੱਕੋ_constants {
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

ਬਣਤਰ vk_op_upscale_ਧੱਕੋ_constants {
    uint32_t ne; uint32_t a_ਵਿੱਥ; uint32_t d_ਵਿੱਥ;
    uint32_t ne00; uint32_t ne01;
    uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13;
    ਦਸ਼ਮਲਵ sf0; ਦਸ਼ਮਲਵ sf1; ਦਸ਼ਮਲਵ sf2; ਦਸ਼ਮਲਵ sf3;
    ਦਸ਼ਮਲਵ pixel_ਵਿੱਥ;
};

ਬਣਤਰ vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants
{
    uint32_t n_ਖਾਨੇ_ਗਿਣਤੀ;
    uint32_t ne01, ne02;
    uint32_t nb01, nb02, nb03;
    uint32_t nb11, nb12, nb13;
    ਦਸ਼ਮਲਵ ਭਾਰ;
    uint32_t misalign_offsets;
    uint32_t ne0_12mp, ne0_12L;
    uint32_t ne0_1mp, ne0_1L;
};

ਸਥਿਰ vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants_init(ਅਚਲ ggml_tensor * src, ਅਚਲ ggml_tensor * dst, int64_t n_ਖਾਨੇ_ਗਿਣਤੀ) {
    uint32_t type_ਮਾਪ = (uint32_t)ggml_type_ਮਾਪ(src->type);
    vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants p = {};
    p.n_ਖਾਨੇ_ਗਿਣਤੀ = (uint32_t)n_ਖਾਨੇ_ਗਿਣਤੀ;
    p.ne01 = (uint32_t)src->ne[1];
    p.ne02 = (uint32_t)src->ne[2];
    p.nb01 = (uint32_t)src->nb[1] / type_ਮਾਪ;
    p.nb02 = (uint32_t)src->nb[2] / type_ਮਾਪ;
    p.nb03 = (uint32_t)src->nb[3] / type_ਮਾਪ;
    p.nb11 = (uint32_t)dst->nb[1] / type_ਮਾਪ;
    p.nb12 = (uint32_t)dst->nb[2] / type_ਮਾਪ;
    p.nb13 = (uint32_t)dst->nb[3] / type_ਮਾਪ;
    p.ਭਾਰ = 1.0f;
    ਵਾਪਸ p;
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_fastdiv(vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants &p) {
    init_fastdiv_ਮੁੱਲਾਂ(p.ne01*p.ne02, p.ne0_12mp, p.ne0_12L);
    init_fastdiv_ਮੁੱਲਾਂ(p.ne01,        p.ne0_1mp,  p.ne0_1L);
}

ਬਣਤਰ vk_quantize_q8_1_ਧੱਕੋ_constants {
    uint32_t ne;
    uint32_t ਅੰਕ_ਖੰਡਾਂ;
};

ਬਣਤਰ vk_op_flash_attn_ਵੰਡੋ_k_reduce_ਧੱਕੋ_constants {
    uint32_t D;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t k_ਅੰਕ;
    uint32_t sinks;
};

ਬਣਤਰ vk_op_flash_attn_ਨਕਾਬ_opt_ਧੱਕੋ_constants {
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

ਬਣਤਰ vk_op_flash_attn_sparse_compact_ਧੱਕੋ_constants {
    uint32_t KV;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nbm1;
    uint32_t nbm2;
    uint32_t nbm3;
    uint32_t n_kv_max;
};

ਸਾਂਚਾ <typename T> ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, T &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    GGML_ਅਣਵਰਤਿਆ(p);
    GGML_ਅਣਵਰਤਿਆ(src0);
    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
    GGML_ਅਣਵਰਤਿਆ(dst);
    static_assert(!std::is_const<T>::ਮੁੱਲ, "unexpected type");
    GGML_ASSERT(!src0 || get_misalign_ਬਾਈਟਾਂ(ctx, src0) == 0);
    GGML_ASSERT(!src1 || get_misalign_ਬਾਈਟਾਂ(ctx, src1) == 0);
    GGML_ASSERT(!src2 || get_misalign_ਬਾਈਟਾਂ(ctx, src2) == 0);
    GGML_ASSERT(!src3 || get_misalign_ਬਾਈਟਾਂ(ctx, src3) == 0);
    GGML_ASSERT(!dst  || get_misalign_ਬਾਈਟਾਂ(ctx, dst) == 0);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_p021_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    ਅਚਲ uint32_t b_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src1) / ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / ggml_type_ਮਾਪ(dst->type);

    p.b_ਵਿੱਥ = b_ਵਿੱਥ;
    p.d_ਵਿੱਥ = d_ਵਿੱਥ;

    GGML_ਅਣਵਰਤਿਆ(src0);
    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_nc_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    ਅਚਲ uint32_t b_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src1) / ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / ggml_type_ਮਾਪ(dst->type);

    p.b_ਵਿੱਥ = b_ਵਿੱਥ;
    p.d_ਵਿੱਥ = d_ਵਿੱਥ;

    GGML_ਅਣਵਰਤਿਆ(src0);
    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_fwht_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    p.src_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    p.dst_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst)  / ggml_type_ਮਾਪ(dst->type);

    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_comb_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    p.m_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    p.s_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src1) / ggml_type_ਮਾਪ(src1->type);
    p.b_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src2) / ggml_type_ਮਾਪ(src2->type);
    p.d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst)  / ggml_type_ਮਾਪ(dst->type);

    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_pre_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    p.x_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    p.w_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src1) / ggml_type_ਮਾਪ(src1->type);
    p.d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst)  / ggml_type_ਮਾਪ(dst->type);

    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_post_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    p.x_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    p.r_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src1) / ggml_type_ਮਾਪ(src1->type);
    p.p_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src2) / ggml_type_ਮਾਪ(src2->type);
    p.c_ਵਿੱਥ = src3 ? get_misalign_ਬਾਈਟਾਂ(ctx, src3) / ggml_type_ਮਾਪ(src3->type) : 0;
    p.d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst)  / ggml_type_ਮਾਪ(dst->type);
}

ਸਾਂਚਾ <typename T> size_t ਧੱਕੋ_constant_ਮਾਪ(ਅਚਲ T &t) {
    static_assert(std::is_class<T>::ਮੁੱਲ, "T must be a struct/class");
    GGML_ਅਣਵਰਤਿਆ(t);
    ਵਾਪਸ ਆਕਾਰ(T);
}

ਸਾਂਚਾ <typename T> size_t ਧੱਕੋ_constant_ਮਾਪ(ਅਚਲ std::ਸਦਿਸ਼<T> &t) {
    GGML_ਅਣਵਰਤਿਆ(t);
    ਵਾਪਸ ਆਕਾਰ(T) * t.ਮਾਪ();
}

ਸਾਂਚਾ <typename T, uint32_t N> size_t ਧੱਕੋ_constant_ਮਾਪ(ਅਚਲ std::array<T, N> &t) {
    GGML_ਅਣਵਰਤਿਆ(t);
    ਵਾਪਸ ਆਕਾਰ(T) * N;
}

ਸਾਂਚਾ <typename T> ਅਚਲ T *ਧੱਕੋ_constant_ਅੰਕੜੇ(ਅਚਲ T &t) {
    static_assert(std::is_class<T>::ਮੁੱਲ, "T must be a struct/class");
    ਵਾਪਸ &t;
}

ਸਾਂਚਾ <typename T> ਅਚਲ T *ਧੱਕੋ_constant_ਅੰਕੜੇ(ਅਚਲ std::ਸਦਿਸ਼<T> &t) {
    ਵਾਪਸ t.ਅੰਕੜੇ();
}

ਸਾਂਚਾ <typename T, uint32_t N> ਅਚਲ T *ਧੱਕੋ_constant_ਅੰਕੜੇ(ਅਚਲ std::array<T, N> &t) {
    ਵਾਪਸ t.ਅੰਕੜੇ();
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_unary_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    ਅਚਲ uint32_t a_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / ggml_type_ਮਾਪ(dst->type);

    p.misalign_offsets = (a_ਵਿੱਥ << 16) | d_ਵਿੱਥ;

    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_glu_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    ਅਚਲ uint32_t a_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t b_ਵਿੱਥ = src1 ? get_misalign_ਬਾਈਟਾਂ(ctx, src1) / ggml_type_ਮਾਪ(src1->type) : a_ਵਿੱਥ;
    ਅਚਲ uint32_t d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / ggml_type_ਮਾਪ(dst->type);

    GGML_ASSERT(a_ਵਿੱਥ < (1u << 8));
    GGML_ASSERT(b_ਵਿੱਥ < (1u << 8));
    GGML_ASSERT(d_ਵਿੱਥ < (1u << 8));

    p.misalign_offsets = (a_ਵਿੱਥ << 16) | (b_ਵਿੱਥ << 8) | d_ਵਿੱਥ;

    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    ਅਚਲ uint32_t a_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / ggml_type_ਮਾਪ(dst->type);

    p.misalign_offsets = (a_ਵਿੱਥ << 16) | d_ਵਿੱਥ;

    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_pad_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    ਅਚਲ uint32_t a_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / ggml_type_ਮਾਪ(dst->type);

    p.misalign_offsets = (a_ਵਿੱਥ << 16) | d_ਵਿੱਥ;

    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv2d_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    p.src_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src1) / ggml_type_ਮਾਪ(src1->type);
    p.dst_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst)  / ggml_type_ਮਾਪ(dst->type);

    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv3d_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    p.src_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src1) / ggml_type_ਮਾਪ(src1->type);
    p.dst_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst)  / ggml_type_ਮਾਪ(dst->type);

    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_im2col_3d_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    ਅਚਲ uint32_t a_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src1) / ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / ggml_type_ਮਾਪ(dst->type);

    p.misalign_offsets = (a_ਵਿੱਥ << 16) | d_ਵਿੱਥ;

    GGML_ਅਣਵਰਤਿਆ(src0);
    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    ਅਚਲ uint32_t a_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t b_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src1) / ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / ggml_type_ਮਾਪ(dst->type);

    GGML_ASSERT(a_ਵਿੱਥ <= 0xFFFF);
    GGML_ASSERT(b_ਵਿੱਥ <= 0xFF);
    GGML_ASSERT(d_ਵਿੱਥ <= 0xFF);

    p.misalign_offsets = (a_ਵਿੱਥ << 16) | (b_ਵਿੱਥ << 8) | d_ਵਿੱਥ;

    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_concat_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    ਅਚਲ uint32_t unit_ਮਾਪ = ggml_vk_concat_unit_ਮਾਪ(dst->type);
    ਅਚਲ uint32_t a_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / unit_ਮਾਪ;
    ਅਚਲ uint32_t b_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src1) / unit_ਮਾਪ;
    ਅਚਲ uint32_t d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / unit_ਮਾਪ;

    p.misalign_offsets = (a_ਵਿੱਥ << 16) | (b_ਵਿੱਥ << 8) | d_ਵਿੱਥ;

    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_upscale_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    ਅਚਲ uint32_t a_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / ggml_type_ਮਾਪ(dst->type);

    p.a_ਵਿੱਥ = a_ਵਿੱਥ;
    p.d_ਵਿੱਥ = d_ਵਿੱਥ;

    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ ਖਾਲੀ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_rope_ਧੱਕੋ_constants &p, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst) {
    p.a_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, src0) / ggml_type_ਮਾਪ(src0->type);
    p.d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, dst)  / ggml_type_ਮਾਪ(dst->type);

    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(src2);
    GGML_ਅਣਵਰਤਿਆ(src3);
}

ਸਥਿਰ vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants ggml_vk_rms_norm_ਧੱਕੋ_constants(
        ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * dst,
        ਦਸ਼ਮਲਵ eps, uint32_t ਅੰਕ_partials) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ਵਾਪਸ {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        eps, 0.0f, (int32_t)ਅੰਕ_partials,
    };
}

