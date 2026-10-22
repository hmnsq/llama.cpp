#సూచన once
#అన్తర్భావ "ggml-vulkan-types.h"

uint32_t get_misalign_బైట్లు(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * t);

uint32_t ggml_vk_concat_unit_కొలత(ggml_type type);

నిర్మాణం vk_mat_mat_నెట్టు_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t పునాది_work_సమూహం_z; uint32_t అంకె_batches;
    uint32_t k_విభజించు;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
    uint32_t padded_N;
};

నిర్మాణం vk_mat_vec_నెట్టు_constants {
    uint32_t ncols;
    uint32_t stride_a;
    uint32_t stride_b;
    uint32_t stride_d;
    uint32_t batch_stride_a;
    uint32_t batch_stride_b;
    uint32_t batch_stride_d;
    uint32_t fusion_flags;
    uint32_t పునాది_work_సమూహం_y;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t broadcast2;
    uint32_t broadcast3;
};

నిర్మాణం vk_mat_vec_p021_నెట్టు_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t nchannels_x;
    uint32_t nchannels_y;
    uint32_t b_స్థానభ్రంశం;
    uint32_t d_స్థానభ్రంశం;
    uint32_t fusion_flags;
};

నిర్మాణం vk_mat_vec_nc_నెట్టు_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t అడ్డువరుస_stride_x;
    uint32_t channel_stride_x;
    uint32_t channel_stride_y;
    uint32_t channel_x_భాజకం;
    uint32_t ne12;
    uint32_t b_స్థానభ్రంశం;
    uint32_t d_స్థానభ్రంశం;
    uint32_t nb03;
    uint32_t nb13;
    uint32_t nb23;
    uint32_t fusion_flags;
};

నిర్మాణం vk_mat_mat_id_నెట్టు_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t nei1; uint32_t nbi1; uint32_t ne11;
    uint32_t n_experts;
    uint32_t అడ్డువరుస_ids_స్థానభ్రంశం;
};

నిర్మాణం vk_mat_vec_id_నెట్టు_constants {
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

నిర్మాణం vk_flash_attn_నెట్టు_constants {
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

    దశాంశం scale;
    దశాంశం max_bias;
    దశాంశం logit_softcap;

    uint32_t ముసుగు_n_తల_log2;
    దశాంశం m0;
    దశాంశం m1;

    uint32_t gqa_నిష్పత్తి;
    uint32_t విభజించు_kv;
    uint32_t k_అంకె;
};

static_assert(పరిమాణం(vk_flash_attn_నెట్టు_constants) <= 128, "sizeof(vk_flash_attn_push_constants) must be <= 128");

నిర్మాణం vk_fa_xe_opt_నెట్టు_constants {
    uint32_t kv_seq_len;
    uint32_t activation_నిడివి;
    uint32_t q_తల;
    uint32_t kv_తల;
    uint32_t qk_నిష్పత్తి;
    uint32_t qk_sub_groups;
    uint32_t flag;
    uint32_t nbkv_tok;
    uint32_t nbkv_తల;
    uint32_t batch_stride_q;
    uint32_t batch_stride_k;
    uint32_t batch_stride_v;
    uint32_t batch_stride_m;
    uint32_t batch_stride_o;
    దశాంశం softmax_scale;
};

నిర్మాణం vk_op_నెట్టు_constants {
    uint32_t KX;
    uint32_t KY;
    దశాంశం param1;
    దశాంశం param2;
    దశాంశం param3;
    దశాంశం param4;
};

నిర్మాణం vk_op_fwht_నెట్టు_constants {
    uint32_t n_అడ్డువరుసలు;
    uint32_t src_స్థానభ్రంశం;
    uint32_t dst_స్థానభ్రంశం;
    దశాంశం scale;
};

నిర్మాణం vk_op_dsv4_hc_comb_నెట్టు_constants {
    uint32_t n_టోకెన్లు;

    uint32_t nbm0; uint32_t nbm1;
    uint32_t nbs0;
    uint32_t nbb0;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t m_స్థానభ్రంశం;
    uint32_t s_స్థానభ్రంశం;
    uint32_t b_స్థానభ్రంశం;
    uint32_t d_స్థానభ్రంశం;

    దశాంశం eps;
    uint32_t n_iter;
};

నిర్మాణం vk_op_dsv4_hc_pre_నెట్టు_constants {
    uint32_t n_embd;
    uint32_t n_టోకెన్లు;

    uint32_t nbx0; uint32_t nbx1; uint32_t nbx2;
    uint32_t nbw0; uint32_t nbw1; uint32_t nbw2;
    uint32_t nbd0; uint32_t nbd1;

    uint32_t x_స్థానభ్రంశం;
    uint32_t w_స్థానభ్రంశం;
    uint32_t d_స్థానభ్రంశం;

    దశాంశం scale;
};

నిర్మాణం vk_op_dsv4_hc_post_నెట్టు_constants {
    uint32_t n_embd;
    uint32_t n_టోకెన్లు;

    uint32_t nbx0; uint32_t nbx1;
    uint32_t nbr0; uint32_t nbr1; uint32_t nbr2;
    uint32_t nbp0; uint32_t nbp1;
    uint32_t nbc0; uint32_t nbc1; uint32_t nbc2;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t x_స్థానభ్రంశం;
    uint32_t r_స్థానభ్రంశం;
    uint32_t p_స్థానభ్రంశం;
    uint32_t c_స్థానభ్రంశం;
    uint32_t d_స్థానభ్రంశం;

    uint32_t gate;
    దశాంశం    gate_scale_లోపల;
    దశాంశం    gate_scale_బహిర్గతం;
};

నిర్మాణం vk_op_లెక్క_experts_నెట్టు_constants {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t nb00;
    uint32_t nb01;
    uint32_t a_స్థానభ్రంశం;
    uint32_t n_experts;
    uint32_t ne00mp;
    uint32_t ne00L;
    uint32_t అడ్డువరుస_tile_కొలత;
    uint32_t అడ్డువరుస_ids_స్థానభ్రంశం;
};

నిర్మాణం vk_op_glu_నెట్టు_constants {
    uint32_t N;
    uint32_t ne00;
    uint32_t ne20;
    uint32_t బహుళకం;  // 0: default, 1: swapped, 2: split
    దశాంశం alpha; // for swiglu_oai
    దశాంశం హద్దు;
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

static_assert(పరిమాణం(vk_op_glu_నెట్టు_constants) <= 128, "sizeof(vk_op_glu_push_constants) must be <= 128");

నిర్మాణం vk_op_unary_నెట్టు_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t misalign_offsets;
    దశాంశం param1; దశాంశం param2; దశాంశం param3; దశాంశం param4;
    uint32_t ne0_012mp; uint32_t ne0_01mp; uint32_t ne0_0mp; uint32_t ne0_Ls;
    uint32_t ne1_012mp; uint32_t ne1_01mp; uint32_t ne1_0mp; uint32_t ne1_Ls;
};

static_assert(పరిమాణం(vk_op_unary_నెట్టు_constants) <= 128, "sizeof(vk_op_unary_push_constants) must be <= 128");

స్థిర vk_op_unary_నెట్టు_constants vk_op_unary_నెట్టు_constants_init(స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * dst, int64_t ne = 0) {
    GGML_ASSERT(ne != 0 || (ggml_nelements(src0) == ggml_nelements(dst)));
    ne = ne != 0 ? ne : ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_unary_నెట్టు_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_కొలత(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_కొలత(dst->type);
    p.ne10 = (uint32_t)dst->ne[0];
    p.ne11 = (uint32_t)dst->ne[1];
    p.ne12 = (uint32_t)dst->ne[2];
    p.ne13 = (uint32_t)dst->ne[3];
    p.nb10 = (uint32_t)(dst->nb[0] / dst_tsize);
    p.nb11 = (uint32_t)(dst->nb[1] / dst_tsize);
    p.nb12 = (uint32_t)(dst->nb[2] / dst_tsize);
    p.nb13 = (uint32_t)(dst->nb[3] / dst_tsize);

    తిరిగి p; // offsets are initialized later in ggml_vk_op
}

నిర్మాణం vk_op_pad_నెట్టు_constants {
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

స్థిర vk_op_pad_నెట్టు_constants vk_op_pad_నెట్టు_constants_init(స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * dst) {
    int64_t ne = ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_pad_నెట్టు_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_కొలత(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_కొలత(dst->type);
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

    తిరిగి p; // fastdiv values and offsets are initialized later in ggml_vk_op
}

స్థిర శూన్య init_fastdiv_విలువలు(uint32_t d, uint32_t &mp, uint32_t &L)
{
    // compute L = ceil(log2(d));
    L = 0;
    అయితే (L < 32 && (uint32_t{1} << L) < d) {
        L++;
    }

    mp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
}

స్థిర uint32_t pack_fastdiv_L(uint32_t L0, uint32_t L1, uint32_t L2) {
    తిరిగి L0 | (L1 << 8) | (L2 << 16);
}

మూస <typename T> శూన్య init_pushconst_fastdiv(T &p) {
    GGML_వాడని(p);
    static_assert(!std::is_const<T>::విలువ, "unexpected type");
}

మూస <> అంతఃవరుస శూన్య init_pushconst_fastdiv(vk_op_unary_నెట్టు_constants &p) {
    // Compute magic values to divide by these six numbers.
    uint32_t ne0_012L;
    uint32_t ne0_01L;
    uint32_t ne0_0L;
    uint32_t ne1_012L;
    uint32_t ne1_01L;
    uint32_t ne1_0L;

    init_fastdiv_విలువలు(p.ne02*p.ne01*p.ne00,  p.ne0_012mp,    ne0_012L);
    init_fastdiv_విలువలు(p.ne01*p.ne00,         p.ne0_01mp,     ne0_01L);
    init_fastdiv_విలువలు(p.ne00,                p.ne0_0mp,      ne0_0L);
    init_fastdiv_విలువలు(p.ne12*p.ne11*p.ne10,  p.ne1_012mp,    ne1_012L);
    init_fastdiv_విలువలు(p.ne11*p.ne10,         p.ne1_01mp,     ne1_01L);
    init_fastdiv_విలువలు(p.ne10,                p.ne1_0mp,      ne1_0L);

    p.ne0_Ls = pack_fastdiv_L(ne0_012L, ne0_01L, ne0_0L);
    p.ne1_Ls = pack_fastdiv_L(ne1_012L, ne1_01L, ne1_0L);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_fastdiv(vk_op_glu_నెట్టు_constants &p) {
    // GLU linearizes over dst, then uses dst coordinates for src0/src1.
    init_fastdiv_విలువలు(p.ne22*p.ne21*p.ne20,  p.ne2_012mp,    p.ne2_012L);
    init_fastdiv_విలువలు(p.ne21*p.ne20,         p.ne2_01mp,     p.ne2_01L);
    init_fastdiv_విలువలు(p.ne20,                p.ne2_0mp,      p.ne2_0L);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_fastdiv(vk_op_లెక్క_experts_నెట్టు_constants &p) {
    init_fastdiv_విలువలు(p.ne00, p.ne00mp, p.ne00L);
}

నిర్మాణం vk_op_ద్విమాన_నెట్టు_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23;
    uint32_t misalign_offsets;
    దశాంశం param1; దశాంశం param2; int32_t param3;
};

నిర్మాణం vk_op_concat_నెట్టు_constants : vk_op_ద్విమాన_నెట్టు_constants {};

static_assert(పరిమాణం(vk_op_concat_నెట్టు_constants) == పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants));

static_assert(std::is_standard_layout_v<vk_op_concat_నెట్టు_constants>);

నిర్మాణం vk_op_multi_కూడు_నెట్టు_constants {
    // shape for dst
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23;

    // strides for srcs+dst
    uint32_t nb[MAX_PARAMETER_లెక్క][4];

    uint32_t rms_partials;
};

static_assert(MAX_PARAMETER_లెక్క == 12);

static_assert(పరిమాణం(vk_op_multi_కూడు_నెట్టు_constants) <= 256);

నిర్మాణం vk_op_topk_moe_నెట్టు_constants {
    uint32_t n_అడ్డువరుసలు;
    uint32_t n_experts_నెట్టు;
    uint32_t n_expert_వాడిన;
    దశాంశం clamp_min;
    దశాంశం clamp_max;
    uint32_t gating_func;
    uint32_t has_bias;
    uint32_t with_norm;
    దశాంశం output_scale;
    దశాంశం output_bias;
};

నిర్మాణం vk_op_కూడు_id_నెట్టు_constants {
    uint32_t ne0;
    uint32_t ne1;
    uint32_t s01;
    uint32_t s02;
    uint32_t s11;
    uint32_t s21;
};

నిర్మాణం vk_op_diag_ముసుగు_నెట్టు_constants {
    uint32_t ncols;
    uint32_t అడ్డువరుసలు_చొప్పున_channel;
    int32_t n_past;
};

నిర్మాణం vk_op_rope_నెట్టు_constants {
    uint32_t rope_బహుళకం;
    uint32_t nrows;
    uint32_t n_dims;
    uint32_t n_offs;
    దశాంశం freq_scale;
    దశాంశం freq_పునాది;
    దశాంశం ext_గుణకం;
    దశాంశం attn_గుణకం;
    దశాంశం corr_dims[2];
    దశాంశం theta_scale;
    uint32_t has_ff;
    int32_t sections[4];
    uint32_t is_imrope;
    uint32_t is_వెనుక;
    uint32_t సమితి_అడ్డువరుసలు_stride;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t a_స్థానభ్రంశం;
    uint32_t d_స్థానభ్రంశం;
};

static_assert(పరిమాణం(vk_op_rope_నెట్టు_constants) <= 128, "sizeof(vk_op_rope_push_constants) must be <= 128");

నిర్మాణం vk_op_rms_norm_mul_rope_నెట్టు_constants {
    vk_op_ద్విమాన_నెట్టు_constants bin;
    vk_op_rope_నెట్టు_constants rope;
};

నిర్మాణం vk_op_soft_max_నెట్టు_constants {
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
    దశాంశం scale;
    దశాంశం max_bias;
    దశాంశం m0;
    దశాంశం m1;
    uint32_t n_తల_log2;
    uint32_t nrows_x;
    uint32_t has_sinks;
};

నిర్మాణం vk_op_argsort_నెట్టు_constants {
    uint32_t ncols;
    uint32_t ncols_padded;
    uint32_t ncols_padded_log2;
    uint32_t nrows;
    uint32_t క్రమం;
    uint32_t outer_ప్రారంభం;
    uint32_t outer_చివర;
    uint32_t inner_ప్రారంభం;
    uint32_t inner_చివర;
};

నిర్మాణం vk_op_topk_నెట్టు_constants {
    uint32_t orig_ncols;
    uint32_t ncols_ఆదానం;
    uint32_t ncols_output;
    uint32_t k;
    uint32_t nrows;
    uint32_t మొదటి_pass;
    uint32_t చివరి_pass;
};

నిర్మాణం vk_op_topk_radix_నెట్టు_constants {
    uint32_t ncols;
    uint32_t k;
    uint32_t nrows;
    uint32_t n_tps;    // QSA only
    uint32_t n_ఖండాలు; // QSA only
    uint32_t n_ప్రవాహం; // QSA only
};

నిర్మాణం vk_op_im2col_నెట్టు_constants {
    uint64_t dst_addr;
    uint32_t batch_స్థానభ్రంశం; uint32_t స్థానభ్రంశం_delta;
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

నిర్మాణం vk_op_im2col_3d_నెట్టు_constants {
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

నిర్మాణం vk_op_timestep_embedding_నెట్టు_constants {
    uint32_t nb1;
    uint32_t dim;
    uint32_t max_period;
};

నిర్మాణం vk_op_col2im_1d_నెట్టు_constants {
    uint32_t T_బహిర్గతం;
    uint32_t OC;
    uint32_t K_OC;
    uint32_t T_లోపల;
    uint32_t K;
    int32_t  stride;
    int32_t  p0;
};

నిర్మాణం vk_op_conv_transpose_1d_నెట్టు_constants {
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

నిర్మాణం vk_op_snake_నెట్టు_constants {
    uint32_t ne0;
    uint32_t ne1;
};

నిర్మాణం vk_op_pool1d_నెట్టు_constants {
    uint32_t IL;
    uint32_t OL;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0;
    int32_t s0;
    int32_t p0;
};

నిర్మాణం vk_op_pool2d_నెట్టు_constants {
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0; int32_t k1;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
};

నిర్మాణం vk_op_rwkv_wkv6_నెట్టు_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

నిర్మాణం vk_op_rwkv_wkv7_నెట్టు_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

నిర్మాణం vk_op_gated_రేఖీయ_attn_నెట్టు_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
    దశాంశం scale;
};

నిర్మాణం vk_op_lightning_indexer_నెట్టు_constants {
    uint32_t n_kv;
    uint32_t n_heads;
    uint32_t n_టోకెన్లు;
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

static_assert(పరిమాణం(vk_op_lightning_indexer_నెట్టు_constants) <= 128);

నిర్మాణం vk_op_gated_delta_net_నెట్టు_constants {
    uint32_t H;
    uint32_t n_టోకెన్లు;
    uint32_t n_seqs;
    uint32_t s_off;
    uint32_t sq1, sq2, sq3;
    uint32_t sv1, sv2, sv3;
    uint32_t sb1, sb2, sb3;
    uint32_t neq1, rq3;
    దశాంశం scale;
    uint32_t K;
};

నిర్మాణం vk_op_ssm_పరిశీలన_నెట్టు_constants {
    uint32_t nb02, nb03, nb12, nb13;
    uint32_t nb21, nb22, nb31;
    uint32_t nb42, nb43, nb52, nb53;
    uint32_t s_off;
    uint32_t n_తల, d_తల, n_సమూహం, n_tok;
    uint32_t n_seq, K;
};

నిర్మాణం vk_op_ssm_conv_నెట్టు_constants {
    uint32_t nb01, nb02;
    uint32_t nb11;
    uint32_t dst_nb0, dst_nb1, dst_nb2;
    uint32_t nc, ncs, nr, n_t, n_s;
};

నిర్మాణం vk_op_conv2d_నెట్టు_constants {
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

    uint32_t knl_స్థానభ్రంశం;
    uint32_t src_స్థానభ్రంశం;
    uint32_t dst_స్థానభ్రంశం;
};

మూస <> అంతఃవరుస శూన్య init_pushconst_fastdiv(vk_op_conv2d_నెట్టు_constants &p) {
    // Compute magic values to divide by OW, OW*OH
    init_fastdiv_విలువలు(p.OW,       p.OWmp,    p.OWL);
    init_fastdiv_విలువలు(p.OW*p.OH,  p.OWOHmp,  p.OWOHL);
}

నిర్మాణం vk_op_conv3d_నెట్టు_constants {
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

    uint32_t knl_స్థానభ్రంశం;
    uint32_t src_స్థానభ్రంశం;
    uint32_t dst_స్థానభ్రంశం;
};

మూస <> అంతఃవరుస శూన్య init_pushconst_fastdiv(vk_op_conv3d_నెట్టు_constants &p) {
    init_fastdiv_విలువలు(p.OW,             p.OWmp,     p.OWL);
    init_fastdiv_విలువలు(p.OW*p.OH,        p.OWOHmp,   p.OWOHL);
    init_fastdiv_విలువలు(p.OW*p.OH*p.OD,   p.OWOHODmp, p.OWOHODL);
}

నిర్మాణం vk_op_conv2d_dw_నెట్టు_constants {
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

నిర్మాణం vk_op_upscale_నెట్టు_constants {
    uint32_t ne; uint32_t a_స్థానభ్రంశం; uint32_t d_స్థానభ్రంశం;
    uint32_t ne00; uint32_t ne01;
    uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13;
    దశాంశం sf0; దశాంశం sf1; దశాంశం sf2; దశాంశం sf3;
    దశాంశం pixel_స్థానభ్రంశం;
};

నిర్మాణం vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants
{
    uint32_t n_నిలువుల_సంఖ్య;
    uint32_t ne01, ne02;
    uint32_t nb01, nb02, nb03;
    uint32_t nb11, nb12, nb13;
    దశాంశం బరువు;
    uint32_t misalign_offsets;
    uint32_t ne0_12mp, ne0_12L;
    uint32_t ne0_1mp, ne0_1L;
};

స్థిర vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants_init(స్థిరరాశి ggml_tensor * src, స్థిరరాశి ggml_tensor * dst, int64_t n_నిలువుల_సంఖ్య) {
    uint32_t type_కొలత = (uint32_t)ggml_type_కొలత(src->type);
    vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants p = {};
    p.n_నిలువుల_సంఖ్య = (uint32_t)n_నిలువుల_సంఖ్య;
    p.ne01 = (uint32_t)src->ne[1];
    p.ne02 = (uint32_t)src->ne[2];
    p.nb01 = (uint32_t)src->nb[1] / type_కొలత;
    p.nb02 = (uint32_t)src->nb[2] / type_కొలత;
    p.nb03 = (uint32_t)src->nb[3] / type_కొలత;
    p.nb11 = (uint32_t)dst->nb[1] / type_కొలత;
    p.nb12 = (uint32_t)dst->nb[2] / type_కొలత;
    p.nb13 = (uint32_t)dst->nb[3] / type_కొలత;
    p.బరువు = 1.0f;
    తిరిగి p;
}

మూస <> అంతఃవరుస శూన్య init_pushconst_fastdiv(vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants &p) {
    init_fastdiv_విలువలు(p.ne01*p.ne02, p.ne0_12mp, p.ne0_12L);
    init_fastdiv_విలువలు(p.ne01,        p.ne0_1mp,  p.ne0_1L);
}

నిర్మాణం vk_quantize_q8_1_నెట్టు_constants {
    uint32_t ne;
    uint32_t అంకె_ఖండాలు;
};

నిర్మాణం vk_op_flash_attn_విభజించు_k_reduce_నెట్టు_constants {
    uint32_t D;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t k_అంకె;
    uint32_t sinks;
};

నిర్మాణం vk_op_flash_attn_ముసుగు_opt_నెట్టు_constants {
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

నిర్మాణం vk_op_flash_attn_sparse_compact_నెట్టు_constants {
    uint32_t KV;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nbm1;
    uint32_t nbm2;
    uint32_t nbm3;
    uint32_t n_kv_max;
};

మూస <typename T> శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, T &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    GGML_వాడని(p);
    GGML_వాడని(src0);
    GGML_వాడని(src1);
    GGML_వాడని(src2);
    GGML_వాడని(src3);
    GGML_వాడని(dst);
    static_assert(!std::is_const<T>::విలువ, "unexpected type");
    GGML_ASSERT(!src0 || get_misalign_బైట్లు(ctx, src0) == 0);
    GGML_ASSERT(!src1 || get_misalign_బైట్లు(ctx, src1) == 0);
    GGML_ASSERT(!src2 || get_misalign_బైట్లు(ctx, src2) == 0);
    GGML_ASSERT(!src3 || get_misalign_బైట్లు(ctx, src3) == 0);
    GGML_ASSERT(!dst  || get_misalign_బైట్లు(ctx, dst) == 0);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_p021_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    స్థిరరాశి uint32_t b_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src1) / ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst) / ggml_type_కొలత(dst->type);

    p.b_స్థానభ్రంశం = b_స్థానభ్రంశం;
    p.d_స్థానభ్రంశం = d_స్థానభ్రంశం;

    GGML_వాడని(src0);
    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_nc_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    స్థిరరాశి uint32_t b_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src1) / ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst) / ggml_type_కొలత(dst->type);

    p.b_స్థానభ్రంశం = b_స్థానభ్రంశం;
    p.d_స్థానభ్రంశం = d_స్థానభ్రంశం;

    GGML_వాడని(src0);
    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_fwht_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    p.src_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    p.dst_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst)  / ggml_type_కొలత(dst->type);

    GGML_వాడని(src1);
    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_comb_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    p.m_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    p.s_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src1) / ggml_type_కొలత(src1->type);
    p.b_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src2) / ggml_type_కొలత(src2->type);
    p.d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst)  / ggml_type_కొలత(dst->type);

    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_pre_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    p.x_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    p.w_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src1) / ggml_type_కొలత(src1->type);
    p.d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst)  / ggml_type_కొలత(dst->type);

    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_post_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    p.x_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    p.r_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src1) / ggml_type_కొలత(src1->type);
    p.p_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src2) / ggml_type_కొలత(src2->type);
    p.c_స్థానభ్రంశం = src3 ? get_misalign_బైట్లు(ctx, src3) / ggml_type_కొలత(src3->type) : 0;
    p.d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst)  / ggml_type_కొలత(dst->type);
}

మూస <typename T> size_t నెట్టు_constant_కొలత(స్థిరరాశి T &t) {
    static_assert(std::is_class<T>::విలువ, "T must be a struct/class");
    GGML_వాడని(t);
    తిరిగి పరిమాణం(T);
}

మూస <typename T> size_t నెట్టు_constant_కొలత(స్థిరరాశి std::సదిశ<T> &t) {
    GGML_వాడని(t);
    తిరిగి పరిమాణం(T) * t.కొలత();
}

మూస <typename T, uint32_t N> size_t నెట్టు_constant_కొలత(స్థిరరాశి std::array<T, N> &t) {
    GGML_వాడని(t);
    తిరిగి పరిమాణం(T) * N;
}

మూస <typename T> స్థిరరాశి T *నెట్టు_constant_దత్తాంశం(స్థిరరాశి T &t) {
    static_assert(std::is_class<T>::విలువ, "T must be a struct/class");
    తిరిగి &t;
}

మూస <typename T> స్థిరరాశి T *నెట్టు_constant_దత్తాంశం(స్థిరరాశి std::సదిశ<T> &t) {
    తిరిగి t.దత్తాంశం();
}

మూస <typename T, uint32_t N> స్థిరరాశి T *నెట్టు_constant_దత్తాంశం(స్థిరరాశి std::array<T, N> &t) {
    తిరిగి t.దత్తాంశం();
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_unary_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    స్థిరరాశి uint32_t a_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst) / ggml_type_కొలత(dst->type);

    p.misalign_offsets = (a_స్థానభ్రంశం << 16) | d_స్థానభ్రంశం;

    GGML_వాడని(src1);
    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_glu_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    స్థిరరాశి uint32_t a_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t b_స్థానభ్రంశం = src1 ? get_misalign_బైట్లు(ctx, src1) / ggml_type_కొలత(src1->type) : a_స్థానభ్రంశం;
    స్థిరరాశి uint32_t d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst) / ggml_type_కొలత(dst->type);

    GGML_ASSERT(a_స్థానభ్రంశం < (1u << 8));
    GGML_ASSERT(b_స్థానభ్రంశం < (1u << 8));
    GGML_ASSERT(d_స్థానభ్రంశం < (1u << 8));

    p.misalign_offsets = (a_స్థానభ్రంశం << 16) | (b_స్థానభ్రంశం << 8) | d_స్థానభ్రంశం;

    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    స్థిరరాశి uint32_t a_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst) / ggml_type_కొలత(dst->type);

    p.misalign_offsets = (a_స్థానభ్రంశం << 16) | d_స్థానభ్రంశం;

    GGML_వాడని(src1);
    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_pad_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    స్థిరరాశి uint32_t a_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst) / ggml_type_కొలత(dst->type);

    p.misalign_offsets = (a_స్థానభ్రంశం << 16) | d_స్థానభ్రంశం;

    GGML_వాడని(src1);
    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv2d_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    p.src_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src1) / ggml_type_కొలత(src1->type);
    p.dst_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst)  / ggml_type_కొలత(dst->type);

    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv3d_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    p.src_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src1) / ggml_type_కొలత(src1->type);
    p.dst_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst)  / ggml_type_కొలత(dst->type);

    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_im2col_3d_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    స్థిరరాశి uint32_t a_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src1) / ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst) / ggml_type_కొలత(dst->type);

    p.misalign_offsets = (a_స్థానభ్రంశం << 16) | d_స్థానభ్రంశం;

    GGML_వాడని(src0);
    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_ద్విమాన_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    స్థిరరాశి uint32_t a_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t b_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src1) / ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst) / ggml_type_కొలత(dst->type);

    GGML_ASSERT(a_స్థానభ్రంశం <= 0xFFFF);
    GGML_ASSERT(b_స్థానభ్రంశం <= 0xFF);
    GGML_ASSERT(d_స్థానభ్రంశం <= 0xFF);

    p.misalign_offsets = (a_స్థానభ్రంశం << 16) | (b_స్థానభ్రంశం << 8) | d_స్థానభ్రంశం;

    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_concat_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    స్థిరరాశి uint32_t unit_కొలత = ggml_vk_concat_unit_కొలత(dst->type);
    స్థిరరాశి uint32_t a_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / unit_కొలత;
    స్థిరరాశి uint32_t b_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src1) / unit_కొలత;
    స్థిరరాశి uint32_t d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst) / unit_కొలత;

    p.misalign_offsets = (a_స్థానభ్రంశం << 16) | (b_స్థానభ్రంశం << 8) | d_స్థానభ్రంశం;

    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_upscale_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    స్థిరరాశి uint32_t a_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst) / ggml_type_కొలత(dst->type);

    p.a_స్థానభ్రంశం = a_స్థానభ్రంశం;
    p.d_స్థానభ్రంశం = d_స్థానభ్రంశం;

    GGML_వాడని(src1);
    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

మూస <> అంతఃవరుస శూన్య init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_rope_నెట్టు_constants &p, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst) {
    p.a_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, src0) / ggml_type_కొలత(src0->type);
    p.d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, dst)  / ggml_type_కొలత(dst->type);

    GGML_వాడని(src1);
    GGML_వాడని(src2);
    GGML_వాడని(src3);
}

స్థిర vk_op_ద్విమాన_నెట్టు_constants ggml_vk_rms_norm_నెట్టు_constants(
        స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * dst,
        దశాంశం eps, uint32_t అంకె_partials) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    తిరిగి {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        eps, 0.0f, (int32_t)అంకె_partials,
    };
}

