#ہدایت once
#ضمن "ggml-vulkan-types.h"

uint32_t get_misalign_بائٹاں(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t);

uint32_t ggml_vk_concat_unit_ناپ(ggml_type type);

بنية vk_mat_mat_ڈالو_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t ادھار_work_گروہ_z; uint32_t عدد_batches;
    uint32_t k_ونڈو;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
    uint32_t padded_N;
};

بنية vk_mat_vec_ڈالو_constants {
    uint32_t ncols;
    uint32_t stride_a;
    uint32_t stride_b;
    uint32_t stride_d;
    uint32_t batch_stride_a;
    uint32_t batch_stride_b;
    uint32_t batch_stride_d;
    uint32_t fusion_flags;
    uint32_t ادھار_work_گروہ_y;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t broadcast2;
    uint32_t broadcast3;
};

بنية vk_mat_vec_p021_ڈالو_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t nchannels_x;
    uint32_t nchannels_y;
    uint32_t b_وتھ;
    uint32_t d_وتھ;
    uint32_t fusion_flags;
};

بنية vk_mat_vec_nc_ڈالو_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t صف_stride_x;
    uint32_t channel_stride_x;
    uint32_t channel_stride_y;
    uint32_t channel_x_مقسوم_علیہ;
    uint32_t ne12;
    uint32_t b_وتھ;
    uint32_t d_وتھ;
    uint32_t nb03;
    uint32_t nb13;
    uint32_t nb23;
    uint32_t fusion_flags;
};

بنية vk_mat_mat_id_ڈالو_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t nei1; uint32_t nbi1; uint32_t ne11;
    uint32_t n_experts;
    uint32_t صف_ids_وتھ;
};

بنية vk_mat_vec_id_ڈالو_constants {
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

بنية vk_flash_attn_ڈالو_constants {
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

    عائم scale;
    عائم max_bias;
    عائم logit_softcap;

    uint32_t نقاب_n_سر_log2;
    عائم m0;
    عائم m1;

    uint32_t gqa_تناسب;
    uint32_t ونڈو_kv;
    uint32_t k_عدد;
};

static_assert(حجم(vk_flash_attn_ڈالو_constants) <= 128, "sizeof(vk_flash_attn_push_constants) must be <= 128");

بنية vk_fa_xe_opt_ڈالو_constants {
    uint32_t kv_seq_len;
    uint32_t activation_لمائی;
    uint32_t q_سر;
    uint32_t kv_سر;
    uint32_t qk_تناسب;
    uint32_t qk_sub_groups;
    uint32_t flag;
    uint32_t nbkv_tok;
    uint32_t nbkv_سر;
    uint32_t batch_stride_q;
    uint32_t batch_stride_k;
    uint32_t batch_stride_v;
    uint32_t batch_stride_m;
    uint32_t batch_stride_o;
    عائم softmax_scale;
};

بنية vk_op_ڈالو_constants {
    uint32_t KX;
    uint32_t KY;
    عائم param1;
    عائم param2;
    عائم param3;
    عائم param4;
};

بنية vk_op_fwht_ڈالو_constants {
    uint32_t n_صفاں;
    uint32_t src_وتھ;
    uint32_t dst_وتھ;
    عائم scale;
};

بنية vk_op_dsv4_hc_comb_ڈالو_constants {
    uint32_t n_ٹوکن;

    uint32_t nbm0; uint32_t nbm1;
    uint32_t nbs0;
    uint32_t nbb0;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t m_وتھ;
    uint32_t s_وتھ;
    uint32_t b_وتھ;
    uint32_t d_وتھ;

    عائم eps;
    uint32_t n_iter;
};

بنية vk_op_dsv4_hc_pre_ڈالو_constants {
    uint32_t n_embd;
    uint32_t n_ٹوکن;

    uint32_t nbx0; uint32_t nbx1; uint32_t nbx2;
    uint32_t nbw0; uint32_t nbw1; uint32_t nbw2;
    uint32_t nbd0; uint32_t nbd1;

    uint32_t x_وتھ;
    uint32_t w_وتھ;
    uint32_t d_وتھ;

    عائم scale;
};

بنية vk_op_dsv4_hc_post_ڈالو_constants {
    uint32_t n_embd;
    uint32_t n_ٹوکن;

    uint32_t nbx0; uint32_t nbx1;
    uint32_t nbr0; uint32_t nbr1; uint32_t nbr2;
    uint32_t nbp0; uint32_t nbp1;
    uint32_t nbc0; uint32_t nbc1; uint32_t nbc2;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t x_وتھ;
    uint32_t r_وتھ;
    uint32_t p_وتھ;
    uint32_t c_وتھ;
    uint32_t d_وتھ;

    uint32_t gate;
    عائم    gate_scale_وچ;
    عائم    gate_scale_باہرلا;
};

بنية vk_op_گنتی_experts_ڈالو_constants {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t nb00;
    uint32_t nb01;
    uint32_t a_وتھ;
    uint32_t n_experts;
    uint32_t ne00mp;
    uint32_t ne00L;
    uint32_t صف_tile_ناپ;
    uint32_t صف_ids_وتھ;
};

بنية vk_op_glu_ڈالو_constants {
    uint32_t N;
    uint32_t ne00;
    uint32_t ne20;
    uint32_t منوال;  // 0: default, 1: swapped, 2: split
    عائم alpha; // for swiglu_oai
    عائم حد_بندی;
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

static_assert(حجم(vk_op_glu_ڈالو_constants) <= 128, "sizeof(vk_op_glu_push_constants) must be <= 128");

بنية vk_op_unary_ڈالو_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t misalign_offsets;
    عائم param1; عائم param2; عائم param3; عائم param4;
    uint32_t ne0_012mp; uint32_t ne0_01mp; uint32_t ne0_0mp; uint32_t ne0_Ls;
    uint32_t ne1_012mp; uint32_t ne1_01mp; uint32_t ne1_0mp; uint32_t ne1_Ls;
};

static_assert(حجم(vk_op_unary_ڈالو_constants) <= 128, "sizeof(vk_op_unary_push_constants) must be <= 128");

ساكن vk_op_unary_ڈالو_constants vk_op_unary_ڈالو_constants_init(ثابت ggml_tensor * src0, ثابت ggml_tensor * dst, int64_t ne = 0) {
    GGML_ASSERT(ne != 0 || (ggml_nelements(src0) == ggml_nelements(dst)));
    ne = ne != 0 ? ne : ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_unary_ڈالو_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_ناپ(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_ناپ(dst->type);
    p.ne10 = (uint32_t)dst->ne[0];
    p.ne11 = (uint32_t)dst->ne[1];
    p.ne12 = (uint32_t)dst->ne[2];
    p.ne13 = (uint32_t)dst->ne[3];
    p.nb10 = (uint32_t)(dst->nb[0] / dst_tsize);
    p.nb11 = (uint32_t)(dst->nb[1] / dst_tsize);
    p.nb12 = (uint32_t)(dst->nb[2] / dst_tsize);
    p.nb13 = (uint32_t)(dst->nb[3] / dst_tsize);

    أرجع p; // offsets are initialized later in ggml_vk_op
}

بنية vk_op_pad_ڈالو_constants {
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

ساكن vk_op_pad_ڈالو_constants vk_op_pad_ڈالو_constants_init(ثابت ggml_tensor * src0, ثابت ggml_tensor * dst) {
    int64_t ne = ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_pad_ڈالو_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_ناپ(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_ناپ(dst->type);
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

    أرجع p; // fastdiv values and offsets are initialized later in ggml_vk_op
}

ساكن فراغ init_fastdiv_اقدار(uint32_t d, uint32_t &mp, uint32_t &L)
{
    // compute L = ceil(log2(d));
    L = 0;
    طالما (L < 32 && (uint32_t{1} << L) < d) {
        L++;
    }

    mp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
}

ساكن uint32_t pack_fastdiv_L(uint32_t L0, uint32_t L1, uint32_t L2) {
    أرجع L0 | (L1 << 8) | (L2 << 16);
}

سانچا <typename T> فراغ init_pushconst_fastdiv(T &p) {
    GGML_بن_ورتیا(p);
    static_assert(!std::is_const<T>::قدر, "unexpected type");
}

سانچا <> مضمن فراغ init_pushconst_fastdiv(vk_op_unary_ڈالو_constants &p) {
    // Compute magic values to divide by these six numbers.
    uint32_t ne0_012L;
    uint32_t ne0_01L;
    uint32_t ne0_0L;
    uint32_t ne1_012L;
    uint32_t ne1_01L;
    uint32_t ne1_0L;

    init_fastdiv_اقدار(p.ne02*p.ne01*p.ne00,  p.ne0_012mp,    ne0_012L);
    init_fastdiv_اقدار(p.ne01*p.ne00,         p.ne0_01mp,     ne0_01L);
    init_fastdiv_اقدار(p.ne00,                p.ne0_0mp,      ne0_0L);
    init_fastdiv_اقدار(p.ne12*p.ne11*p.ne10,  p.ne1_012mp,    ne1_012L);
    init_fastdiv_اقدار(p.ne11*p.ne10,         p.ne1_01mp,     ne1_01L);
    init_fastdiv_اقدار(p.ne10,                p.ne1_0mp,      ne1_0L);

    p.ne0_Ls = pack_fastdiv_L(ne0_012L, ne0_01L, ne0_0L);
    p.ne1_Ls = pack_fastdiv_L(ne1_012L, ne1_01L, ne1_0L);
}

سانچا <> مضمن فراغ init_pushconst_fastdiv(vk_op_glu_ڈالو_constants &p) {
    // GLU linearizes over dst, then uses dst coordinates for src0/src1.
    init_fastdiv_اقدار(p.ne22*p.ne21*p.ne20,  p.ne2_012mp,    p.ne2_012L);
    init_fastdiv_اقدار(p.ne21*p.ne20,         p.ne2_01mp,     p.ne2_01L);
    init_fastdiv_اقدار(p.ne20,                p.ne2_0mp,      p.ne2_0L);
}

سانچا <> مضمن فراغ init_pushconst_fastdiv(vk_op_گنتی_experts_ڈالو_constants &p) {
    init_fastdiv_اقدار(p.ne00, p.ne00mp, p.ne00L);
}

بنية vk_op_ثنائی_ڈالو_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23;
    uint32_t misalign_offsets;
    عائم param1; عائم param2; int32_t param3;
};

بنية vk_op_concat_ڈالو_constants : vk_op_ثنائی_ڈالو_constants {};

static_assert(حجم(vk_op_concat_ڈالو_constants) == حجم(vk_op_ثنائی_ڈالو_constants));

static_assert(std::is_standard_layout_v<vk_op_concat_ڈالو_constants>);

بنية vk_op_multi_جوڑو_ڈالو_constants {
    // shape for dst
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23;

    // strides for srcs+dst
    uint32_t nb[MAX_PARAMETER_گنتی][4];

    uint32_t rms_partials;
};

static_assert(MAX_PARAMETER_گنتی == 12);

static_assert(حجم(vk_op_multi_جوڑو_ڈالو_constants) <= 256);

بنية vk_op_topk_moe_ڈالو_constants {
    uint32_t n_صفاں;
    uint32_t n_experts_ڈالو;
    uint32_t n_expert_ورتیا;
    عائم clamp_min;
    عائم clamp_max;
    uint32_t gating_func;
    uint32_t has_bias;
    uint32_t with_norm;
    عائم output_scale;
    عائم output_bias;
};

بنية vk_op_جوڑو_id_ڈالو_constants {
    uint32_t ne0;
    uint32_t ne1;
    uint32_t s01;
    uint32_t s02;
    uint32_t s11;
    uint32_t s21;
};

بنية vk_op_diag_نقاب_ڈالو_constants {
    uint32_t ncols;
    uint32_t صفاں_فی_channel;
    int32_t n_past;
};

بنية vk_op_rope_ڈالو_constants {
    uint32_t rope_منوال;
    uint32_t nrows;
    uint32_t n_dims;
    uint32_t n_offs;
    عائم freq_scale;
    عائم freq_ادھار;
    عائم ext_ضارب;
    عائم attn_ضارب;
    عائم corr_dims[2];
    عائم theta_scale;
    uint32_t has_ff;
    int32_t sections[4];
    uint32_t is_imrope;
    uint32_t is_پٹھ;
    uint32_t سیٹ_صفاں_stride;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t a_وتھ;
    uint32_t d_وتھ;
};

static_assert(حجم(vk_op_rope_ڈالو_constants) <= 128, "sizeof(vk_op_rope_push_constants) must be <= 128");

بنية vk_op_rms_norm_mul_rope_ڈالو_constants {
    vk_op_ثنائی_ڈالو_constants bin;
    vk_op_rope_ڈالو_constants rope;
};

بنية vk_op_soft_max_ڈالو_constants {
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
    عائم scale;
    عائم max_bias;
    عائم m0;
    عائم m1;
    uint32_t n_سر_log2;
    uint32_t nrows_x;
    uint32_t has_sinks;
};

بنية vk_op_argsort_ڈالو_constants {
    uint32_t ncols;
    uint32_t ncols_padded;
    uint32_t ncols_padded_log2;
    uint32_t nrows;
    uint32_t ترتیب;
    uint32_t outer_شروع;
    uint32_t outer_سرا;
    uint32_t inner_شروع;
    uint32_t inner_سرا;
};

بنية vk_op_topk_ڈالو_constants {
    uint32_t orig_ncols;
    uint32_t ncols_داخلہ;
    uint32_t ncols_output;
    uint32_t k;
    uint32_t nrows;
    uint32_t پہلا_pass;
    uint32_t آخری_pass;
};

بنية vk_op_topk_radix_ڈالو_constants {
    uint32_t ncols;
    uint32_t k;
    uint32_t nrows;
    uint32_t n_tps;    // QSA only
    uint32_t n_بلاکاں; // QSA only
    uint32_t n_دھارا; // QSA only
};

بنية vk_op_im2col_ڈالو_constants {
    uint64_t dst_addr;
    uint32_t batch_وتھ; uint32_t وتھ_delta;
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

بنية vk_op_im2col_3d_ڈالو_constants {
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

بنية vk_op_timestep_embedding_ڈالو_constants {
    uint32_t nb1;
    uint32_t dim;
    uint32_t max_period;
};

بنية vk_op_col2im_1d_ڈالو_constants {
    uint32_t T_باہرلا;
    uint32_t OC;
    uint32_t K_OC;
    uint32_t T_وچ;
    uint32_t K;
    int32_t  stride;
    int32_t  p0;
};

بنية vk_op_conv_transpose_1d_ڈالو_constants {
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

بنية vk_op_snake_ڈالو_constants {
    uint32_t ne0;
    uint32_t ne1;
};

بنية vk_op_pool1d_ڈالو_constants {
    uint32_t IL;
    uint32_t OL;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0;
    int32_t s0;
    int32_t p0;
};

بنية vk_op_pool2d_ڈالو_constants {
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0; int32_t k1;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
};

بنية vk_op_rwkv_wkv6_ڈالو_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

بنية vk_op_rwkv_wkv7_ڈالو_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

بنية vk_op_gated_خطی_attn_ڈالو_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
    عائم scale;
};

بنية vk_op_lightning_indexer_ڈالو_constants {
    uint32_t n_kv;
    uint32_t n_heads;
    uint32_t n_ٹوکن;
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

static_assert(حجم(vk_op_lightning_indexer_ڈالو_constants) <= 128);

بنية vk_op_gated_delta_net_ڈالو_constants {
    uint32_t H;
    uint32_t n_ٹوکن;
    uint32_t n_seqs;
    uint32_t s_off;
    uint32_t sq1, sq2, sq3;
    uint32_t sv1, sv2, sv3;
    uint32_t sb1, sb2, sb3;
    uint32_t neq1, rq3;
    عائم scale;
    uint32_t K;
};

بنية vk_op_ssm_جانچ_ڈالو_constants {
    uint32_t nb02, nb03, nb12, nb13;
    uint32_t nb21, nb22, nb31;
    uint32_t nb42, nb43, nb52, nb53;
    uint32_t s_off;
    uint32_t n_سر, d_سر, n_گروہ, n_tok;
    uint32_t n_seq, K;
};

بنية vk_op_ssm_conv_ڈالو_constants {
    uint32_t nb01, nb02;
    uint32_t nb11;
    uint32_t dst_nb0, dst_nb1, dst_nb2;
    uint32_t nc, ncs, nr, n_t, n_s;
};

بنية vk_op_conv2d_ڈالو_constants {
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

    uint32_t knl_وتھ;
    uint32_t src_وتھ;
    uint32_t dst_وتھ;
};

سانچا <> مضمن فراغ init_pushconst_fastdiv(vk_op_conv2d_ڈالو_constants &p) {
    // Compute magic values to divide by OW, OW*OH
    init_fastdiv_اقدار(p.OW,       p.OWmp,    p.OWL);
    init_fastdiv_اقدار(p.OW*p.OH,  p.OWOHmp,  p.OWOHL);
}

بنية vk_op_conv3d_ڈالو_constants {
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

    uint32_t knl_وتھ;
    uint32_t src_وتھ;
    uint32_t dst_وتھ;
};

سانچا <> مضمن فراغ init_pushconst_fastdiv(vk_op_conv3d_ڈالو_constants &p) {
    init_fastdiv_اقدار(p.OW,             p.OWmp,     p.OWL);
    init_fastdiv_اقدار(p.OW*p.OH,        p.OWOHmp,   p.OWOHL);
    init_fastdiv_اقدار(p.OW*p.OH*p.OD,   p.OWOHODmp, p.OWOHODL);
}

بنية vk_op_conv2d_dw_ڈالو_constants {
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

بنية vk_op_upscale_ڈالو_constants {
    uint32_t ne; uint32_t a_وتھ; uint32_t d_وتھ;
    uint32_t ne00; uint32_t ne01;
    uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13;
    عائم sf0; عائم sf1; عائم sf2; عائم sf3;
    عائم pixel_وتھ;
};

بنية vk_op_جوڑ_صفاں_ڈالو_constants
{
    uint32_t n_خانے_گنتی;
    uint32_t ne01, ne02;
    uint32_t nb01, nb02, nb03;
    uint32_t nb11, nb12, nb13;
    عائم بھار;
    uint32_t misalign_offsets;
    uint32_t ne0_12mp, ne0_12L;
    uint32_t ne0_1mp, ne0_1L;
};

ساكن vk_op_جوڑ_صفاں_ڈالو_constants vk_op_جوڑ_صفاں_ڈالو_constants_init(ثابت ggml_tensor * src, ثابت ggml_tensor * dst, int64_t n_خانے_گنتی) {
    uint32_t type_ناپ = (uint32_t)ggml_type_ناپ(src->type);
    vk_op_جوڑ_صفاں_ڈالو_constants p = {};
    p.n_خانے_گنتی = (uint32_t)n_خانے_گنتی;
    p.ne01 = (uint32_t)src->ne[1];
    p.ne02 = (uint32_t)src->ne[2];
    p.nb01 = (uint32_t)src->nb[1] / type_ناپ;
    p.nb02 = (uint32_t)src->nb[2] / type_ناپ;
    p.nb03 = (uint32_t)src->nb[3] / type_ناپ;
    p.nb11 = (uint32_t)dst->nb[1] / type_ناپ;
    p.nb12 = (uint32_t)dst->nb[2] / type_ناپ;
    p.nb13 = (uint32_t)dst->nb[3] / type_ناپ;
    p.بھار = 1.0f;
    أرجع p;
}

سانچا <> مضمن فراغ init_pushconst_fastdiv(vk_op_جوڑ_صفاں_ڈالو_constants &p) {
    init_fastdiv_اقدار(p.ne01*p.ne02, p.ne0_12mp, p.ne0_12L);
    init_fastdiv_اقدار(p.ne01,        p.ne0_1mp,  p.ne0_1L);
}

بنية vk_quantize_q8_1_ڈالو_constants {
    uint32_t ne;
    uint32_t عدد_بلاکاں;
};

بنية vk_op_flash_attn_ونڈو_k_reduce_ڈالو_constants {
    uint32_t D;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t k_عدد;
    uint32_t sinks;
};

بنية vk_op_flash_attn_نقاب_opt_ڈالو_constants {
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

بنية vk_op_flash_attn_sparse_compact_ڈالو_constants {
    uint32_t KV;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nbm1;
    uint32_t nbm2;
    uint32_t nbm3;
    uint32_t n_kv_max;
};

سانچا <typename T> فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, T &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    GGML_بن_ورتیا(p);
    GGML_بن_ورتیا(src0);
    GGML_بن_ورتیا(src1);
    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
    GGML_بن_ورتیا(dst);
    static_assert(!std::is_const<T>::قدر, "unexpected type");
    GGML_ASSERT(!src0 || get_misalign_بائٹاں(ctx, src0) == 0);
    GGML_ASSERT(!src1 || get_misalign_بائٹاں(ctx, src1) == 0);
    GGML_ASSERT(!src2 || get_misalign_بائٹاں(ctx, src2) == 0);
    GGML_ASSERT(!src3 || get_misalign_بائٹاں(ctx, src3) == 0);
    GGML_ASSERT(!dst  || get_misalign_بائٹاں(ctx, dst) == 0);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_p021_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t b_وتھ = get_misalign_بائٹاں(ctx, src1) / ggml_type_ناپ(src1->type);
    ثابت uint32_t d_وتھ = get_misalign_بائٹاں(ctx, dst) / ggml_type_ناپ(dst->type);

    p.b_وتھ = b_وتھ;
    p.d_وتھ = d_وتھ;

    GGML_بن_ورتیا(src0);
    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_nc_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t b_وتھ = get_misalign_بائٹاں(ctx, src1) / ggml_type_ناپ(src1->type);
    ثابت uint32_t d_وتھ = get_misalign_بائٹاں(ctx, dst) / ggml_type_ناپ(dst->type);

    p.b_وتھ = b_وتھ;
    p.d_وتھ = d_وتھ;

    GGML_بن_ورتیا(src0);
    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_fwht_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.src_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    p.dst_وتھ = get_misalign_بائٹاں(ctx, dst)  / ggml_type_ناپ(dst->type);

    GGML_بن_ورتیا(src1);
    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_comb_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.m_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    p.s_وتھ = get_misalign_بائٹاں(ctx, src1) / ggml_type_ناپ(src1->type);
    p.b_وتھ = get_misalign_بائٹاں(ctx, src2) / ggml_type_ناپ(src2->type);
    p.d_وتھ = get_misalign_بائٹاں(ctx, dst)  / ggml_type_ناپ(dst->type);

    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_pre_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.x_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    p.w_وتھ = get_misalign_بائٹاں(ctx, src1) / ggml_type_ناپ(src1->type);
    p.d_وتھ = get_misalign_بائٹاں(ctx, dst)  / ggml_type_ناپ(dst->type);

    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_post_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.x_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    p.r_وتھ = get_misalign_بائٹاں(ctx, src1) / ggml_type_ناپ(src1->type);
    p.p_وتھ = get_misalign_بائٹاں(ctx, src2) / ggml_type_ناپ(src2->type);
    p.c_وتھ = src3 ? get_misalign_بائٹاں(ctx, src3) / ggml_type_ناپ(src3->type) : 0;
    p.d_وتھ = get_misalign_بائٹاں(ctx, dst)  / ggml_type_ناپ(dst->type);
}

سانچا <typename T> size_t ڈالو_constant_ناپ(ثابت T &t) {
    static_assert(std::is_class<T>::قدر, "T must be a struct/class");
    GGML_بن_ورتیا(t);
    أرجع حجم(T);
}

سانچا <typename T> size_t ڈالو_constant_ناپ(ثابت std::سمتیہ<T> &t) {
    GGML_بن_ورتیا(t);
    أرجع حجم(T) * t.ناپ();
}

سانچا <typename T, uint32_t N> size_t ڈالو_constant_ناپ(ثابت std::array<T, N> &t) {
    GGML_بن_ورتیا(t);
    أرجع حجم(T) * N;
}

سانچا <typename T> ثابت T *ڈالو_constant_انکڑے(ثابت T &t) {
    static_assert(std::is_class<T>::قدر, "T must be a struct/class");
    أرجع &t;
}

سانچا <typename T> ثابت T *ڈالو_constant_انکڑے(ثابت std::سمتیہ<T> &t) {
    أرجع t.انکڑے();
}

سانچا <typename T, uint32_t N> ثابت T *ڈالو_constant_انکڑے(ثابت std::array<T, N> &t) {
    أرجع t.انکڑے();
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_unary_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    ثابت uint32_t d_وتھ = get_misalign_بائٹاں(ctx, dst) / ggml_type_ناپ(dst->type);

    p.misalign_offsets = (a_وتھ << 16) | d_وتھ;

    GGML_بن_ورتیا(src1);
    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_glu_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    ثابت uint32_t b_وتھ = src1 ? get_misalign_بائٹاں(ctx, src1) / ggml_type_ناپ(src1->type) : a_وتھ;
    ثابت uint32_t d_وتھ = get_misalign_بائٹاں(ctx, dst) / ggml_type_ناپ(dst->type);

    GGML_ASSERT(a_وتھ < (1u << 8));
    GGML_ASSERT(b_وتھ < (1u << 8));
    GGML_ASSERT(d_وتھ < (1u << 8));

    p.misalign_offsets = (a_وتھ << 16) | (b_وتھ << 8) | d_وتھ;

    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_جوڑ_صفاں_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    ثابت uint32_t d_وتھ = get_misalign_بائٹاں(ctx, dst) / ggml_type_ناپ(dst->type);

    p.misalign_offsets = (a_وتھ << 16) | d_وتھ;

    GGML_بن_ورتیا(src1);
    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_pad_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    ثابت uint32_t d_وتھ = get_misalign_بائٹاں(ctx, dst) / ggml_type_ناپ(dst->type);

    p.misalign_offsets = (a_وتھ << 16) | d_وتھ;

    GGML_بن_ورتیا(src1);
    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv2d_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    p.src_وتھ = get_misalign_بائٹاں(ctx, src1) / ggml_type_ناپ(src1->type);
    p.dst_وتھ = get_misalign_بائٹاں(ctx, dst)  / ggml_type_ناپ(dst->type);

    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv3d_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    p.src_وتھ = get_misalign_بائٹاں(ctx, src1) / ggml_type_ناپ(src1->type);
    p.dst_وتھ = get_misalign_بائٹاں(ctx, dst)  / ggml_type_ناپ(dst->type);

    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_im2col_3d_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_وتھ = get_misalign_بائٹاں(ctx, src1) / ggml_type_ناپ(src1->type);
    ثابت uint32_t d_وتھ = get_misalign_بائٹاں(ctx, dst) / ggml_type_ناپ(dst->type);

    p.misalign_offsets = (a_وتھ << 16) | d_وتھ;

    GGML_بن_ورتیا(src0);
    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_ثنائی_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    ثابت uint32_t b_وتھ = get_misalign_بائٹاں(ctx, src1) / ggml_type_ناپ(src1->type);
    ثابت uint32_t d_وتھ = get_misalign_بائٹاں(ctx, dst) / ggml_type_ناپ(dst->type);

    GGML_ASSERT(a_وتھ <= 0xFFFF);
    GGML_ASSERT(b_وتھ <= 0xFF);
    GGML_ASSERT(d_وتھ <= 0xFF);

    p.misalign_offsets = (a_وتھ << 16) | (b_وتھ << 8) | d_وتھ;

    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_concat_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t unit_ناپ = ggml_vk_concat_unit_ناپ(dst->type);
    ثابت uint32_t a_وتھ = get_misalign_بائٹاں(ctx, src0) / unit_ناپ;
    ثابت uint32_t b_وتھ = get_misalign_بائٹاں(ctx, src1) / unit_ناپ;
    ثابت uint32_t d_وتھ = get_misalign_بائٹاں(ctx, dst) / unit_ناپ;

    p.misalign_offsets = (a_وتھ << 16) | (b_وتھ << 8) | d_وتھ;

    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_upscale_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    ثابت uint32_t d_وتھ = get_misalign_بائٹاں(ctx, dst) / ggml_type_ناپ(dst->type);

    p.a_وتھ = a_وتھ;
    p.d_وتھ = d_وتھ;

    GGML_بن_ورتیا(src1);
    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

سانچا <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_rope_ڈالو_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.a_وتھ = get_misalign_بائٹاں(ctx, src0) / ggml_type_ناپ(src0->type);
    p.d_وتھ = get_misalign_بائٹاں(ctx, dst)  / ggml_type_ناپ(dst->type);

    GGML_بن_ورتیا(src1);
    GGML_بن_ورتیا(src2);
    GGML_بن_ورتیا(src3);
}

ساكن vk_op_ثنائی_ڈالو_constants ggml_vk_rms_norm_ڈالو_constants(
        ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst,
        عائم eps, uint32_t عدد_partials) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    أرجع {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        eps, 0.0f, (int32_t)عدد_partials,
    };
}

