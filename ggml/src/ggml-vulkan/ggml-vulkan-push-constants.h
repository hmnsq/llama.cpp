#رهنمود once
#ضمن "ggml-vulkan-types.h"

uint32_t get_misalign_بایتها(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t);

uint32_t ggml_vk_concat_unit_اندازه(ggml_type type);

بنية vk_mat_mat_افزودن_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t پایه_work_گروه_z; uint32_t رقم_batches;
    uint32_t k_تقسیم;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
    uint32_t padded_N;
};

بنية vk_mat_vec_افزودن_constants {
    uint32_t ncols;
    uint32_t stride_a;
    uint32_t stride_b;
    uint32_t stride_d;
    uint32_t batch_stride_a;
    uint32_t batch_stride_b;
    uint32_t batch_stride_d;
    uint32_t fusion_flags;
    uint32_t پایه_work_گروه_y;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t broadcast2;
    uint32_t broadcast3;
};

بنية vk_mat_vec_p021_افزودن_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t nchannels_x;
    uint32_t nchannels_y;
    uint32_t b_جابه_جایی;
    uint32_t d_جابه_جایی;
    uint32_t fusion_flags;
};

بنية vk_mat_vec_nc_افزودن_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t ردیف_stride_x;
    uint32_t channel_stride_x;
    uint32_t channel_stride_y;
    uint32_t channel_x_مقسوم_علیه;
    uint32_t ne12;
    uint32_t b_جابه_جایی;
    uint32_t d_جابه_جایی;
    uint32_t nb03;
    uint32_t nb13;
    uint32_t nb23;
    uint32_t fusion_flags;
};

بنية vk_mat_mat_id_افزودن_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t nei1; uint32_t nbi1; uint32_t ne11;
    uint32_t n_experts;
    uint32_t ردیف_ids_جابه_جایی;
};

بنية vk_mat_vec_id_افزودن_constants {
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

بنية vk_flash_attn_افزودن_constants {
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

    uint32_t ماسک_n_سر_log2;
    عائم m0;
    عائم m1;

    uint32_t gqa_نسبت;
    uint32_t تقسیم_kv;
    uint32_t k_رقم;
};

static_assert(حجم(vk_flash_attn_افزودن_constants) <= 128, "sizeof(vk_flash_attn_push_constants) must be <= 128");

بنية vk_fa_xe_opt_افزودن_constants {
    uint32_t kv_seq_len;
    uint32_t activation_درازی;
    uint32_t q_سر;
    uint32_t kv_سر;
    uint32_t qk_نسبت;
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

بنية vk_op_افزودن_constants {
    uint32_t KX;
    uint32_t KY;
    عائم param1;
    عائم param2;
    عائم param3;
    عائم param4;
};

بنية vk_op_fwht_افزودن_constants {
    uint32_t n_ردیفها;
    uint32_t src_جابه_جایی;
    uint32_t dst_جابه_جایی;
    عائم scale;
};

بنية vk_op_dsv4_hc_comb_افزودن_constants {
    uint32_t n_نشانهها;

    uint32_t nbm0; uint32_t nbm1;
    uint32_t nbs0;
    uint32_t nbb0;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t m_جابه_جایی;
    uint32_t s_جابه_جایی;
    uint32_t b_جابه_جایی;
    uint32_t d_جابه_جایی;

    عائم eps;
    uint32_t n_iter;
};

بنية vk_op_dsv4_hc_pre_افزودن_constants {
    uint32_t n_embd;
    uint32_t n_نشانهها;

    uint32_t nbx0; uint32_t nbx1; uint32_t nbx2;
    uint32_t nbw0; uint32_t nbw1; uint32_t nbw2;
    uint32_t nbd0; uint32_t nbd1;

    uint32_t x_جابه_جایی;
    uint32_t w_جابه_جایی;
    uint32_t d_جابه_جایی;

    عائم scale;
};

بنية vk_op_dsv4_hc_post_افزودن_constants {
    uint32_t n_embd;
    uint32_t n_نشانهها;

    uint32_t nbx0; uint32_t nbx1;
    uint32_t nbr0; uint32_t nbr1; uint32_t nbr2;
    uint32_t nbp0; uint32_t nbp1;
    uint32_t nbc0; uint32_t nbc1; uint32_t nbc2;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t x_جابه_جایی;
    uint32_t r_جابه_جایی;
    uint32_t p_جابه_جایی;
    uint32_t c_جابه_جایی;
    uint32_t d_جابه_جایی;

    uint32_t gate;
    عائم    gate_scale_درون;
    عائم    gate_scale_بیرون;
};

بنية vk_op_شمار_experts_افزودن_constants {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t nb00;
    uint32_t nb01;
    uint32_t a_جابه_جایی;
    uint32_t n_experts;
    uint32_t ne00mp;
    uint32_t ne00L;
    uint32_t ردیف_tile_اندازه;
    uint32_t ردیف_ids_جابه_جایی;
};

بنية vk_op_glu_افزودن_constants {
    uint32_t N;
    uint32_t ne00;
    uint32_t ne20;
    uint32_t نما;  // 0: default, 1: swapped, 2: split
    عائم alpha; // for swiglu_oai
    عائم حد;
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

static_assert(حجم(vk_op_glu_افزودن_constants) <= 128, "sizeof(vk_op_glu_push_constants) must be <= 128");

بنية vk_op_unary_افزودن_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t misalign_offsets;
    عائم param1; عائم param2; عائم param3; عائم param4;
    uint32_t ne0_012mp; uint32_t ne0_01mp; uint32_t ne0_0mp; uint32_t ne0_Ls;
    uint32_t ne1_012mp; uint32_t ne1_01mp; uint32_t ne1_0mp; uint32_t ne1_Ls;
};

static_assert(حجم(vk_op_unary_افزودن_constants) <= 128, "sizeof(vk_op_unary_push_constants) must be <= 128");

ساكن vk_op_unary_افزودن_constants vk_op_unary_افزودن_constants_init(ثابت ggml_tensor * src0, ثابت ggml_tensor * dst, int64_t ne = 0) {
    GGML_ASSERT(ne != 0 || (ggml_nelements(src0) == ggml_nelements(dst)));
    ne = ne != 0 ? ne : ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_unary_افزودن_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_اندازه(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_اندازه(dst->type);
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

بنية vk_op_pad_افزودن_constants {
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

ساكن vk_op_pad_افزودن_constants vk_op_pad_افزودن_constants_init(ثابت ggml_tensor * src0, ثابت ggml_tensor * dst) {
    int64_t ne = ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_pad_افزودن_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_اندازه(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_اندازه(dst->type);
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

ساكن فراغ init_fastdiv_مقادیر(uint32_t d, uint32_t &mp, uint32_t &L)
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

قالب <typename T> فراغ init_pushconst_fastdiv(T &p) {
    GGML_استفادهنشده(p);
    static_assert(!std::is_const<T>::مقدار, "unexpected type");
}

قالب <> مضمن فراغ init_pushconst_fastdiv(vk_op_unary_افزودن_constants &p) {
    // Compute magic values to divide by these six numbers.
    uint32_t ne0_012L;
    uint32_t ne0_01L;
    uint32_t ne0_0L;
    uint32_t ne1_012L;
    uint32_t ne1_01L;
    uint32_t ne1_0L;

    init_fastdiv_مقادیر(p.ne02*p.ne01*p.ne00,  p.ne0_012mp,    ne0_012L);
    init_fastdiv_مقادیر(p.ne01*p.ne00,         p.ne0_01mp,     ne0_01L);
    init_fastdiv_مقادیر(p.ne00,                p.ne0_0mp,      ne0_0L);
    init_fastdiv_مقادیر(p.ne12*p.ne11*p.ne10,  p.ne1_012mp,    ne1_012L);
    init_fastdiv_مقادیر(p.ne11*p.ne10,         p.ne1_01mp,     ne1_01L);
    init_fastdiv_مقادیر(p.ne10,                p.ne1_0mp,      ne1_0L);

    p.ne0_Ls = pack_fastdiv_L(ne0_012L, ne0_01L, ne0_0L);
    p.ne1_Ls = pack_fastdiv_L(ne1_012L, ne1_01L, ne1_0L);
}

قالب <> مضمن فراغ init_pushconst_fastdiv(vk_op_glu_افزودن_constants &p) {
    // GLU linearizes over dst, then uses dst coordinates for src0/src1.
    init_fastdiv_مقادیر(p.ne22*p.ne21*p.ne20,  p.ne2_012mp,    p.ne2_012L);
    init_fastdiv_مقادیر(p.ne21*p.ne20,         p.ne2_01mp,     p.ne2_01L);
    init_fastdiv_مقادیر(p.ne20,                p.ne2_0mp,      p.ne2_0L);
}

قالب <> مضمن فراغ init_pushconst_fastdiv(vk_op_شمار_experts_افزودن_constants &p) {
    init_fastdiv_مقادیر(p.ne00, p.ne00mp, p.ne00L);
}

بنية vk_op_دودویی_افزودن_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23;
    uint32_t misalign_offsets;
    عائم param1; عائم param2; int32_t param3;
};

بنية vk_op_concat_افزودن_constants : vk_op_دودویی_افزودن_constants {};

static_assert(حجم(vk_op_concat_افزودن_constants) == حجم(vk_op_دودویی_افزودن_constants));

static_assert(std::is_standard_layout_v<vk_op_concat_افزودن_constants>);

بنية vk_op_multi_جمع_کن_افزودن_constants {
    // shape for dst
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23;

    // strides for srcs+dst
    uint32_t nb[MAX_PARAMETER_شمار][4];

    uint32_t rms_partials;
};

static_assert(MAX_PARAMETER_شمار == 12);

static_assert(حجم(vk_op_multi_جمع_کن_افزودن_constants) <= 256);

بنية vk_op_topk_moe_افزودن_constants {
    uint32_t n_ردیفها;
    uint32_t n_experts_افزودن;
    uint32_t n_expert_استفاده_شده;
    عائم clamp_min;
    عائم clamp_max;
    uint32_t gating_func;
    uint32_t has_bias;
    uint32_t with_norm;
    عائم output_scale;
    عائم output_bias;
};

بنية vk_op_جمع_کن_id_افزودن_constants {
    uint32_t ne0;
    uint32_t ne1;
    uint32_t s01;
    uint32_t s02;
    uint32_t s11;
    uint32_t s21;
};

بنية vk_op_diag_ماسک_افزودن_constants {
    uint32_t ncols;
    uint32_t ردیفها_در_هر_channel;
    int32_t n_past;
};

بنية vk_op_rope_افزودن_constants {
    uint32_t rope_نما;
    uint32_t nrows;
    uint32_t n_dims;
    uint32_t n_offs;
    عائم freq_scale;
    عائم freq_پایه;
    عائم ext_ضریب;
    عائم attn_ضریب;
    عائم corr_dims[2];
    عائم theta_scale;
    uint32_t has_ff;
    int32_t sections[4];
    uint32_t is_imrope;
    uint32_t is_پشت;
    uint32_t مجموعه_ردیفها_stride;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t a_جابه_جایی;
    uint32_t d_جابه_جایی;
};

static_assert(حجم(vk_op_rope_افزودن_constants) <= 128, "sizeof(vk_op_rope_push_constants) must be <= 128");

بنية vk_op_rms_norm_mul_rope_افزودن_constants {
    vk_op_دودویی_افزودن_constants bin;
    vk_op_rope_افزودن_constants rope;
};

بنية vk_op_soft_max_افزودن_constants {
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

بنية vk_op_argsort_افزودن_constants {
    uint32_t ncols;
    uint32_t ncols_padded;
    uint32_t ncols_padded_log2;
    uint32_t nrows;
    uint32_t ترتیب;
    uint32_t outer_شروع;
    uint32_t outer_انتها;
    uint32_t inner_شروع;
    uint32_t inner_انتها;
};

بنية vk_op_topk_افزودن_constants {
    uint32_t orig_ncols;
    uint32_t ncols_درون_داد;
    uint32_t ncols_output;
    uint32_t k;
    uint32_t nrows;
    uint32_t اولی_pass;
    uint32_t آخرین_pass;
};

بنية vk_op_topk_radix_افزودن_constants {
    uint32_t ncols;
    uint32_t k;
    uint32_t nrows;
    uint32_t n_tps;    // QSA only
    uint32_t n_بلوکها; // QSA only
    uint32_t n_جریان; // QSA only
};

بنية vk_op_im2col_افزودن_constants {
    uint64_t dst_addr;
    uint32_t batch_جابه_جایی; uint32_t جابه_جایی_delta;
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

بنية vk_op_im2col_3d_افزودن_constants {
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

بنية vk_op_timestep_embedding_افزودن_constants {
    uint32_t nb1;
    uint32_t dim;
    uint32_t max_period;
};

بنية vk_op_col2im_1d_افزودن_constants {
    uint32_t T_بیرون;
    uint32_t OC;
    uint32_t K_OC;
    uint32_t T_درون;
    uint32_t K;
    int32_t  stride;
    int32_t  p0;
};

بنية vk_op_conv_transpose_1d_افزودن_constants {
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

بنية vk_op_snake_افزودن_constants {
    uint32_t ne0;
    uint32_t ne1;
};

بنية vk_op_pool1d_افزودن_constants {
    uint32_t IL;
    uint32_t OL;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0;
    int32_t s0;
    int32_t p0;
};

بنية vk_op_pool2d_افزودن_constants {
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0; int32_t k1;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
};

بنية vk_op_rwkv_wkv6_افزودن_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

بنية vk_op_rwkv_wkv7_افزودن_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

بنية vk_op_gated_خطی_attn_افزودن_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
    عائم scale;
};

بنية vk_op_lightning_indexer_افزودن_constants {
    uint32_t n_kv;
    uint32_t n_heads;
    uint32_t n_نشانهها;
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

static_assert(حجم(vk_op_lightning_indexer_افزودن_constants) <= 128);

بنية vk_op_gated_delta_net_افزودن_constants {
    uint32_t H;
    uint32_t n_نشانهها;
    uint32_t n_seqs;
    uint32_t s_off;
    uint32_t sq1, sq2, sq3;
    uint32_t sv1, sv2, sv3;
    uint32_t sb1, sb2, sb3;
    uint32_t neq1, rq3;
    عائم scale;
    uint32_t K;
};

بنية vk_op_ssm_پیمایش_افزودن_constants {
    uint32_t nb02, nb03, nb12, nb13;
    uint32_t nb21, nb22, nb31;
    uint32_t nb42, nb43, nb52, nb53;
    uint32_t s_off;
    uint32_t n_سر, d_سر, n_گروه, n_tok;
    uint32_t n_seq, K;
};

بنية vk_op_ssm_conv_افزودن_constants {
    uint32_t nb01, nb02;
    uint32_t nb11;
    uint32_t dst_nb0, dst_nb1, dst_nb2;
    uint32_t nc, ncs, nr, n_t, n_s;
};

بنية vk_op_conv2d_افزودن_constants {
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

    uint32_t knl_جابه_جایی;
    uint32_t src_جابه_جایی;
    uint32_t dst_جابه_جایی;
};

قالب <> مضمن فراغ init_pushconst_fastdiv(vk_op_conv2d_افزودن_constants &p) {
    // Compute magic values to divide by OW, OW*OH
    init_fastdiv_مقادیر(p.OW,       p.OWmp,    p.OWL);
    init_fastdiv_مقادیر(p.OW*p.OH,  p.OWOHmp,  p.OWOHL);
}

بنية vk_op_conv3d_افزودن_constants {
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

    uint32_t knl_جابه_جایی;
    uint32_t src_جابه_جایی;
    uint32_t dst_جابه_جایی;
};

قالب <> مضمن فراغ init_pushconst_fastdiv(vk_op_conv3d_افزودن_constants &p) {
    init_fastdiv_مقادیر(p.OW,             p.OWmp,     p.OWL);
    init_fastdiv_مقادیر(p.OW*p.OH,        p.OWOHmp,   p.OWOHL);
    init_fastdiv_مقادیر(p.OW*p.OH*p.OD,   p.OWOHODmp, p.OWOHODL);
}

بنية vk_op_conv2d_dw_افزودن_constants {
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

بنية vk_op_upscale_افزودن_constants {
    uint32_t ne; uint32_t a_جابه_جایی; uint32_t d_جابه_جایی;
    uint32_t ne00; uint32_t ne01;
    uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13;
    عائم sf0; عائم sf1; عائم sf2; عائم sf3;
    عائم pixel_جابه_جایی;
};

بنية vk_op_جمع_ردیفها_افزودن_constants
{
    uint32_t n_تعداد_ستون;
    uint32_t ne01, ne02;
    uint32_t nb01, nb02, nb03;
    uint32_t nb11, nb12, nb13;
    عائم وزن;
    uint32_t misalign_offsets;
    uint32_t ne0_12mp, ne0_12L;
    uint32_t ne0_1mp, ne0_1L;
};

ساكن vk_op_جمع_ردیفها_افزودن_constants vk_op_جمع_ردیفها_افزودن_constants_init(ثابت ggml_tensor * src, ثابت ggml_tensor * dst, int64_t n_تعداد_ستون) {
    uint32_t type_اندازه = (uint32_t)ggml_type_اندازه(src->type);
    vk_op_جمع_ردیفها_افزودن_constants p = {};
    p.n_تعداد_ستون = (uint32_t)n_تعداد_ستون;
    p.ne01 = (uint32_t)src->ne[1];
    p.ne02 = (uint32_t)src->ne[2];
    p.nb01 = (uint32_t)src->nb[1] / type_اندازه;
    p.nb02 = (uint32_t)src->nb[2] / type_اندازه;
    p.nb03 = (uint32_t)src->nb[3] / type_اندازه;
    p.nb11 = (uint32_t)dst->nb[1] / type_اندازه;
    p.nb12 = (uint32_t)dst->nb[2] / type_اندازه;
    p.nb13 = (uint32_t)dst->nb[3] / type_اندازه;
    p.وزن = 1.0f;
    أرجع p;
}

قالب <> مضمن فراغ init_pushconst_fastdiv(vk_op_جمع_ردیفها_افزودن_constants &p) {
    init_fastdiv_مقادیر(p.ne01*p.ne02, p.ne0_12mp, p.ne0_12L);
    init_fastdiv_مقادیر(p.ne01,        p.ne0_1mp,  p.ne0_1L);
}

بنية vk_quantize_q8_1_افزودن_constants {
    uint32_t ne;
    uint32_t رقم_بلوکها;
};

بنية vk_op_flash_attn_تقسیم_k_reduce_افزودن_constants {
    uint32_t D;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t k_رقم;
    uint32_t sinks;
};

بنية vk_op_flash_attn_ماسک_opt_افزودن_constants {
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

بنية vk_op_flash_attn_sparse_compact_افزودن_constants {
    uint32_t KV;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nbm1;
    uint32_t nbm2;
    uint32_t nbm3;
    uint32_t n_kv_max;
};

قالب <typename T> فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, T &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    GGML_استفادهنشده(p);
    GGML_استفادهنشده(src0);
    GGML_استفادهنشده(src1);
    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
    GGML_استفادهنشده(dst);
    static_assert(!std::is_const<T>::مقدار, "unexpected type");
    GGML_ASSERT(!src0 || get_misalign_بایتها(ctx, src0) == 0);
    GGML_ASSERT(!src1 || get_misalign_بایتها(ctx, src1) == 0);
    GGML_ASSERT(!src2 || get_misalign_بایتها(ctx, src2) == 0);
    GGML_ASSERT(!src3 || get_misalign_بایتها(ctx, src3) == 0);
    GGML_ASSERT(!dst  || get_misalign_بایتها(ctx, dst) == 0);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_p021_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t b_جابه_جایی = get_misalign_بایتها(ctx, src1) / ggml_type_اندازه(src1->type);
    ثابت uint32_t d_جابه_جایی = get_misalign_بایتها(ctx, dst) / ggml_type_اندازه(dst->type);

    p.b_جابه_جایی = b_جابه_جایی;
    p.d_جابه_جایی = d_جابه_جایی;

    GGML_استفادهنشده(src0);
    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_nc_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t b_جابه_جایی = get_misalign_بایتها(ctx, src1) / ggml_type_اندازه(src1->type);
    ثابت uint32_t d_جابه_جایی = get_misalign_بایتها(ctx, dst) / ggml_type_اندازه(dst->type);

    p.b_جابه_جایی = b_جابه_جایی;
    p.d_جابه_جایی = d_جابه_جایی;

    GGML_استفادهنشده(src0);
    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_fwht_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.src_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    p.dst_جابه_جایی = get_misalign_بایتها(ctx, dst)  / ggml_type_اندازه(dst->type);

    GGML_استفادهنشده(src1);
    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_comb_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.m_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    p.s_جابه_جایی = get_misalign_بایتها(ctx, src1) / ggml_type_اندازه(src1->type);
    p.b_جابه_جایی = get_misalign_بایتها(ctx, src2) / ggml_type_اندازه(src2->type);
    p.d_جابه_جایی = get_misalign_بایتها(ctx, dst)  / ggml_type_اندازه(dst->type);

    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_pre_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.x_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    p.w_جابه_جایی = get_misalign_بایتها(ctx, src1) / ggml_type_اندازه(src1->type);
    p.d_جابه_جایی = get_misalign_بایتها(ctx, dst)  / ggml_type_اندازه(dst->type);

    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_post_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.x_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    p.r_جابه_جایی = get_misalign_بایتها(ctx, src1) / ggml_type_اندازه(src1->type);
    p.p_جابه_جایی = get_misalign_بایتها(ctx, src2) / ggml_type_اندازه(src2->type);
    p.c_جابه_جایی = src3 ? get_misalign_بایتها(ctx, src3) / ggml_type_اندازه(src3->type) : 0;
    p.d_جابه_جایی = get_misalign_بایتها(ctx, dst)  / ggml_type_اندازه(dst->type);
}

قالب <typename T> size_t افزودن_constant_اندازه(ثابت T &t) {
    static_assert(std::is_class<T>::مقدار, "T must be a struct/class");
    GGML_استفادهنشده(t);
    أرجع حجم(T);
}

قالب <typename T> size_t افزودن_constant_اندازه(ثابت std::وکتور<T> &t) {
    GGML_استفادهنشده(t);
    أرجع حجم(T) * t.اندازه();
}

قالب <typename T, uint32_t N> size_t افزودن_constant_اندازه(ثابت std::array<T, N> &t) {
    GGML_استفادهنشده(t);
    أرجع حجم(T) * N;
}

قالب <typename T> ثابت T *افزودن_constant_داده(ثابت T &t) {
    static_assert(std::is_class<T>::مقدار, "T must be a struct/class");
    أرجع &t;
}

قالب <typename T> ثابت T *افزودن_constant_داده(ثابت std::وکتور<T> &t) {
    أرجع t.داده();
}

قالب <typename T, uint32_t N> ثابت T *افزودن_constant_داده(ثابت std::array<T, N> &t) {
    أرجع t.داده();
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_unary_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    ثابت uint32_t d_جابه_جایی = get_misalign_بایتها(ctx, dst) / ggml_type_اندازه(dst->type);

    p.misalign_offsets = (a_جابه_جایی << 16) | d_جابه_جایی;

    GGML_استفادهنشده(src1);
    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_glu_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    ثابت uint32_t b_جابه_جایی = src1 ? get_misalign_بایتها(ctx, src1) / ggml_type_اندازه(src1->type) : a_جابه_جایی;
    ثابت uint32_t d_جابه_جایی = get_misalign_بایتها(ctx, dst) / ggml_type_اندازه(dst->type);

    GGML_ASSERT(a_جابه_جایی < (1u << 8));
    GGML_ASSERT(b_جابه_جایی < (1u << 8));
    GGML_ASSERT(d_جابه_جایی < (1u << 8));

    p.misalign_offsets = (a_جابه_جایی << 16) | (b_جابه_جایی << 8) | d_جابه_جایی;

    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_جمع_ردیفها_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    ثابت uint32_t d_جابه_جایی = get_misalign_بایتها(ctx, dst) / ggml_type_اندازه(dst->type);

    p.misalign_offsets = (a_جابه_جایی << 16) | d_جابه_جایی;

    GGML_استفادهنشده(src1);
    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_pad_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    ثابت uint32_t d_جابه_جایی = get_misalign_بایتها(ctx, dst) / ggml_type_اندازه(dst->type);

    p.misalign_offsets = (a_جابه_جایی << 16) | d_جابه_جایی;

    GGML_استفادهنشده(src1);
    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv2d_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    p.src_جابه_جایی = get_misalign_بایتها(ctx, src1) / ggml_type_اندازه(src1->type);
    p.dst_جابه_جایی = get_misalign_بایتها(ctx, dst)  / ggml_type_اندازه(dst->type);

    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv3d_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    p.src_جابه_جایی = get_misalign_بایتها(ctx, src1) / ggml_type_اندازه(src1->type);
    p.dst_جابه_جایی = get_misalign_بایتها(ctx, dst)  / ggml_type_اندازه(dst->type);

    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_im2col_3d_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_جابه_جایی = get_misalign_بایتها(ctx, src1) / ggml_type_اندازه(src1->type);
    ثابت uint32_t d_جابه_جایی = get_misalign_بایتها(ctx, dst) / ggml_type_اندازه(dst->type);

    p.misalign_offsets = (a_جابه_جایی << 16) | d_جابه_جایی;

    GGML_استفادهنشده(src0);
    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_دودویی_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    ثابت uint32_t b_جابه_جایی = get_misalign_بایتها(ctx, src1) / ggml_type_اندازه(src1->type);
    ثابت uint32_t d_جابه_جایی = get_misalign_بایتها(ctx, dst) / ggml_type_اندازه(dst->type);

    GGML_ASSERT(a_جابه_جایی <= 0xFFFF);
    GGML_ASSERT(b_جابه_جایی <= 0xFF);
    GGML_ASSERT(d_جابه_جایی <= 0xFF);

    p.misalign_offsets = (a_جابه_جایی << 16) | (b_جابه_جایی << 8) | d_جابه_جایی;

    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_concat_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t unit_اندازه = ggml_vk_concat_unit_اندازه(dst->type);
    ثابت uint32_t a_جابه_جایی = get_misalign_بایتها(ctx, src0) / unit_اندازه;
    ثابت uint32_t b_جابه_جایی = get_misalign_بایتها(ctx, src1) / unit_اندازه;
    ثابت uint32_t d_جابه_جایی = get_misalign_بایتها(ctx, dst) / unit_اندازه;

    p.misalign_offsets = (a_جابه_جایی << 16) | (b_جابه_جایی << 8) | d_جابه_جایی;

    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_upscale_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    ثابت uint32_t a_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    ثابت uint32_t d_جابه_جایی = get_misalign_بایتها(ctx, dst) / ggml_type_اندازه(dst->type);

    p.a_جابه_جایی = a_جابه_جایی;
    p.d_جابه_جایی = d_جابه_جایی;

    GGML_استفادهنشده(src1);
    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

قالب <> مضمن فراغ init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_rope_افزودن_constants &p, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst) {
    p.a_جابه_جایی = get_misalign_بایتها(ctx, src0) / ggml_type_اندازه(src0->type);
    p.d_جابه_جایی = get_misalign_بایتها(ctx, dst)  / ggml_type_اندازه(dst->type);

    GGML_استفادهنشده(src1);
    GGML_استفادهنشده(src2);
    GGML_استفادهنشده(src3);
}

ساكن vk_op_دودویی_افزودن_constants ggml_vk_rms_norm_افزودن_constants(
        ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst,
        عائم eps, uint32_t رقم_partials) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    أرجع {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        eps, 0.0f, (int32_t)رقم_partials,
    };
}

