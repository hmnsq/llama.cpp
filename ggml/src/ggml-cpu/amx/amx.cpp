#شامل "amx.h"
#شامل "common.h"
#شامل "mmq.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"
#شامل "ggml-cpu.h"
#شامل "traits.h"

#اگر defined(__linux__)
#شامل <sys/syscall.h>
#شامل <unistd.h>
#اگر_ختم

#شامل <cstdlib>
#شامل <cstring>
#شامل <memory>

#اگر defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
نامخلا ggml::cpu::amx {
جماعت tensor_traits : عوامی ggml::cpu::tensor_traits {
    منطقی work_سائز(صحیح /* n_threads */, ثابت ساخت ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        واپس سچ;
    }

    منطقی compute_forward(ساخت ggml_compute_params * params, ساخت ggml_tensor * op) override {
        اگر (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            واپس سچ;
        }
        واپس جھوٹ;
    }
};

ساکن ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_بفر_t, ساخت ggml_tensor *) {
    ساکن tensor_traits traits;
    واپس &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
ساکن خالی ggml_backend_amx_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    آزاد(بفر->context);
}

ساکن خالی * ggml_backend_amx_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    واپس (خالی *) (بفر->context);
}

ساکن شمار ggml_حالت ggml_backend_amx_بفر_init_tensor(ggml_backend_بفر_t بفر, ساخت ggml_tensor * tensor) {
    tensor->extra = (خالی *) ggml::cpu::amx::get_tensor_traits(بفر, tensor);

    GGML_غیر_مستعمل(بفر);
    واپس GGML_حالت_SUCCESS;
}

ساکن خالی ggml_backend_amx_بفر_memset_tensor(ggml_backend_بفر_t بفر, ساخت ggml_tensor * tensor,
                                                  uint8_t value, size_t انحراف, size_t size) {
    memset((حرف *) tensor->ڈیٹا + انحراف, value, size);

    GGML_غیر_مستعمل(بفر);
}

ساکن خالی ggml_backend_amx_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر, ساخت ggml_tensor * tensor,
                                               ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    اگر (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_وزن(tensor, ڈیٹا, انحراف, size);
    } ورنہ {
        حافظہ_نقل((حرف *) tensor->ڈیٹا + انحراف, ڈیٹا, size);
    }

    GGML_غیر_مستعمل(بفر);
}

/*
// need to figure what we need to do with buffer->extra.
static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(!qtype_has_amx_kernels(tensor->type));
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (qtype_has_amx_kernels(src->type)) {
            ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_nbytes(dst));
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}
*/

ساکن خالی ggml_backend_amx_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    memset(بفر->context, value, بفر->size);
}

ساکن ggml_backend_بفر_i ggml_backend_amx_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_amx_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_amx_بفر_get_بنیاد,
    /* .init_tensor     = */ ggml_backend_amx_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_بفر_سیٹ_tensor,
    /* .get_tensor      = */ خالی_اشاریہ,
    /* .set_tensor_2d   = */ خالی_اشاریہ,
    /* .get_tensor_2d   = */ خالی_اشاریہ,
    /* .cpy_tensor      = */ خالی_اشاریہ,
    /* .clear           = */ ggml_backend_amx_بفر_clear,
    /* .reset           = */ خالی_اشاریہ,
};

ساکن ثابت حرف * ggml_backend_amx_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    واپس "AMX";

    GGML_غیر_مستعمل(buft);
}

ساکن ggml_backend_بفر_t ggml_backend_amx_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    خالی * ڈیٹا = ggml_aligned_malloc(size);
    اگر (ڈیٹا == تہی) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        واپس تہی;
    }

    واپس ggml_backend_بفر_init(buft, ggml_backend_amx_بفر_interface, ڈیٹا, size);
}

ساکن size_t ggml_backend_amx_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    واپس TENSOR_ALIGNMENT;

    GGML_غیر_مستعمل(buft);
}

نامخلا ggml::cpu::amx {
جماعت extra_بفر_type : ggml::cpu::extra_بفر_type {
    منطقی supports_op(ggml_backend_dev_t, ثابت ساخت ggml_tensor * op) override {
        اگر (op->op != GGML_OP_MUL_MAT) {
            واپس جھوٹ;
        }
        خودکار * src0 = op->src[0];
        خودکار * src1 = op->src[1];

        اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            واپس جھوٹ;
        }
        اگر (!src0->بفر || src0->بفر->buft != ggml_backend_amx_بفر_type()) {
            واپس جھوٹ;
        }
        اگر (src1->بفر && !ggml_backend_buft_is_host(src1->بفر->buft)) {
            واپس جھوٹ;
        }
        اگر (op->ne[0] % (TILE_N * 2)) {
            واپس جھوٹ;
        }
        صحیح alignment;
        انتخاب (src0->type) {
            حالت GGML_TYPE_Q4_0:
            حالت GGML_TYPE_Q4_1:
            حالت GGML_TYPE_Q8_0:
                alignment = TILE_K;
                توڑو;
            حالت GGML_TYPE_Q4_K:
            حالت GGML_TYPE_Q5_K:
            حالت GGML_TYPE_Q6_K:
            حالت GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                توڑو;
            حالت GGML_TYPE_F16:
                alignment = 16;
                توڑو;
            طےشدہ:
                واپس جھوٹ;
        }
        اگر (src0->ne[0] % alignment) {
            واپس جھوٹ;
        }
        اگر (src1->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }
        واپس سچ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت ساخت ggml_tensor * op) override {
        اگر (op->op == GGML_OP_MUL_MAT && op->src[0]->بفر &&
            op->src[0]->بفر->buft == ggml_backend_amx_بفر_type()) {
            واپس (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        واپس خالی_اشاریہ;
    }
};
}  // namespace ggml::cpu::amx

ساکن size_t ggml_backend_amx_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    واپس ggml_backend_amx_get_alloc_سائز(tensor);

    GGML_غیر_مستعمل(buft);
}

#تعریف_کرو ARCH_GET_XCOMP_PERM     0x1022
#تعریف_کرو ARCH_REQ_XCOMP_PERM     0x1023
#تعریف_کرو XFEATURE_XTILECFG       17
#تعریف_کرو XFEATURE_XTILEDATA      18

ساکن منطقی ggml_amx_init() {
#اگر defined(__linux__)
    اگر (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        واپس جھوٹ;
    }
    واپس سچ;
#وگرنہ_اگر defined(_WIN32)
    واپس سچ;
#ورنہ
    واپس جھوٹ;
#اگر_ختم
}

ggml_backend_بفر_type_t ggml_backend_amx_بفر_type() {
    ساکن ساخت ggml_backend_بفر_type ggml_backend_بفر_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_بفر_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_بفر_type_alloc_بفر,
                        /* .alloc_buffer_n      = */ خالی_اشاریہ,
                        /* .get_alignment       = */ ggml_backend_amx_بفر_type_get_alignment,
                        /* .get_max_size        = */ خالی_اشاریہ,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_بفر_type_get_alloc_سائز,
                        /* .get_alloc_size_n    = */ تہی,
                        /* .is_host             = */ خالی_اشاریہ,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ نیا ggml::cpu::amx::extra_بفر_type(),
    };

    اگر (!ggml_amx_init()) {
        واپس خالی_اشاریہ;
    }

    واپس &ggml_backend_بفر_type_amx;
}

#اگر_ختم  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
