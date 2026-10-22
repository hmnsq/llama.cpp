#ضمن "amx.h"
#ضمن "common.h"
#ضمن "mmq.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-cpu.h"
#ضمن "traits.h"

#إذا defined(__linux__)
#ضمن <sys/syscall.h>
#ضمن <unistd.h>
#د_که_پای

#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <memory>

#إذا defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
نطاق ggml::cpu::amx {
صنف tensor_traits : عام ggml::cpu::tensor_traits {
    منطقي work_اندازه(صحيح /* n_threads */, ثابت بنية ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        أرجع ریښتیا;
    }

    منطقي compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op) override {
        إذا (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            أرجع ریښتیا;
        }
        أرجع خطأ;
    }
};

ساكن ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_بفر_t, بنية ggml_tensor *) {
    ساكن tensor_traits traits;
    أرجع &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
ساكن فراغ ggml_backend_amx_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    حرر(بفر->context);
}

ساكن فراغ * ggml_backend_amx_بفر_get_بنسټ(ggml_backend_بفر_t بفر) {
    أرجع (فراغ *) (بفر->context);
}

ساكن تعداد ggml_حالت ggml_backend_amx_بفر_init_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor) {
    tensor->extra = (فراغ *) ggml::cpu::amx::get_tensor_traits(بفر, tensor);

    GGML_ناکارول_شوی(بفر);
    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_amx_بفر_memset_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor,
                                                  uint8_t value, size_t دننه_واټن, size_t size) {
    memset((حرف *) tensor->معلومات + دننه_واټن, value, size);

    GGML_ناکارول_شوی(بفر);
}

ساكن فراغ ggml_backend_amx_بفر_سېټ_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor,
                                               ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    إذا (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_وزن(tensor, معلومات, دننه_واټن, size);
    } وإلا {
        حافظه_نقل((حرف *) tensor->معلومات + دننه_واټن, معلومات, size);
    }

    GGML_ناکارول_شوی(بفر);
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

ساكن فراغ ggml_backend_amx_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    memset(بفر->context, value, بفر->size);
}

ساكن ggml_backend_بفر_i ggml_backend_amx_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_amx_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_amx_بفر_get_بنسټ,
    /* .init_tensor     = */ ggml_backend_amx_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_بفر_سېټ_tensor,
    /* .get_tensor      = */ تش_اشاره,
    /* .set_tensor_2d   = */ تش_اشاره,
    /* .get_tensor_2d   = */ تش_اشاره,
    /* .cpy_tensor      = */ تش_اشاره,
    /* .clear           = */ ggml_backend_amx_بفر_clear,
    /* .reset           = */ تش_اشاره,
};

ساكن ثابت حرف * ggml_backend_amx_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع "AMX";

    GGML_ناکارول_شوی(buft);
}

ساكن ggml_backend_بفر_t ggml_backend_amx_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    فراغ * معلومات = ggml_aligned_malloc(size);
    إذا (معلومات == عدم) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        أرجع عدم;
    }

    أرجع ggml_backend_بفر_init(buft, ggml_backend_amx_بفر_interface, معلومات, size);
}

ساكن size_t ggml_backend_amx_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع TENSOR_ALIGNMENT;

    GGML_ناکارول_شوی(buft);
}

نطاق ggml::cpu::amx {
صنف extra_بفر_type : ggml::cpu::extra_بفر_type {
    منطقي supports_op(ggml_backend_dev_t, ثابت بنية ggml_tensor * op) override {
        إذا (op->op != GGML_OP_MUL_MAT) {
            أرجع خطأ;
        }
        خپلکاری * src0 = op->src[0];
        خپلکاری * src1 = op->src[1];

        إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            أرجع خطأ;
        }
        إذا (!src0->بفر || src0->بفر->buft != ggml_backend_amx_بفر_type()) {
            أرجع خطأ;
        }
        إذا (src1->بفر && !ggml_backend_buft_is_host(src1->بفر->buft)) {
            أرجع خطأ;
        }
        إذا (op->ne[0] % (TILE_N * 2)) {
            أرجع خطأ;
        }
        صحيح alignment;
        بدل (src0->type) {
            حالة GGML_TYPE_Q4_0:
            حالة GGML_TYPE_Q4_1:
            حالة GGML_TYPE_Q8_0:
                alignment = TILE_K;
                اكسر;
            حالة GGML_TYPE_Q4_K:
            حالة GGML_TYPE_Q5_K:
            حالة GGML_TYPE_Q6_K:
            حالة GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                اكسر;
            حالة GGML_TYPE_F16:
                alignment = 16;
                اكسر;
            افتراضي:
                أرجع خطأ;
        }
        إذا (src0->ne[0] % alignment) {
            أرجع خطأ;
        }
        إذا (src1->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        أرجع ریښتیا;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت بنية ggml_tensor * op) override {
        إذا (op->op == GGML_OP_MUL_MAT && op->src[0]->بفر &&
            op->src[0]->بفر->buft == ggml_backend_amx_بفر_type()) {
            أرجع (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        أرجع تش_اشاره;
    }
};
}  // namespace ggml::cpu::amx

ساكن size_t ggml_backend_amx_بفر_type_get_alloc_اندازه(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    أرجع ggml_backend_amx_get_alloc_اندازه(tensor);

    GGML_ناکارول_شوی(buft);
}

#تعریف_کړه ARCH_GET_XCOMP_PERM     0x1022
#تعریف_کړه ARCH_REQ_XCOMP_PERM     0x1023
#تعریف_کړه XFEATURE_XTILECFG       17
#تعریف_کړه XFEATURE_XTILEDATA      18

ساكن منطقي ggml_amx_init() {
#إذا defined(__linux__)
    إذا (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        أرجع خطأ;
    }
    أرجع ریښتیا;
#وإلاإذا defined(_WIN32)
    أرجع ریښتیا;
#وإلا
    أرجع خطأ;
#د_که_پای
}

ggml_backend_بفر_type_t ggml_backend_amx_بفر_type() {
    ساكن بنية ggml_backend_بفر_type ggml_backend_بفر_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_بفر_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_بفر_type_alloc_بفر,
                        /* .alloc_buffer_n      = */ تش_اشاره,
                        /* .get_alignment       = */ ggml_backend_amx_بفر_type_get_alignment,
                        /* .get_max_size        = */ تش_اشاره,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_بفر_type_get_alloc_اندازه,
                        /* .get_alloc_size_n    = */ عدم,
                        /* .is_host             = */ تش_اشاره,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ جديد ggml::cpu::amx::extra_بفر_type(),
    };

    إذا (!ggml_amx_init()) {
        أرجع تش_اشاره;
    }

    أرجع &ggml_backend_بفر_type_amx;
}

#د_که_پای  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
