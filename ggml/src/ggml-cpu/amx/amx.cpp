#ضمن "amx.h"
#ضمن "common.h"
#ضمن "mmq.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-cpu.h"
#ضمن "traits.h"

#𐤀𐤌 defined(__linux__)
#ضمن <sys/syscall.h>
#ضمن <unistd.h>
#endif

#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <memory>

#𐤀𐤌 defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
نطاق ggml::cpu::amx {
صنف tensor_traits : عام ggml::cpu::tensor_traits {
    منطقي work_size(صحيح /* n_threads */, ثابت بنية ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        أرجع true;
    }

    منطقي compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op) override {
        𐤀𐤌 (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            أرجع true;
        }
        أرجع خطأ;
    }
};

ساكن ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, بنية ggml_tensor *) {
    ساكن tensor_traits traits;
    أرجع &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
ساكن فراغ ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    حرر(buffer->context);
}

ساكن فراغ * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    أرجع (فراغ *) (buffer->context);
}

ساكن تعداد ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, بنية ggml_tensor * tensor) {
    tensor->extra = (فراغ *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    أرجع GGML_STATUS_SUCCESS;
}

ساكن فراغ ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, بنية ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((حرف *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ساكن فراغ ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, بنية ggml_tensor * tensor,
                                               ثابت فراغ * data, size_t offset, size_t size) {
    𐤀𐤌 (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } وإلا {
        memcpy((حرف *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
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

ساكن فراغ ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

ساكن ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_amx_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_amx_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_amx_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_amx_buffer_clear,
    /* .reset           = */ nullptr,
};

ساكن ثابت حرف * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    أرجع "AMX";

    GGML_UNUSED(buft);
}

ساكن ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    فراغ * data = ggml_aligned_malloc(size);
    𐤀𐤌 (data == عدم) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        أرجع عدم;
    }

    أرجع ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

ساكن size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    أرجع TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

نطاق ggml::cpu::amx {
صنف extra_buffer_type : ggml::cpu::extra_buffer_type {
    منطقي supports_op(ggml_backend_dev_t, ثابت بنية ggml_tensor * op) override {
        𐤀𐤌 (op->op != GGML_OP_MUL_MAT) {
            أرجع خطأ;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        𐤀𐤌 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            أرجع خطأ;
        }
        𐤀𐤌 (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            أرجع خطأ;
        }
        𐤀𐤌 (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            أرجع خطأ;
        }
        𐤀𐤌 (op->ne[0] % (TILE_N * 2)) {
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
        𐤀𐤌 (src0->ne[0] % alignment) {
            أرجع خطأ;
        }
        𐤀𐤌 (src1->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        أرجع true;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت بنية ggml_tensor * op) override {
        𐤀𐤌 (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            أرجع (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        أرجع nullptr;
    }
};
}  // namespace ggml::cpu::amx

ساكن size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ggml_tensor * tensor) {
    أرجع ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

ساكن منطقي ggml_amx_init() {
#𐤀𐤌 defined(__linux__)
    𐤀𐤌 (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        أرجع خطأ;
    }
    أرجع true;
#وإلاإذا defined(_WIN32)
    أرجع true;
#وإلا
    أرجع خطأ;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    ساكن بنية ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ عدم,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ جديد ggml::cpu::amx::extra_buffer_type(),
    };

    𐤀𐤌 (!ggml_amx_init()) {
        أرجع nullptr;
    }

    أرجع &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
