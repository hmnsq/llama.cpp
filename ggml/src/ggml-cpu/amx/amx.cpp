#অন্তর্ভাব "amx.h"
#অন্তর্ভাব "common.h"
#অন্তর্ভাব "mmq.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-cpu.h"
#অন্তর্ভাব "traits.h"

#যদি defined(__linux__)
#অন্তর্ভাব <sys/syscall.h>
#অন্তর্ভাব <unistd.h>
#endif

#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <memory>

#যদি defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
নামাকাশ ggml::cpu::amx {
বর্গ tensor_traits : সার্বজনিক ggml::cpu::tensor_traits {
    তার্কিক work_size(পূর্ণাঙ্ক /* n_threads */, ধ্রুব সংরচনা ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        প্রত্যাগম সত্য;
    }

    তার্কিক compute_forward(সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * op) override {
        যদি (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            প্রত্যাগম সত্য;
        }
        প্রত্যাগম অসত্য;
    }
};

স্থির ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, সংরচনা ggml_tensor *) {
    স্থির tensor_traits traits;
    প্রত্যাগম &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
স্থির শূন্য ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    মুক্ত(buffer->context);
}

স্থির শূন্য * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    প্রত্যাগম (শূন্য *) (buffer->context);
}

স্থির গণনা ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, সংরচনা ggml_tensor * tensor) {
    tensor->extra = (শূন্য *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    প্রত্যাগম GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, সংরচনা ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((বর্ণ *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

স্থির শূন্য ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, সংরচনা ggml_tensor * tensor,
                                               ধ্রুব শূন্য * data, size_t offset, size_t size) {
    যদি (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } অন্যথা {
        memcpy((বর্ণ *) tensor->data + offset, data, size);
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

স্থির শূন্য ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

স্থির ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
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

স্থির ধ্রুব বর্ণ * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম "AMX";

    GGML_UNUSED(buft);
}

স্থির ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    শূন্য * data = ggml_aligned_malloc(size);
    যদি (data == রিক্ত) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        প্রত্যাগম রিক্ত;
    }

    প্রত্যাগম ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

স্থির size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

নামাকাশ ggml::cpu::amx {
বর্গ extra_buffer_type : ggml::cpu::extra_buffer_type {
    তার্কিক supports_op(ggml_backend_dev_t, ধ্রুব সংরচনা ggml_tensor * op) override {
        যদি (op->op != GGML_OP_MUL_MAT) {
            প্রত্যাগম অসত্য;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            প্রত্যাগম অসত্য;
        }
        যদি (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            প্রত্যাগম অসত্য;
        }
        যদি (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            প্রত্যাগম অসত্য;
        }
        যদি (op->ne[0] % (TILE_N * 2)) {
            প্রত্যাগম অসত্য;
        }
        পূর্ণাঙ্ক alignment;
        বিকল্প (src0->type) {
            পক্ষ GGML_TYPE_Q4_0:
            পক্ষ GGML_TYPE_Q4_1:
            পক্ষ GGML_TYPE_Q8_0:
                alignment = TILE_K;
                ভঙ্গ;
            পক্ষ GGML_TYPE_Q4_K:
            পক্ষ GGML_TYPE_Q5_K:
            পক্ষ GGML_TYPE_Q6_K:
            পক্ষ GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                alignment = 16;
                ভঙ্গ;
            স্বভাব:
                প্রত্যাগম অসত্য;
        }
        যদি (src0->ne[0] % alignment) {
            প্রত্যাগম অসত্য;
        }
        যদি (src1->type != GGML_TYPE_F32) {
            প্রত্যাগম অসত্য;
        }
        প্রত্যাগম সত্য;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ধ্রুব সংরচনা ggml_tensor * op) override {
        যদি (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            প্রত্যাগম (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        প্রত্যাগম nullptr;
    }
};
}  // namespace ggml::cpu::amx

স্থির size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ধ্রুব ggml_tensor * tensor) {
    প্রত্যাগম ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

স্থির তার্কিক ggml_amx_init() {
#যদি defined(__linux__)
    যদি (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        প্রত্যাগম অসত্য;
    }
    প্রত্যাগম সত্য;
#নোচেদ্যদি defined(_WIN32)
    প্রত্যাগম সত্য;
#অন্যথা
    প্রত্যাগম অসত্য;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    স্থির সংরচনা ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ রিক্ত,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ নূতন ggml::cpu::amx::extra_buffer_type(),
    };

    যদি (!ggml_amx_init()) {
        প্রত্যাগম nullptr;
    }

    প্রত্যাগম &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
