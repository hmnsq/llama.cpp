#அந்தர்பாவ "amx.h"
#அந்தர்பாவ "common.h"
#அந்தர்பாவ "mmq.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-cpu.h"
#அந்தர்பாவ "traits.h"

#எனில் defined(__linux__)
#அந்தர்பாவ <sys/syscall.h>
#அந்தர்பாவ <unistd.h>
#endif

#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <memory>

#எனில் defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
நாமாகாஶ ggml::cpu::amx {
வகுப்பு tensor_traits : பொது ggml::cpu::tensor_traits {
    தருக்க work_size(முழு /* n_threads */, மாறிலி கட்டமைப்பு ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        திரும்பு உண்மை;
    }

    தருக்க compute_forward(கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * op) override {
        எனில் (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            திரும்பு உண்மை;
        }
        திரும்பு பொய்;
    }
};

நிலையான ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, கட்டமைப்பு ggml_tensor *) {
    நிலையான tensor_traits traits;
    திரும்பு &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
நிலையான வெறுமை ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    விடுவி(buffer->context);
}

நிலையான வெறுமை * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    திரும்பு (வெறுமை *) (buffer->context);
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, கட்டமைப்பு ggml_tensor * tensor) {
    tensor->extra = (வெறுமை *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    திரும்பு GGML_STATUS_SUCCESS;
}

நிலையான வெறுமை ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, கட்டமைப்பு ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((எழுத்து *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

நிலையான வெறுமை ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, கட்டமைப்பு ggml_tensor * tensor,
                                               மாறிலி வெறுமை * data, size_t offset, size_t size) {
    எனில் (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } இல்லையெனில் {
        memcpy((எழுத்து *) tensor->data + offset, data, size);
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

நிலையான வெறுமை ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

நிலையான ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
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

நிலையான மாறிலி எழுத்து * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    திரும்பு "AMX";

    GGML_UNUSED(buft);
}

நிலையான ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    வெறுமை * data = ggml_aligned_malloc(size);
    எனில் (data == வெற்று) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        திரும்பு வெற்று;
    }

    திரும்பு ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

நிலையான size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    திரும்பு TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

நாமாகாஶ ggml::cpu::amx {
வகுப்பு extra_buffer_type : ggml::cpu::extra_buffer_type {
    தருக்க supports_op(ggml_backend_dev_t, மாறிலி கட்டமைப்பு ggml_tensor * op) override {
        எனில் (op->op != GGML_OP_MUL_MAT) {
            திரும்பு பொய்;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            திரும்பு பொய்;
        }
        எனில் (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            திரும்பு பொய்;
        }
        எனில் (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            திரும்பு பொய்;
        }
        எனில் (op->ne[0] % (TILE_N * 2)) {
            திரும்பு பொய்;
        }
        முழு alignment;
        தேர்வு (src0->type) {
            நிலை GGML_TYPE_Q4_0:
            நிலை GGML_TYPE_Q4_1:
            நிலை GGML_TYPE_Q8_0:
                alignment = TILE_K;
                நிறுத்து;
            நிலை GGML_TYPE_Q4_K:
            நிலை GGML_TYPE_Q5_K:
            நிலை GGML_TYPE_Q6_K:
            நிலை GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                alignment = 16;
                நிறுத்து;
            இயல்பு:
                திரும்பு பொய்;
        }
        எனில் (src0->ne[0] % alignment) {
            திரும்பு பொய்;
        }
        எனில் (src1->type != GGML_TYPE_F32) {
            திரும்பு பொய்;
        }
        திரும்பு உண்மை;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(மாறிலி கட்டமைப்பு ggml_tensor * op) override {
        எனில் (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            திரும்பு (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        திரும்பு nullptr;
    }
};
}  // namespace ggml::cpu::amx

நிலையான size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, மாறிலி ggml_tensor * tensor) {
    திரும்பு ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

நிலையான தருக்க ggml_amx_init() {
#எனில் defined(__linux__)
    எனில் (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        திரும்பு பொய்;
    }
    திரும்பு உண்மை;
#அல்லதெனில் defined(_WIN32)
    திரும்பு உண்மை;
#இல்லையெனில்
    திரும்பு பொய்;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    நிலையான கட்டமைப்பு ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ வெற்று,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ புதிய ggml::cpu::amx::extra_buffer_type(),
    };

    எனில் (!ggml_amx_init()) {
        திரும்பு nullptr;
    }

    திரும்பு &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
