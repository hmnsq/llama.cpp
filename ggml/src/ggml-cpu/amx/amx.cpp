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
#endif

#شامل <cstdlib>
#شامل <cstring>
#شامل <memory>

#اگر defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
فضاینام ggml::cpu::amx {
رده tensor_traits : عمومی ggml::cpu::tensor_traits {
    منطقی work_size(صحیح /* n_threads */, ثابت ساختار ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        بازگشت درست;
    }

    منطقی compute_forward(ساختار ggml_compute_params * params, ساختار ggml_tensor * op) override {
        اگر (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            بازگشت درست;
        }
        بازگشت نادرست;
    }
};

ایستا ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, ساختار ggml_tensor *) {
    ایستا tensor_traits traits;
    بازگشت &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
ایستا تهی ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    آزاد(buffer->context);
}

ایستا تهی * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    بازگشت (تهی *) (buffer->context);
}

ایستا شمارش ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, ساختار ggml_tensor * tensor) {
    tensor->extra = (تهی *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    بازگشت GGML_STATUS_SUCCESS;
}

ایستا تهی ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, ساختار ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((نویسه *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ایستا تهی ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, ساختار ggml_tensor * tensor,
                                               ثابت تهی * data, size_t offset, size_t size) {
    اگر (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } وگرنه {
        memcpy((نویسه *) tensor->data + offset, data, size);
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

ایستا تهی ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

ایستا ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
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

ایستا ثابت نویسه * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    بازگشت "AMX";

    GGML_UNUSED(buft);
}

ایستا ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    تهی * data = ggml_aligned_malloc(size);
    اگر (data == پوچ) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        بازگشت پوچ;
    }

    بازگشت ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

ایستا size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    بازگشت TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

فضاینام ggml::cpu::amx {
رده extra_buffer_type : ggml::cpu::extra_buffer_type {
    منطقی supports_op(ggml_backend_dev_t, ثابت ساختار ggml_tensor * op) override {
        اگر (op->op != GGML_OP_MUL_MAT) {
            بازگشت نادرست;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            بازگشت نادرست;
        }
        اگر (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            بازگشت نادرست;
        }
        اگر (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            بازگشت نادرست;
        }
        اگر (op->ne[0] % (TILE_N * 2)) {
            بازگشت نادرست;
        }
        صحیح alignment;
        انتخاب (src0->type) {
            حالت GGML_TYPE_Q4_0:
            حالت GGML_TYPE_Q4_1:
            حالت GGML_TYPE_Q8_0:
                alignment = TILE_K;
                بشکن;
            حالت GGML_TYPE_Q4_K:
            حالت GGML_TYPE_Q5_K:
            حالت GGML_TYPE_Q6_K:
            حالت GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                بشکن;
            حالت GGML_TYPE_F16:
                alignment = 16;
                بشکن;
            پیشفرض:
                بازگشت نادرست;
        }
        اگر (src0->ne[0] % alignment) {
            بازگشت نادرست;
        }
        اگر (src1->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }
        بازگشت درست;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت ساختار ggml_tensor * op) override {
        اگر (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            بازگشت (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        بازگشت nullptr;
    }
};
}  // namespace ggml::cpu::amx

ایستا size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ggml_tensor * tensor) {
    بازگشت ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

ایستا منطقی ggml_amx_init() {
#اگر defined(__linux__)
    اگر (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        بازگشت نادرست;
    }
    بازگشت درست;
#وگرنهاگر defined(_WIN32)
    بازگشت درست;
#وگرنه
    بازگشت نادرست;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    ایستا ساختار ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ پوچ,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ جدید ggml::cpu::amx::extra_buffer_type(),
    };

    اگر (!ggml_amx_init()) {
        بازگشت nullptr;
    }

    بازگشت &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
