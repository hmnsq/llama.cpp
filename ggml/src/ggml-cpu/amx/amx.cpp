#ଅନ୍ତର୍ଭାଵ "amx.h"
#ଅନ୍ତର୍ଭାଵ "common.h"
#ଅନ୍ତର୍ଭାଵ "mmq.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu.h"
#ଅନ୍ତର୍ଭାଵ "traits.h"

#ଯଦି defined(__linux__)
#ଅନ୍ତର୍ଭାଵ <sys/syscall.h>
#ଅନ୍ତର୍ଭାଵ <unistd.h>
#endif

#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <memory>

#ଯଦି defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
ନାମାକାଶ ggml::cpu::amx {
ଶ୍ରେଣୀ tensor_traits : ସର୍ବସାଧାରଣ ggml::cpu::tensor_traits {
    ତାର୍କିକ work_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା /* n_threads */, ଅଚଳ ସଂରଚନା ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ତାର୍କିକ compute_forward(ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * op) override {
        ଯଦି (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
};

ସ୍ଥିର ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, ସଂରଚନା ggml_tensor *) {
    ସ୍ଥିର tensor_traits traits;
    ଫେରନ୍ତୁ &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ମୁକ୍ତ(buffer->context);
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) (buffer->context);
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, ସଂରଚନା ggml_tensor * tensor) {
    tensor->extra = (ଶୂନ୍ୟ *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, ସଂରଚନା ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((ଅକ୍ଷର *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, ସଂରଚନା ggml_tensor * tensor,
                                               ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ଯଦି (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } ଅନ୍ୟଥା {
        memcpy((ଅକ୍ଷର *) tensor->data + offset, data, size);
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

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

ସ୍ଥିର ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
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

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ "AMX";

    GGML_UNUSED(buft);
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ଶୂନ୍ୟ * data = ggml_aligned_malloc(size);
    ଯଦି (data == ଖାଲି) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଫେରନ୍ତୁ ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

ସ୍ଥିର size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

ନାମାକାଶ ggml::cpu::amx {
ଶ୍ରେଣୀ extra_buffer_type : ggml::cpu::extra_buffer_type {
    ତାର୍କିକ supports_op(ggml_backend_dev_t, ଅଚଳ ସଂରଚନା ggml_tensor * op) override {
        ଯଦି (op->op != GGML_OP_MUL_MAT) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (op->ne[0] % (TILE_N * 2)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା alignment;
        ବାଛ (src0->type) {
            ସ୍ଥିତି GGML_TYPE_Q4_0:
            ସ୍ଥିତି GGML_TYPE_Q4_1:
            ସ୍ଥିତି GGML_TYPE_Q8_0:
                alignment = TILE_K;
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q4_K:
            ସ୍ଥିତି GGML_TYPE_Q5_K:
            ସ୍ଥିତି GGML_TYPE_Q6_K:
            ସ୍ଥିତି GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                alignment = 16;
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (src0->ne[0] % alignment) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (src1->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ଅଚଳ ସଂରଚନା ggml_tensor * op) override {
        ଯଦି (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            ଫେରନ୍ତୁ (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        ଫେରନ୍ତୁ nullptr;
    }
};
}  // namespace ggml::cpu::amx

ସ୍ଥିର size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

ସ୍ଥିର ତାର୍କିକ ggml_amx_init() {
#ଯଦି defined(__linux__)
    ଯଦି (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
#ନଚେତଯଦି defined(_WIN32)
    ଫେରନ୍ତୁ ସତ୍ୟ;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    ସ୍ଥିର ସଂରଚନା ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ ଖାଲି,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ନୂତନ ggml::cpu::amx::extra_buffer_type(),
    };

    ଯଦି (!ggml_amx_init()) {
        ଫେରନ୍ତୁ nullptr;
    }

    ଫେରନ୍ତୁ &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
