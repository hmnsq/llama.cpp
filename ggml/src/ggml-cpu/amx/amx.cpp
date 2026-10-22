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
শ্রেণী tensor_traits : সর্বজনীন ggml::cpu::tensor_traits {
    যৌক্তিক work_আকার(পূর্ণসংখ্যা /* n_threads */, ধ্রুবক কাঠামো ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        ফেরত সত্য;
    }

    যৌক্তিক compute_forward(কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * op) override {
        যদি (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            ফেরত সত্য;
        }
        ফেরত মিথ্যা;
    }
};

স্থির ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_বাফার_t, কাঠামো ggml_tensor *) {
    স্থির tensor_traits traits;
    ফেরত &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
স্থির শূন্য ggml_backend_amx_বাফার_free_বাফার(ggml_backend_বাফার_t বাফার) {
    মুক্ত(বাফার->context);
}

স্থির শূন্য * ggml_backend_amx_বাফার_get_base(ggml_backend_বাফার_t বাফার) {
    ফেরত (শূন্য *) (বাফার->context);
}

স্থির গণনা ggml_status ggml_backend_amx_বাফার_init_tensor(ggml_backend_বাফার_t বাফার, কাঠামো ggml_tensor * tensor) {
    tensor->extra = (শূন্য *) ggml::cpu::amx::get_tensor_traits(বাফার, tensor);

    GGML_UNUSED(বাফার);
    ফেরত GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_amx_বাফার_memset_tensor(ggml_backend_বাফার_t বাফার, কাঠামো ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((অক্ষর *) tensor->তথ্য + offset, value, size);

    GGML_UNUSED(বাফার);
}

স্থির শূন্য ggml_backend_amx_বাফার_set_tensor(ggml_backend_বাফার_t বাফার, কাঠামো ggml_tensor * tensor,
                                               ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    যদি (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, তথ্য, offset, size);
    } অন্যথা {
        memcpy((অক্ষর *) tensor->তথ্য + offset, তথ্য, size);
    }

    GGML_UNUSED(বাফার);
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

স্থির শূন্য ggml_backend_amx_বাফার_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    memset(বাফার->context, value, বাফার->size);
}

স্থির ggml_backend_বাফার_i ggml_backend_amx_বাফার_interface = {
    /* .free_buffer     = */ ggml_backend_amx_বাফার_free_বাফার,
    /* .get_base        = */ ggml_backend_amx_বাফার_get_base,
    /* .init_tensor     = */ ggml_backend_amx_বাফার_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_বাফার_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_বাফার_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_amx_বাফার_clear,
    /* .reset           = */ nullptr,
};

স্থির ধ্রুবক অক্ষর * ggml_backend_amx_বাফার_type_get_name(ggml_backend_বাফার_type_t buft) {
    ফেরত "AMX";

    GGML_UNUSED(buft);
}

স্থির ggml_backend_বাফার_t ggml_backend_amx_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size) {
    শূন্য * তথ্য = ggml_aligned_malloc(size);
    যদি (তথ্য == ফাঁকা) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        ফেরত ফাঁকা;
    }

    ফেরত ggml_backend_বাফার_init(buft, ggml_backend_amx_বাফার_interface, তথ্য, size);
}

স্থির size_t ggml_backend_amx_বাফার_type_get_alignment(ggml_backend_বাফার_type_t buft) {
    ফেরত TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

নামাকাশ ggml::cpu::amx {
শ্রেণী extra_বাফার_type : ggml::cpu::extra_বাফার_type {
    যৌক্তিক supports_op(ggml_backend_dev_t, ধ্রুবক কাঠামো ggml_tensor * op) override {
        যদি (op->op != GGML_OP_MUL_MAT) {
            ফেরত মিথ্যা;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            ফেরত মিথ্যা;
        }
        যদি (!src0->বাফার || src0->বাফার->buft != ggml_backend_amx_বাফার_type()) {
            ফেরত মিথ্যা;
        }
        যদি (src1->বাফার && !ggml_backend_buft_is_host(src1->বাফার->buft)) {
            ফেরত মিথ্যা;
        }
        যদি (op->ne[0] % (TILE_N * 2)) {
            ফেরত মিথ্যা;
        }
        পূর্ণসংখ্যা alignment;
        নির্বাচন (src0->type) {
            ক্ষেত্র GGML_TYPE_Q4_0:
            ক্ষেত্র GGML_TYPE_Q4_1:
            ক্ষেত্র GGML_TYPE_Q8_0:
                alignment = TILE_K;
                বিরতি;
            ক্ষেত্র GGML_TYPE_Q4_K:
            ক্ষেত্র GGML_TYPE_Q5_K:
            ক্ষেত্র GGML_TYPE_Q6_K:
            ক্ষেত্র GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                বিরতি;
            ক্ষেত্র GGML_TYPE_F16:
                alignment = 16;
                বিরতি;
            পূর্বনির্ধারিত:
                ফেরত মিথ্যা;
        }
        যদি (src0->ne[0] % alignment) {
            ফেরত মিথ্যা;
        }
        যদি (src1->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }
        ফেরত সত্য;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ধ্রুবক কাঠামো ggml_tensor * op) override {
        যদি (op->op == GGML_OP_MUL_MAT && op->src[0]->বাফার &&
            op->src[0]->বাফার->buft == ggml_backend_amx_বাফার_type()) {
            ফেরত (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        ফেরত nullptr;
    }
};
}  // namespace ggml::cpu::amx

স্থির size_t ggml_backend_amx_বাফার_type_get_alloc_আকার(ggml_backend_বাফার_type_t buft, ধ্রুবক ggml_tensor * tensor) {
    ফেরত ggml_backend_amx_get_alloc_আকার(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

স্থির যৌক্তিক ggml_amx_init() {
#যদি defined(__linux__)
    যদি (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        ফেরত মিথ্যা;
    }
    ফেরত সত্য;
#নতুবা_যদি defined(_WIN32)
    ফেরত সত্য;
#অন্যথা
    ফেরত মিথ্যা;
#endif
}

ggml_backend_বাফার_type_t ggml_backend_amx_বাফার_type() {
    স্থির কাঠামো ggml_backend_বাফার_type ggml_backend_বাফার_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_বাফার_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_বাফার_type_alloc_বাফার,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_বাফার_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_বাফার_type_get_alloc_আকার,
                        /* .get_alloc_size_n    = */ ফাঁকা,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ নতুন ggml::cpu::amx::extra_বাফার_type(),
    };

    যদি (!ggml_amx_init()) {
        ফেরত nullptr;
    }

    ফেরত &ggml_backend_বাফার_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
