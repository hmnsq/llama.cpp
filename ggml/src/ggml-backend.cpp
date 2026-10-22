// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#অন্তর্ভাব <windows.h>
#endif

#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-alloc.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <limits.h>
#অন্তর্ভাব <stdarg.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <string.h>
#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <vector>

#ifdef __APPLE__
#অন্তর্ভাব <sys/types.h>
#অন্তর্ভাব <sys/sysctl.h>
#endif


// backend buffer type

ধ্রুবক অক্ষর * ggml_backend_buft_name(ggml_backend_বাফার_type_t buft) {
    GGML_ASSERT(buft);
    ফেরত buft->iface.get_name(buft);
}

ggml_backend_বাফার_t ggml_backend_buft_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    যদি (size == 0) {
        // return a dummy buffer for zero-sized allocations
        ফেরত ggml_backend_বাফার_init(buft, {}, ফাঁকা, 0);
    }
    ফেরত buft->iface.alloc_বাফার(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
কাঠামো ggml_backend_buft_alloc_বাফার_n_plan_item {
    size_t size;  // total bytes for this buffer
    পূর্ণসংখ্যা    first; // first tensor index (inclusive)
    পূর্ণসংখ্যা    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_বাফার_n_plan_t = std::vector<ggml_backend_buft_alloc_বাফার_n_plan_item>;

স্থির ggml_backend_buft_alloc_বাফার_n_plan_t ggml_backend_buft_alloc_বাফার_n_plan(
        ggml_backend_বাফার_type_t buft, কাঠামো ggml_tensor ** tensors, পূর্ণসংখ্যা n_tensors) {
    ggml_backend_buft_alloc_বাফার_n_plan_t plan;

    ধ্রুবক size_t alignment = ggml_backend_buft_get_alignment(buft);
    ধ্রুবক size_t max_size  = ggml_backend_buft_get_max_আকার(buft);

    size_t cur_buf_আকার = 0;
    পূর্ণসংখ্যা    first        = 0;

    জন্য (পূর্ণসংখ্যা i = 0; i < n_tensors; i++) {
        size_t this_আকার = 0;
        কাঠামো ggml_tensor * t = tensors[i];
        যদি (t->তথ্য == ফাঁকা && t->view_src == ফাঁকা) {
            this_আকার = GGML_PAD(ggml_backend_buft_get_alloc_আকার(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        যদি (cur_buf_আকার > 0 && (cur_buf_আকার + this_আকার) > max_size) {
            plan.push_back({ cur_buf_আকার, first, i });
            cur_buf_আকার = this_আকার;
            first        = i;
        } অন্যথা {
            cur_buf_আকার += this_আকার;
        }
    }

    যদি (cur_buf_আকার > 0) {
        plan.push_back({ cur_buf_আকার, first, n_tensors });
    }

    ফেরত plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
স্থির ggml_backend_বাফার_t ggml_backend_buft_alloc_বাফার_n_default(ggml_backend_বাফার_type_t buft, কাঠামো ggml_tensor ** tensors, পূর্ণসংখ্যা n_tensors) {
    ধ্রুবক ggml_backend_buft_alloc_বাফার_n_plan_t plan = ggml_backend_buft_alloc_বাফার_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_বাফার_t> buffers;
    buffers.reserve(plan.size());

    জন্য (ধ্রুবক ggml_backend_buft_alloc_বাফার_n_plan_item & item : plan) {
        ggml_backend_বাফার_t বাফার = ggml_backend_buft_alloc_বাফার(buft, item.size);
        যদি (বাফার == ফাঁকা) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            জন্য (ggml_backend_বাফার_t b : buffers) {
                ggml_backend_বাফার_free(b);
            }
            ফেরত ফাঁকা;
        }

        কাঠামো ggml_tallocr tallocr = ggml_tallocr_new(বাফার);

        // allocate tensors in the current buffer
        কাঠামো ggml_tensor * t_failed = ফাঁকা;
        জন্য (পূর্ণসংখ্যা j = item.first; j < item.last; j++) {
            কাঠামো ggml_tensor * t = tensors[j];
            যদি (t->তথ্য == ফাঁকা) {
                যদি (t->view_src == ফাঁকা) {
                    যদি (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        বিরতি;
                    }
                } অন্যথা যদি (t->বাফার == ফাঁকা) {
                    যদি (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        বিরতি;
                    }
                }
            } অন্যথা {
                যদি (t->view_src != ফাঁকা && t->বাফার == ফাঁকা) {
                    // view of a pre-allocated tensor
                    যদি (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        বিরতি;
                    }
                }
            }
        }
        যদি (t_failed != ফাঁকা) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            জন্য (ggml_backend_বাফার_t b : buffers) {
                ggml_backend_বাফার_free(b);
            }
            ggml_backend_বাফার_free(বাফার);
            ফেরত ফাঁকা;
        }

        buffers.push_back(বাফার);
    }

    যদি (buffers.empty()) {
        ফেরত ফাঁকা;
    }

    যদি (buffers.size() == 1) {
        ফেরত buffers[0];
    }

    ফেরত ggml_backend_multi_বাফার_alloc_বাফার(buffers.তথ্য(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
স্থির size_t ggml_backend_buft_get_alloc_আকার_n_default(ggml_backend_বাফার_type_t buft, কাঠামো ggml_tensor ** tensors, পূর্ণসংখ্যা n_tensors) {
    ধ্রুবক ggml_backend_buft_alloc_বাফার_n_plan_t plan = ggml_backend_buft_alloc_বাফার_n_plan(buft, tensors, n_tensors);

    size_t যোগ = 0;
    জন্য (ধ্রুবক ggml_backend_buft_alloc_বাফার_n_plan_item & item : plan) {
        যোগ += item.size;
    }
    ফেরত যোগ;
}

ggml_backend_বাফার_t ggml_backend_buft_alloc_বাফার_n(ggml_backend_বাফার_type_t buft, কাঠামো ggml_tensor ** tensors, পূর্ণসংখ্যা n_tensors) {
    GGML_ASSERT(buft);
    যদি (buft->iface.alloc_বাফার_n) {
        ফেরত buft->iface.alloc_বাফার_n(buft, tensors, n_tensors);
    }
    ফেরত ggml_backend_buft_alloc_বাফার_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_বাফার_type_t buft) {
    GGML_ASSERT(buft);
    ফেরত buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_আকার(ggml_backend_বাফার_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    যদি (buft->iface.get_max_আকার) {
        ফেরত buft->iface.get_max_আকার(buft);
    }
    ফেরত SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_আকার(ggml_backend_বাফার_type_t buft, ধ্রুবক কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    যদি (buft->iface.get_alloc_আকার) {
        size_t size = buft->iface.get_alloc_আকার(buft, tensor);
        প্রতিজ্ঞা(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_আকার_may_expand(tensor->op));

        ফেরত size;
    }
    ফেরত ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_আকার_n(ggml_backend_বাফার_type_t buft, কাঠামো ggml_tensor ** tensors, পূর্ণসংখ্যা n_tensors) {
    GGML_ASSERT(buft);
    যদি (buft->iface.get_alloc_আকার_n) {
        ফেরত buft->iface.get_alloc_আকার_n(buft, tensors, n_tensors);
    }
    ফেরত ggml_backend_buft_get_alloc_আকার_n_default(buft, tensors, n_tensors);
}

যৌক্তিক ggml_backend_buft_is_host(ggml_backend_বাফার_type_t buft) {
    GGML_ASSERT(buft);
    যদি (buft->iface.is_host) {
        ফেরত buft->iface.is_host(buft);
    }
    ফেরত মিথ্যা;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_বাফার_type_t buft) {
    GGML_ASSERT(buft);
    ফেরত buft->device;
}

// backend buffer

ggml_backend_বাফার_t ggml_backend_বাফার_init(
               ggml_backend_বাফার_type_t buft,
        কাঠামো ggml_backend_বাফার_i      iface,
               শূন্য *                     context,
               size_t                     size) {
    ggml_backend_বাফার_t বাফার = নতুন ggml_backend_বাফার {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_বাফার_USAGE_ANY
    };

    ফেরত বাফার;
}

ধ্রুবক অক্ষর * ggml_backend_বাফার_name(ggml_backend_বাফার_t বাফার) {
    ফেরত ggml_backend_buft_name(ggml_backend_বাফার_get_type(বাফার));
}

শূন্য ggml_backend_বাফার_free(ggml_backend_বাফার_t বাফার) {
    যদি (বাফার == ফাঁকা) {
        ফেরত;
    }

    যদি (বাফার->iface.free_বাফার != ফাঁকা) {
        বাফার->iface.free_বাফার(বাফার);
    }
    delete বাফার;
}

size_t ggml_backend_বাফার_get_আকার(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);
    ফেরত বাফার->size;
}

শূন্য * ggml_backend_বাফার_get_base(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);
    // get_base is optional if the buffer is zero-sized
    যদি (!ggml_backend_বাফার_is_meta(বাফার) && বাফার->size == 0) {
        ফেরত ফাঁকা;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    যদি (!বাফার->iface.get_base) {
        ফেরত ফাঁকা;
    }

    শূন্য * base = বাফার->iface.get_base(বাফার);

    GGML_ASSERT(base != ফাঁকা && "backend buffer base cannot be NULL");

    ফেরত base;
}

গণনা ggml_status ggml_backend_বাফার_init_tensor(ggml_backend_বাফার_t বাফার, কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(বাফার);
    // init_tensor is optional
    যদি (বাফার->iface.init_tensor) {
        ফেরত বাফার->iface.init_tensor(বাফার, tensor);
    }
    ফেরত GGML_STATUS_SUCCESS;
}

শূন্য ggml_backend_বাফার_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    GGML_ASSERT(বাফার);
    // clear is optional if the buffer is zero-sized
    যদি (বাফার->size == 0) {
        ফেরত;
    }

    বাফার->iface.clear(বাফার, value);
}

size_t ggml_backend_বাফার_get_alignment(ggml_backend_বাফার_t বাফার) {
    ফেরত ggml_backend_buft_get_alignment(ggml_backend_বাফার_get_type(বাফার));
}

size_t ggml_backend_বাফার_get_max_আকার(ggml_backend_বাফার_t বাফার) {
    ফেরত ggml_backend_buft_get_max_আকার(ggml_backend_বাফার_get_type(বাফার));
}

size_t ggml_backend_বাফার_get_alloc_আকার(ggml_backend_বাফার_t বাফার, ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত ggml_backend_buft_get_alloc_আকার(ggml_backend_বাফার_get_type(বাফার), tensor);
}

যৌক্তিক ggml_backend_বাফার_is_host(ggml_backend_বাফার_t বাফার) {
    ফেরত ggml_backend_buft_is_host(ggml_backend_বাফার_get_type(বাফার));
}

শূন্য ggml_backend_বাফার_set_usage(ggml_backend_বাফার_t বাফার, গণনা ggml_backend_বাফার_usage usage) {
    GGML_ASSERT(বাফার);
    বাফার->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    যদি (ggml_backend_বাফার_is_multi_বাফার(বাফার)) {
        ggml_backend_multi_বাফার_set_usage(বাফার, usage);
    } অন্যথা যদি (ggml_backend_বাফার_is_meta(বাফার)) {
        ggml_backend_meta_বাফার_set_usage(বাফার, usage);
    }
}

গণনা ggml_backend_বাফার_usage ggml_backend_বাফার_get_usage(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);
    ফেরত বাফার->usage;
}

ggml_backend_বাফার_type_t ggml_backend_বাফার_get_type(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);
    ফেরত বাফার->buft;
}

শূন্য ggml_backend_বাফার_reset(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);
    যদি (বাফার->iface.reset) {
        বাফার->iface.reset(বাফার);
    }
}

যৌক্তিক ggml_backend_বাফার_copy_tensor(ধ্রুবক কাঠামো ggml_tensor * src, কাঠামো ggml_tensor * dst) {
    ggml_backend_বাফার_t dst_buf = dst->view_src ? dst->view_src->বাফার : dst->বাফার;
    যদি (dst_buf->iface.cpy_tensor) {
        ফেরত dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    ফেরত মিথ্যা;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    যদি (backend == ফাঁকা) {
        ফেরত ফাঁকা;
    }
    ফেরত backend->guid;
}

ধ্রুবক অক্ষর * ggml_backend_name(ggml_backend_t backend) {
    যদি (backend == ফাঁকা) {
        ফেরত "NULL";
    }
    ফেরত backend->iface.get_name(backend);
}

শূন্য ggml_backend_free(ggml_backend_t backend) {
    যদি (backend == ফাঁকা) {
        ফেরত;
    }

    backend->iface.মুক্ত(backend);
}

ggml_backend_বাফার_type_t ggml_backend_get_default_বাফার_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ফেরত ggml_backend_dev_বাফার_type(backend->device);
}

ggml_backend_বাফার_t ggml_backend_alloc_বাফার(ggml_backend_t backend, size_t size) {
    ফেরত ggml_backend_buft_alloc_বাফার(ggml_backend_get_default_বাফার_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    ফেরত ggml_backend_buft_get_alignment(ggml_backend_get_default_বাফার_type(backend));
}

size_t ggml_backend_get_max_আকার(ggml_backend_t backend) {
    ফেরত ggml_backend_buft_get_max_আকার(ggml_backend_get_default_বাফার_type(backend));
}

শূন্য ggml_backend_tensor_set_async(ggml_backend_t backend, কাঠামো ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    যদি (backend->iface.set_tensor_async == ফাঁকা) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, তথ্য, offset, size);
    } অন্যথা {
        backend->iface.set_tensor_async(backend, tensor, তথ্য, offset, size);
    }
}

শূন্য ggml_backend_tensor_get_async(ggml_backend_t backend, ধ্রুবক কাঠামো ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    যদি (backend->iface.get_tensor_async == ফাঁকা) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, তথ্য, offset, size);
    } অন্যথা {
        backend->iface.get_tensor_async(backend, tensor, তথ্য, offset, size);
    }
}

শূন্য ggml_backend_tensor_set_2d_async(ggml_backend_t backend, কাঠামো ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_তথ্য) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");

    যদি (n_copies <= 1 || backend->iface.set_tensor_2d_async == ফাঁকা) {
        জন্য (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (ধ্রুবক অক্ষর *) তথ্য + i*stride_তথ্য, offset + i*stride_tensor, size);
        }
        ফেরত;
    }
    যদি (size == 0) {
        ফেরত;
    }

    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, তথ্য, offset, size, n_copies, stride_tensor, stride_তথ্য);
}

শূন্য ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ধ্রুবক কাঠামো ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_তথ্য) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");

    যদি (n_copies <= 1 || backend->iface.get_tensor_2d_async == ফাঁকা) {
        জন্য (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (অক্ষর *) তথ্য + i*stride_তথ্য, offset + i*stride_tensor, size);
        }
        ফেরত;
    }
    যদি (size == 0) {
        ফেরত;
    }

    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, তথ্য, offset, size, n_copies, stride_tensor, stride_তথ্য);
}

শূন্য ggml_backend_tensor_set(কাঠামো ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_বাফার_t buf = tensor->view_src ? tensor->view_src->বাফার : tensor->বাফার;
    GGML_ASSERT(buf != ফাঁকা && "tensor buffer not set");

    যদি (size == 0) {
        ফেরত;
    }

    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, তথ্য, offset, size);
}

শূন্য ggml_backend_tensor_get(ধ্রুবক কাঠামো ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_বাফার_t buf = tensor->view_src ? tensor->view_src->বাফার : tensor->বাফার;
    GGML_ASSERT(buf != ফাঁকা && "tensor buffer not set");

    যদি (size == 0) {
        ফেরত;
    }

    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, তথ্য, offset, size);
}

শূন্য ggml_backend_tensor_set_2d(কাঠামো ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_তথ্য) {
    GGML_ASSERT(tensor);
    ggml_backend_বাফার_t buf = tensor->view_src ? tensor->view_src->বাফার : tensor->বাফার;
    GGML_ASSERT(buf != ফাঁকা && "tensor buffer not set");

    যদি (n_copies <= 1 || buf->iface.set_tensor_2d == ফাঁকা) {
        জন্য (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (ধ্রুবক অক্ষর *) তথ্য + i*stride_তথ্য, offset + i*stride_tensor, size);
        }
        ফেরত;
    }
    যদি (size == 0) {
        ফেরত;
    }

    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, তথ্য, offset, size, n_copies, stride_tensor, stride_তথ্য);
}

শূন্য ggml_backend_tensor_get_2d(ধ্রুবক কাঠামো ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_তথ্য) {
    GGML_ASSERT(tensor);
    ggml_backend_বাফার_t buf = tensor->view_src ? tensor->view_src->বাফার : tensor->বাফার;
    GGML_ASSERT(buf != ফাঁকা && "tensor buffer not set");

    যদি (n_copies <= 1 || buf->iface.get_tensor_2d == ফাঁকা) {
        জন্য (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (অক্ষর *) তথ্য + i*stride_তথ্য, offset + i*stride_tensor, size);
        }
        ফেরত;
    }
    যদি (size == 0) {
        ফেরত;
    }

    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, তথ্য, offset, size, n_copies, stride_tensor, stride_তথ্য);
}

শূন্য ggml_backend_tensor_memset(কাঠামো ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_বাফার_t buf = tensor->view_src ? tensor->view_src->বাফার : tensor->বাফার;

    যদি (size == 0) {
        ফেরত;
    }

    GGML_ASSERT(buf != ফাঁকা && "tensor buffer not set");
    GGML_ASSERT(tensor->তথ্য != ফাঁকা && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != ফাঁকা && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

শূন্য ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    যদি (backend->iface.synchronize == ফাঁকা) {
        ফেরত;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, কাঠামো ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != ফাঁকা);

    ফেরত backend->iface.graph_plan_create(backend, cgraph);
}

শূন্য ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != ফাঁকা);

    backend->iface.graph_plan_free(backend, plan);
}

গণনা ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != ফাঁকা);

    ফেরত backend->iface.graph_plan_compute(backend, plan);
}

গণনা ggml_status ggml_backend_graph_compute(ggml_backend_t backend, কাঠামো ggml_cgraph * cgraph) {
    গণনা ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    ফেরত err;
}

গণনা ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, কাঠামো ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    ফেরত backend->iface.graph_compute(backend, cgraph);
}

যৌক্তিক ggml_backend_supports_op(ggml_backend_t backend, ধ্রুবক কাঠামো ggml_tensor * op) {
    GGML_ASSERT(backend);
    ফেরত ggml_backend_dev_supports_op(backend->device, op);
}

যৌক্তিক ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_বাফার_type_t buft) {
    GGML_ASSERT(backend);
    ফেরত ggml_backend_dev_supports_buft(backend->device, buft);
}

যৌক্তিক ggml_backend_offload_op(ggml_backend_t backend, ধ্রুবক কাঠামো ggml_tensor * op) {
    GGML_ASSERT(backend);
    ফেরত ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ফেরত backend->device;
}

// backend copy

শূন্য ggml_backend_tensor_copy(ধ্রুবক কাঠামো ggml_tensor * src, কাঠামো ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    যদি (src == dst) {
        ফেরত;
    }

    যদি (ggml_backend_বাফার_is_host(src->বাফার)) {
        ggml_backend_tensor_set(dst, src->তথ্য, 0, ggml_nbytes(src));
    } অন্যথা যদি (ggml_backend_বাফার_is_host(dst->বাফার)) {
        ggml_backend_tensor_get(src, dst->তথ্য, 0, ggml_nbytes(src));
    } অন্যথা যদি (!ggml_backend_বাফার_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_বাফার_name(src->বাফার), ggml_backend_বাফার_name(dst->বাফার));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        শূন্য * তথ্য = malloc(nbytes);
        ggml_backend_tensor_get(src, তথ্য, 0, nbytes);
        ggml_backend_tensor_set(dst, তথ্য, 0, nbytes);
        মুক্ত(তথ্য);
    }
}

শূন্য ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ধ্রুবক কাঠামো ggml_tensor * src, কাঠামো ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    যদি (src == dst) {
        ফেরত;
    }

    GGML_ASSERT(backend_dst);
    যদি (backend_dst->iface.cpy_tensor_async != ফাঁকা) {
        যদি (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            ফেরত;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    যদি (device == ফাঁকা || device->iface.event_new == ফাঁকা) {
        ফেরত ফাঁকা;
    }
    ফেরত device->iface.event_new(device);
}

শূন্য ggml_backend_event_free(ggml_backend_event_t event) {
    যদি (event == ফাঁকা) {
        ফেরত;
    }
    event->device->iface.event_free(event->device, event);
}

শূন্য ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != ফাঁকা);

    backend->iface.event_record(backend, event);
}

শূন্য ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

শূন্য ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != ফাঁকা);

    backend->iface.event_wait(backend, event);
}

স্থির শূন্য ggml_backend_graph_optimize(ggml_backend_t backend, কাঠামো ggml_cgraph * cgraph, কাঠামো ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    যদি (backend->iface.graph_optimize != ফাঁকা) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

ধ্রুবক অক্ষর * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ফেরত device->iface.get_name(device);
}

ধ্রুবক অক্ষর * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ফেরত device->iface.get_description(device);
}

শূন্য ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * মুক্ত, size_t * যোগ) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, মুক্ত, যোগ);
}

গণনা ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ফেরত device->iface.get_type(device);
}

শূন্য ggml_backend_dev_get_props(ggml_backend_dev_t device, কাঠামো ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, আকার(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ফেরত device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, ধ্রুবক অক্ষর * params) {
    GGML_ASSERT(device);
    ফেরত device->iface.init_backend(device, params);
}

ggml_backend_বাফার_type_t ggml_backend_dev_বাফার_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ফেরত device->iface.get_বাফার_type(device);
}

ggml_backend_বাফার_type_t ggml_backend_dev_host_বাফার_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    যদি (device->iface.get_host_বাফার_type == ফাঁকা) {
        ফেরত ফাঁকা;
    }

    ফেরত device->iface.get_host_বাফার_type(device);
}

ggml_backend_বাফার_t ggml_backend_dev_বাফার_from_host_ptr(ggml_backend_dev_t device, শূন্য * ptr, size_t size, size_t max_tensor_আকার) {
    GGML_ASSERT(device);
    ফেরত device->iface.বাফার_from_host_ptr(device, ptr, size, max_tensor_আকার);
}

যৌক্তিক ggml_backend_dev_supports_op(ggml_backend_dev_t device, ধ্রুবক কাঠামো ggml_tensor * op) {
    GGML_ASSERT(device);
    ফেরত device->iface.supports_op(device, op);
}

যৌক্তিক ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_বাফার_type_t buft) {
    GGML_ASSERT(device);
    ফেরত device->iface.supports_buft(device, buft);
}

যৌক্তিক ggml_backend_dev_offload_op(ggml_backend_dev_t device, ধ্রুবক কাঠামো ggml_tensor * op) {
    GGML_ASSERT(device);
    যদি (device->iface.offload_op != ফাঁকা) {
        ফেরত device->iface.offload_op(device, op);
    }

    ফেরত মিথ্যা;
}

// Backend (reg)

ধ্রুবক অক্ষর * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    ফেরত reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_গুনতি(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    ফেরত reg->iface.get_device_গুনতি(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t সূচক) {
    GGML_ASSERT(reg);
    ফেরত reg->iface.get_device(reg, সূচক);
}

শূন্য * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, ধ্রুবক অক্ষর * name) {
    GGML_ASSERT(reg);
    যদি (!reg->iface.get_proc_address) {
        ফেরত ফাঁকা;
    }
    ফেরত reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

কাঠামো ggml_backend_multi_বাফার_context {
    ggml_backend_বাফার_t * buffers;
    size_t n_buffers;
};

স্থির শূন্য ggml_backend_multi_বাফার_free_বাফার(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);
    ggml_backend_multi_বাফার_context * ctx = (ggml_backend_multi_বাফার_context *) বাফার->context;
    জন্য (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_বাফার_free(ctx->buffers[i]);
    }

    মুক্ত(ctx->buffers);
    মুক্ত(ctx);
}

স্থির শূন্য ggml_backend_multi_বাফার_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    GGML_ASSERT(বাফার);
    ggml_backend_multi_বাফার_context * ctx = (ggml_backend_multi_বাফার_context *) বাফার->context;
    জন্য (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_বাফার_clear(ctx->buffers[i], value);
    }
}

স্থির ধ্রুবক কাঠামো ggml_backend_বাফার_i ggml_backend_multi_বাফার_i = {
    /* .free_buffer     = */ ggml_backend_multi_বাফার_free_বাফার,
    /* .get_base        = */ ফাঁকা,
    /* .init_tensor     = */ ফাঁকা,
    /* .memset_tensor   = */ ফাঁকা,
    /* .set_tensor      = */ ফাঁকা,
    /* .get_tensor      = */ ফাঁকা,
    /* .set_tensor_2d   = */ ফাঁকা,
    /* .get_tensor_2d   = */ ফাঁকা,
    /* .cpy_tensor      = */ ফাঁকা,
    /* .clear           = */ ggml_backend_multi_বাফার_clear,
    /* .reset           = */ ফাঁকা,
};

ggml_backend_বাফার_t ggml_backend_multi_বাফার_alloc_বাফার(ggml_backend_বাফার_t * buffers, size_t n_buffers) {
    ggml_backend_multi_বাফার_context * ctx = (ggml_backend_multi_বাফার_context *) malloc(আকার(কাঠামো ggml_backend_multi_বাফার_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_বাফার_t *) malloc(n_buffers * আকার(ggml_backend_বাফার_t));

    GGML_ASSERT(ctx->buffers != ফাঁকা);

    size_t যোগ_আকার = 0;
    জন্য (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        যোগ_আকার += ggml_backend_বাফার_get_আকার(buffers[i]);
    }

    ফেরত ggml_backend_বাফার_init(buffers[0]->buft, ggml_backend_multi_বাফার_i, ctx, যোগ_আকার);
}

যৌক্তিক ggml_backend_বাফার_is_multi_বাফার(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);
    ফেরত বাফার->iface.free_বাফার == ggml_backend_multi_বাফার_free_বাফার;
}

শূন্য ggml_backend_multi_বাফার_set_usage(ggml_backend_বাফার_t বাফার, গণনা ggml_backend_বাফার_usage usage) {
    GGML_ASSERT(বাফার);
    GGML_ASSERT(ggml_backend_বাফার_is_multi_বাফার(বাফার));
    ggml_backend_multi_বাফার_context * ctx = (ggml_backend_multi_বাফার_context *) বাফার->context;
    জন্য (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_বাফার_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
স্থির কাঠামো ggml_tensor * ggml_dup_tensor_layout(কাঠামো ggml_context * ctx, ধ্রুবক কাঠামো ggml_tensor * tensor) {
    কাঠামো ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    ফেরত dup;
}

স্থির যৌক্তিক ggml_is_view_op(গণনা ggml_op op) {
    ফেরত op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

কাঠামো ggml_backend_sched_split {
    পূর্ণসংখ্যা backend_id;
    পূর্ণসংখ্যা i_start;
    পূর্ণসংখ্যা i_end;
    কাঠামো ggml_tensor ** inputs;
    পূর্ণসংখ্যা n_inputs;
    পূর্ণসংখ্যা inputs_capacity;
    // graph view of this split
    কাঠামো ggml_cgraph graph;
};

কাঠামো ggml_backend_sched {
    যৌক্তিক is_reset; // true if the scheduler has been reset since the last graph split
    যৌক্তিক is_alloc;

    পূর্ণসংখ্যা n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_বাফার_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    কাঠামো ggml_hash_set  hash_set;
    পূর্ণসংখ্যা                 * hv_tensor_backend_ids; // [hash_set.size]
    কাঠামো ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    পূর্ণসংখ্যা * গ্রন্থি_backend_ids; // [graph_size]
    পূর্ণসংখ্যা * leaf_backend_ids; // [graph_size]

    পূর্ণসংখ্যা * পূর্ব_গ্রন্থি_backend_ids; // [graph_size]
    পূর্ণসংখ্যা * পূর্ব_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    কাঠামো ggml_cgraph graph;

    // graph splits
    কাঠামো ggml_backend_sched_split * splits;
    পূর্ণসংখ্যা n_splits;
    পূর্ণসংখ্যা splits_capacity;

    // pipeline parallelism support
    পূর্ণসংখ্যা n_copies;
    পূর্ণসংখ্যা cur_copy;
    পূর্ণসংখ্যা পরবর্তী_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    কাঠামো ggml_tensor ** graph_inputs;
    পূর্ণসংখ্যা n_graph_inputs;
    পূর্ণসংখ্যা graph_inputs_capacity;

    কাঠামো ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    শূন্য * callback_eval_user_তথ্য;

    ggml_backend_sched_copy_callback callback_copy;
    শূন্য * callback_copy_user_তথ্য;

    ggml_backend_sched_realloc_callback callback_realloc;
    শূন্য * callback_realloc_user_তথ্য;

    অক্ষর * context_বাফার;
    size_t context_বাফার_আকার;

    যৌক্তিক op_offload;

    পূর্ণসংখ্যা debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    পূর্ণসংখ্যা graph_আকার;
    পূর্ণসংখ্যা পূর্ব_graph_আকার;

    পূর্ণসংখ্যা debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_ঢোকাও(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

স্থির শূন্য ggml_backend_sched_split_inputs_grow(কাঠামো ggml_backend_sched_split * split) {
    পূর্ণসংখ্যা new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    যদি (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (কাঠামো ggml_tensor **) realloc((শূন্য *) split->inputs, new_cap * আকার(কাঠামো ggml_tensor *));
    যদি (pnew == ফাঁকা) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * আকার(কাঠামো ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

স্থির শূন্য ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    পূর্ণসংখ্যা new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    যদি (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (কাঠামো ggml_tensor **) realloc((শূন্য *) sched->graph_inputs, new_cap * আকার(কাঠামো ggml_tensor *));
    যদি (pnew == ফাঁকা) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * আকার(কাঠামো ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
স্থির পূর্ণসংখ্যা ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    জন্য (পূর্ণসংখ্যা i = 0; i < sched->n_backends; i++) {
        যদি (sched->backends[i] == backend) {
            ফেরত i;
        }
    }
    ফেরত -1;
}

স্থির পূর্ণসংখ্যা ggml_backend_sched_backend_from_বাফার(ggml_backend_sched_t sched, ধ্রুবক কাঠামো ggml_tensor * tensor, ধ্রুবক কাঠামো ggml_tensor * op) {
    ggml_backend_বাফার_t বাফার = tensor->view_src ? tensor->view_src->বাফার : tensor->বাফার;
    যদি (বাফার == ফাঁকা) {
        ফেরত -1;
    }

    // find highest prio backend that supports the buffer type and the op
    জন্য (পূর্ণসংখ্যা i = 0; i < sched->n_backends; i++) {
        যদি (ggml_backend_supports_buft(sched->backends[i], বাফার->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            ফেরত i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_বাফার_name(বাফার), tensor->name);
#endif

    ফেরত -1;
}

#যদি 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
স্থির অক্ষর causes[GGML_DEFAULT_GRAPH_আকার*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(গ্রন্থি, ...) sprintf(causes[hash_id(গ্রন্থি)], __VA_ARGS__)
#define GET_CAUSE(গ্রন্থি) causes[hash_id(গ্রন্থি)]
#অন্যথা
#define SET_CAUSE(গ্রন্থি, ...)
#define GET_CAUSE(গ্রন্থি) ""
#endif

// returns the backend that should be used for the node based on the current locations
স্থির পূর্ণসংখ্যা ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, কাঠামো ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    পূর্ণসংখ্যা cur_backend_id = ggml_backend_sched_backend_from_বাফার(sched, tensor, tensor);
    যদি (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        ফেরত cur_backend_id;
    }

    // view_src
    যদি (tensor->view_src != ফাঁকা) {
        cur_backend_id = ggml_backend_sched_backend_from_বাফার(sched, tensor->view_src, tensor);
        যদি (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            ফেরত cur_backend_id;
        }
    }

    যদি (tensor->বাফার || (tensor->view_src && tensor->view_src->বাফার)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_বাফার_t বাফার = tensor->view_src ? tensor->view_src->বাফার : tensor->বাফার;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_বাফার_name(বাফার), ggml_op_name(tensor->op));
    }

    // graph input
    যদি (tensor->flags & GGML_TENSOR_FLAG_প্রবেশ) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        ফেরত cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    যৌক্তিক allow = সত্য;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    যদি (allow) {
        জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
            ধ্রুবক কাঠামো ggml_tensor * src = tensor->src[i];
            যদি (src == ফাঁকা) {
                চালু;
            }
            যদি (src->বাফার != ফাঁকা && src->বাফার->usage == GGML_BACKEND_বাফার_USAGE_WEIGHTS) {
                পূর্ণসংখ্যা src_backend_id = ggml_backend_sched_backend_from_বাফার(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                যদি (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_বাফার_is_host(src->বাফার)) {
                    জন্য (পূর্ণসংখ্যা b = 0; b < src_backend_id; b++) {
                        যদি (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            ফেরত b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                ফেরত src_backend_id;
            }
        }
    }

    ফেরত -1;
}

স্থির অক্ষর * fmt_আকার(size_t size) {
    স্থির অক্ষর বাফার[128];
    যদি (size >= 1024*1024) {
        snprintf(বাফার, আকার(বাফার), "%zuM", size/1024/1024);
    } অন্যথা {
        snprintf(বাফার, আকার(বাফার), "%zuK", size/1024);
    }
    ফেরত বাফার;
}

স্থির শূন্য ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, কাঠামো ggml_cgraph * graph) {
    পূর্ণসংখ্যা cur_split = 0;
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        যদি (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            জন্য (পূর্ণসংখ্যা j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                যদি (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_আকার(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        যদি (ggml_is_view_op(গ্রন্থি->op)) {
            চালু;
        }
        যদি (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, গ্রন্থি);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(গ্রন্থি), গ্রন্থি->name,
                fmt_আকার(ggml_nbytes(গ্রন্থি)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(গ্রন্থি),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, গ্রন্থি)], গ্রন্থি->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
                কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
                যদি (src == ফাঁকা) {
                    চালু;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_আকার(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

স্থির যৌক্তিক ggml_backend_sched_বাফার_supported(ggml_backend_sched_t sched, কাঠামো ggml_tensor * t, পূর্ণসংখ্যা backend_id) {
    ggml_backend_বাফার_t buf = t->view_src ? t->view_src->বাফার : t->বাফার;
    ggml_backend_বাফার_type_t buft = ফাঁকা;

    যদি (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } অন্যথা {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        পূর্ণসংখ্যা tensor_backend_id = tensor_backend_id(t);
        যদি (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        যদি (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    ফেরত buft != ফাঁকা && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

স্থির শূন্য ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, কাঠামো ggml_tensor * গ্রন্থি, পূর্ণসংখ্যা cur_backend_id, পূর্ণসংখ্যা * গ্রন্থি_backend_id) {
    যদি (ggml_backend_supports_op(sched->backends[cur_backend_id], গ্রন্থি)) {
        *গ্রন্থি_backend_id = cur_backend_id;
        SET_CAUSE(গ্রন্থি, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
শূন্য ggml_backend_sched_split_graph(ggml_backend_sched_t sched, কাঠামো ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = মিথ্যা;

    কাঠামো ggml_init_params params = {
        /* .mem_size =   */ sched->context_বাফার_আকার,
        /* .mem_buffer = */ sched->context_বাফার,
        /* .no_alloc =   */ সত্য
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    যদি (sched->ctx == ফাঁকা) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_পরবর্তী_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_leafs; i++) {
        কাঠামো ggml_tensor * leaf = graph->leafs[i];
        পূর্ণসংখ্যা * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        যদি (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        পূর্ণসংখ্যা * গ্রন্থি_backend_id = &tensor_backend_id(গ্রন্থি);
        // do not overwrite user assignments
        যদি (*গ্রন্থি_backend_id == -1) {
            *গ্রন্থি_backend_id = ggml_backend_sched_backend_id_from_cur(sched, গ্রন্থি);

#যদি 0
            // src
            যদি (গ্রন্থি->op == GGML_OP_NONE) {
                চালু;
            }

            জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
                কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
                যদি (src == ফাঁকা) {
                    চালু;
                }
                পূর্ণসংখ্যা * src_backend_id = &tensor_backend_id(src);
                যদি (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        পূর্ণসংখ্যা cur_backend_id = -1;
        জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
            কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
            যদি (ggml_is_view_op(গ্রন্থি->op)) {
                চালু;
            }
            পূর্ণসংখ্যা * গ্রন্থি_backend_id = &tensor_backend_id(গ্রন্থি);
            যদি (*গ্রন্থি_backend_id != -1) {
                যদি (*গ্রন্থি_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } অন্যথা {
                    cur_backend_id = *গ্রন্থি_backend_id;
                }
            } অন্যথা যদি (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, গ্রন্থি, cur_backend_id, গ্রন্থি_backend_id);
            }
        }
    }
    // expand gpu up
    {
        পূর্ণসংখ্যা cur_backend_id = -1;
        জন্য (পূর্ণসংখ্যা i = graph->n_nodes - 1; i >= 0; i--) {
            কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
            যদি (ggml_is_view_op(গ্রন্থি->op)) {
                চালু;
            }
            পূর্ণসংখ্যা * গ্রন্থি_backend_id = &tensor_backend_id(গ্রন্থি);
            যদি (*গ্রন্থি_backend_id != -1) {
                যদি (*গ্রন্থি_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } অন্যথা {
                    cur_backend_id = *গ্রন্থি_backend_id;
                }
            } অন্যথা যদি (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, গ্রন্থি, cur_backend_id, গ্রন্থি_backend_id);
            }
        }
    }
    // expand rest down
    {
        পূর্ণসংখ্যা cur_backend_id = -1;
        জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
            কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
            যদি (ggml_is_view_op(গ্রন্থি->op)) {
                চালু;
            }
            পূর্ণসংখ্যা * গ্রন্থি_backend_id = &tensor_backend_id(গ্রন্থি);
            যদি (*গ্রন্থি_backend_id != -1) {
                cur_backend_id = *গ্রন্থি_backend_id;
            } অন্যথা যদি (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, গ্রন্থি, cur_backend_id, গ্রন্থি_backend_id);
            }
        }
    }
    // expand rest up
    {
        পূর্ণসংখ্যা cur_backend_id = -1;
        জন্য (পূর্ণসংখ্যা i = graph->n_nodes - 1; i >= 0; i--) {
            কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
            যদি (ggml_is_view_op(গ্রন্থি->op)) {
                চালু;
            }
            পূর্ণসংখ্যা * গ্রন্থি_backend_id = &tensor_backend_id(গ্রন্থি);
            যদি (*গ্রন্থি_backend_id != -1) {
                cur_backend_id = *গ্রন্থি_backend_id;
            } অন্যথা যদি (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, গ্রন্থি, cur_backend_id, গ্রন্থি_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        যদি (ggml_is_view_op(গ্রন্থি->op)) {
            চালু;
        }
        পূর্ণসংখ্যা * গ্রন্থি_backend_id = &tensor_backend_id(গ্রন্থি);
        যদি (*গ্রন্থি_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            পূর্ণসংখ্যা n_supported_best = -1;
            জন্য (পূর্ণসংখ্যা b = 0; b < sched->n_backends; b++) {
                যদি (ggml_backend_supports_op(sched->backends[b], গ্রন্থি)) {
                    পূর্ণসংখ্যা n_supported = 0;
                    জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
                        কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
                        যদি (src == ফাঁকা) {
                            চালু;
                        }
                        যদি ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_বাফার_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    যদি (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *গ্রন্থি_backend_id = b;
                        SET_CAUSE(গ্রন্থি, "3.best");
                    }
                }
            }
        } অন্যথা {
            // assigned node: upgrade to higher prio backend if possible
            জন্য (পূর্ণসংখ্যা b = 0; b < *গ্রন্থি_backend_id; b++) {
                যদি (sched->bufts[b] == sched->bufts[*গ্রন্থি_backend_id] && ggml_backend_supports_op(sched->backends[b], গ্রন্থি)) {
                    যৌক্তিক supported = সত্য;
                    জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
                        কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
                        যদি (src == ফাঁকা) {
                            চালু;
                        }
                        যদি (!ggml_backend_sched_বাফার_supported(sched, src, b)) {
                            supported = মিথ্যা;
                            বিরতি;
                        }
                    }
                    যদি (supported) {
                        *গ্রন্থি_backend_id = b;
                        SET_CAUSE(গ্রন্থি, "3.upg");
                        বিরতি;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        পূর্ণসংখ্যা * cur_backend_id = &tensor_backend_id(গ্রন্থি);
        যদি (গ্রন্থি->view_src != ফাঁকা && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(গ্রন্থি->view_src);
            SET_CAUSE(গ্রন্থি, "4.vsrc");
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
            কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
            যদি (src == ফাঁকা) {
                চালু;
            }
            পূর্ণসংখ্যা * src_backend_id = &tensor_backend_id(src);
            যদি (*src_backend_id == -1) {
                যদি (src->view_src != ফাঁকা) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } অন্যথা {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        জন্য (পূর্ণসংখ্যা b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, গ্রন্থি, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        পূর্ণসংখ্যা i_split = 0;
        কাঠামো ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        পূর্ণসংখ্যা i = 0;
        জন্য (; i < graph->n_nodes; i++) {
            কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
            যদি (!ggml_is_view_op(গ্রন্থি->op)) {
                split->backend_id = tensor_backend_id(গ্রন্থি);
                বিরতি;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        পূর্ণসংখ্যা cur_backend_id = split->backend_id;
        জন্য (; i < graph->n_nodes; i++) {
            কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];

            যদি (ggml_is_view_op(গ্রন্থি->op)) {
                চালু;
            }

            ধ্রুবক পূর্ণসংখ্যা গ্রন্থি_backend_id = tensor_backend_id(গ্রন্থি);

            GGML_ASSERT(গ্রন্থি_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            যৌক্তিক need_new_split = মিথ্যা;
            যদি (গ্রন্থি_backend_id == cur_backend_id && split->n_inputs > 0) {
                জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
                    কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
                    যদি (src == ফাঁকা) {
                        চালু;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    যদি (src->বাফার != ফাঁকা && src->বাফার->usage == GGML_BACKEND_বাফার_USAGE_WEIGHTS) {
                        পূর্ণসংখ্যা src_backend_id = tensor_backend_id(src);
                        যদি (src_backend_id != cur_backend_id && !ggml_backend_sched_বাফার_supported(sched, src, cur_backend_id)) {
                            need_new_split = সত্য;
                            বিরতি;
                        }
                    }
                }
            }

            যদি (গ্রন্থি_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                যদি (i_split >= sched->splits_capacity) {
                    পূর্ণসংখ্যা old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * আকার(কাঠামো ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != ফাঁকা);
                    জন্য (পূর্ণসংখ্যা k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, আকার(কাঠামো ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = গ্রন্থি_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = গ্রন্থি_backend_id;
            }

            // find inputs that are not on the same backend
            জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
                কাঠামো ggml_tensor * src = গ্রন্থি->src[j];
                যদি (src == ফাঁকা) {
                    চালু;
                }

                size_t src_id = hash_id(src);
                ধ্রুবক পূর্ণসংখ্যা src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                যদি (src_backend_id != cur_backend_id && !ggml_backend_sched_বাফার_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    যদি (tensor_id_copy(src_id, cur_backend_id, 0) == ফাঁকা) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        জন্য (পূর্ণসংখ্যা c = 0; c < sched->n_copies; c++) {
                            কাঠামো ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            যদি (sched->n_copies > 1) {
                                ggml_set_প্রবেশ(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        পূর্ণসংখ্যা n_inputs = split->n_inputs++;
                        যদি (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    গ্রন্থি->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    যদি (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    যদি (sched->n_copies > 1) {
        জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_leafs; i++) {
            কাঠামো ggml_tensor * leaf = graph->leafs[i];
            যদি ((leaf->flags & GGML_TENSOR_FLAG_প্রবেশ) == 0) {
                চালু;
            }

            ধ্রুবক size_t leaf_id = hash_id(leaf);
            ধ্রুবক পূর্ণসংখ্যা leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            যদি (tensor_id_copy(leaf_id, leaf_backend_id, 0) == ফাঁকা) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                জন্য (পূর্ণসংখ্যা c = 0; c < sched->n_copies; c++) {
                    কাঠামো ggml_tensor * tensor_copy;
                    যদি (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } অন্যথা {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_প্রবেশ(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            পূর্ণসংখ্যা n_graph_inputs = sched->n_graph_inputs++;
            যদি (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        পূর্ণসংখ্যা * tmp = sched->গ্রন্থি_backend_ids;
        sched->গ্রন্থি_backend_ids = sched->পূর্ব_গ্রন্থি_backend_ids;
        sched->পূর্ব_গ্রন্থি_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->পূর্ব_leaf_backend_ids;
        sched->পূর্ব_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    কাঠামো ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](শূন্য * user_তথ্য, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_তথ্য;
            std::vector<ggml_tensor *> & keep = deps[until];
            যদি (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    জন্য (পূর্ণসংখ্যা i = 0; i < sched->n_splits; i++) {
        কাঠামো ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    পূর্ণসংখ্যা n_dep_nodes = 0;
    জন্য (ধ্রুবক auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    পূর্ণসংখ্যা যোগ_inputs = sched->n_graph_inputs;
    জন্য (পূর্ণসংখ্যা i = 0; i < sched->n_splits; i++) {
        যোগ_inputs += sched->splits[i].n_inputs;
    }
    পূর্ণসংখ্যা graph_আকার = std::max(graph->n_nodes, graph->n_leafs) + যোগ_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->পূর্ব_graph_আকার = sched->graph_আকার;
    sched->graph_আকার = graph_আকার;

    যদি (sched->graph.size < graph_আকার) {
        sched->graph.size = graph_আকার;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_আকার * আকার(কাঠামো ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_আকার * আকার(কাঠামো ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != ফাঁকা);
        GGML_ASSERT(sched->graph.leafs != ফাঁকা);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    কাঠামো ggml_cgraph * graph_copy = &sched->graph;

    পূর্ণসংখ্যা n_dep_nodes_added = 0;

    জন্য (পূর্ণসংখ্যা i = 0; i < sched->n_splits; i++) {
        কাঠামো ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        জন্য (পূর্ণসংখ্যা j = 0; j < split->n_inputs; j++) {
            প্রতিজ্ঞা(graph_copy->size > (graph_copy->n_nodes + 1));

            কাঠামো ggml_tensor * প্রবেশ = split->inputs[j];
            ধ্রুবক size_t প্রবেশ_id = hash_id(প্রবেশ);
            কাঠামো ggml_tensor * প্রবেশ_cpy = tensor_id_copy(প্রবেশ_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            কাঠামো ggml_tensor * প্রবেশ_dep = ggml_view_tensor(sched->ctx, প্রবেশ);
            প্রবেশ_dep->src[0] = প্রবেশ;
            sched->গ্রন্থি_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[প্রবেশ_id];
            graph_copy->nodes[graph_copy->n_nodes++] = প্রবেশ_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->গ্রন্থি_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = প্রবেশ_cpy;
        }

        জন্য (পূর্ণসংখ্যা j = split->i_start; j < split->i_end; j++) {
            প্রতিজ্ঞা(graph_copy->size > graph_copy->n_nodes);
            sched->গ্রন্থি_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            যদি (alloc_deps.empty()) {
                চালু;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            যদি (it != alloc_deps.end()) {
                ধ্রুবক std::vector<ggml_tensor *> & keep = it->second;
                জন্য (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    কাঠামো ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    জন্য (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    প্রতিজ্ঞা(graph_copy->size > graph_copy->n_nodes);
                    sched->গ্রন্থি_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    যদি (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        জন্য (পূর্ণসংখ্যা i = 0; i < sched->n_graph_inputs; i++) {
            কাঠামো ggml_tensor * প্রবেশ = sched->graph_inputs[i];
            size_t id = hash_id(প্রবেশ);
            পূর্ণসংখ্যা backend_id = tensor_backend_id(প্রবেশ);
            জন্য (পূর্ণসংখ্যা c = 0; c < sched->n_copies; c++) {
                কাঠামো ggml_tensor * প্রবেশ_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                প্রতিজ্ঞা(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = প্রবেশ_cpy;
            }
        }

        জন্য (পূর্ণসংখ্যা i = 0; i < sched->n_splits; i++) {
            কাঠামো ggml_backend_sched_split * split = &sched->splits[i];
            পূর্ণসংখ্যা backend_id = split->backend_id;
            জন্য (পূর্ণসংখ্যা j = 0; j < split->n_inputs; j++) {
                কাঠামো ggml_tensor * প্রবেশ = split->inputs[j];
                size_t id = hash_id(প্রবেশ);
                জন্য (পূর্ণসংখ্যা c = 0; c < sched->n_copies; c++) {
                    কাঠামো ggml_tensor * প্রবেশ_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    প্রতিজ্ঞা(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = প্রবেশ_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_leafs; i++) {
        কাঠামো ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        প্রতিজ্ঞা(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    জন্য (পূর্ণসংখ্যা i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_পরবর্তী_uid();
    }
}

স্থির যৌক্তিক ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    যৌক্তিক backend_ids_changed = মিথ্যা;
    জন্য (পূর্ণসংখ্যা i = 0; i < sched->graph.n_nodes; i++) {
        যদি (sched->গ্রন্থি_backend_ids[i] != sched->পূর্ব_গ্রন্থি_backend_ids[i] &&
            sched->bufts[sched->গ্রন্থি_backend_ids[i]] != sched->bufts[sched->পূর্ব_গ্রন্থি_backend_ids[i]]) {
            backend_ids_changed = সত্য;
            বিরতি;
        }
    }
    যদি (!backend_ids_changed) {
        জন্য (পূর্ণসংখ্যা i = 0; i < sched->graph.n_leafs; i++) {
            যদি (sched->leaf_backend_ids[i] != sched->পূর্ব_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->পূর্ব_leaf_backend_ids[i]]) {
                backend_ids_changed = সত্য;
                বিরতি;
            }
        }
    }

    // allocate graph
    যদি (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ধ্রুবক যৌক্তিক graph_আকার_changed = sched->পূর্ব_graph_আকার != sched->graph_আকার;

        যদি (sched->callback_realloc != ফাঁকা) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_আকার_changed, sched->callback_realloc_user_তথ্য);
        }

        যদি (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ধ্রুবক যৌক্তিক unexpected = !backend_ids_changed && !graph_আকার_changed;

            যদি (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_আকার, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        জন্য (পূর্ণসংখ্যা i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        যদি (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->গ্রন্থি_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            ফেরত মিথ্যা;
        }
        যদি (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ফেরত মিথ্যা;
        }
    }

    ফেরত সত্য;
}

স্থির যৌক্তিক ggml_backend_sched_is_host_weight(ধ্রুবক কাঠামো ggml_tensor * t) {
    ফেরত t->বাফার != ফাঁকা &&
        ggml_backend_বাফার_get_usage(t->বাফার) == GGML_BACKEND_বাফার_USAGE_WEIGHTS &&
        ggml_backend_বাফার_is_host(t->বাফার);
}

স্থির শূন্য ggml_backend_sched_copy_প্রবেশ(ggml_backend_sched_t sched, কাঠামো ggml_backend_sched_split * split, কাঠামো ggml_tensor * প্রবেশ) {
    ধ্রুবক পূর্ণসংখ্যা split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t প্রবেশ_backend = ggml_backend_sched_get_tensor_backend(sched, প্রবেশ);
    কাঠামো ggml_tensor * প্রবেশ_cpy = tensor_copy(প্রবেশ, split_backend_id, sched->cur_copy);

    যদি (প্রবেশ->flags & GGML_TENSOR_FLAG_প্রবেশ) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        যদি (sched->events[split_backend_id][sched->cur_copy] != ফাঁকা) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } অন্যথা {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(প্রবেশ, প্রবেশ_cpy);
        ফেরত;
    }

    // wait for the split backend to finish using the input before overwriting it
    যদি (sched->events[split_backend_id][sched->cur_copy] != ফাঁকা) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } অন্যথা {
        ggml_backend_synchronize(split_backend);
    }

    যদি (sched->callback_copy != ফাঁকা && ggml_backend_sched_is_host_weight(প্রবেশ) &&
        sched->callback_copy(split_backend, প্রবেশ, প্রবেশ_cpy, &split->graph, sched->callback_copy_user_তথ্য)) {
        ফেরত;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    যদি (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(প্রবেশ_backend, split_backend, প্রবেশ, প্রবেশ_cpy)) {
        ggml_backend_synchronize(প্রবেশ_backend);
        যদি (sched->events[split_backend_id][sched->cur_copy] != ফাঁকা) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } অন্যথা {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(প্রবেশ, প্রবেশ_cpy);
    }
}

স্থির গণনা ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    কাঠামো ggml_backend_sched_split * splits = sched->splits;

    পূর্ণসংখ্যা পূর্ব_backend_id = -1;

    জন্য (পূর্ণসংখ্যা split_id = 0; split_id < sched->n_splits; split_id++) {
        কাঠামো ggml_backend_sched_split * split = &splits[split_id];
        পূর্ণসংখ্যা split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        যদি (split->n_inputs == 0 && পূর্ব_backend_id >= 0 && পূর্ব_backend_id != split_backend_id) {
            যদি (sched->events[পূর্ব_backend_id][sched->cur_copy] != ফাঁকা) {
                ggml_backend_event_synchronize(sched->events[পূর্ব_backend_id][sched->cur_copy]);
            } অন্যথা {
                ggml_backend_synchronize(sched->backends[পূর্ব_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        জন্য (পূর্ণসংখ্যা প্রবেশ_id = 0; প্রবেশ_id < split->n_inputs; প্রবেশ_id++) {
            যদি (!ggml_backend_sched_is_host_weight(split->inputs[প্রবেশ_id])) {
                ggml_backend_sched_copy_প্রবেশ(sched, split, split->inputs[প্রবেশ_id]);
            }
        }
        জন্য (পূর্ণসংখ্যা প্রবেশ_id = 0; প্রবেশ_id < split->n_inputs; প্রবেশ_id++) {
            যদি (ggml_backend_sched_is_host_weight(split->inputs[প্রবেশ_id])) {
                ggml_backend_sched_copy_প্রবেশ(sched, split, split->inputs[প্রবেশ_id]);
            }
        }

        যদি (!sched->callback_eval) {
            গণনা ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            যদি (ec != GGML_STATUS_SUCCESS) {
                ফেরত ec;
            }
        } অন্যথা {
            // similar to ggml_backend_compare_graph_backend
            জন্য (পূর্ণসংখ্যা j0 = 0; j0 < split->graph.n_nodes; j0++) {
                কাঠামো ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                যৌক্তিক need = sched->callback_eval(t, সত্য, sched->callback_eval_user_তথ্য);

                পূর্ণসংখ্যা j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                যখন (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, সত্য, sched->callback_eval_user_তথ্য);
                }

                কাঠামো ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                গণনা ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                যদি (ec != GGML_STATUS_SUCCESS) {
                    ফেরত ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                যদি (need && !sched->callback_eval(t, মিথ্যা, sched->callback_eval_user_তথ্য)) {
                    বিরতি;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        যদি (sched->events[split_backend_id][sched->cur_copy] != ফাঁকা) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        পূর্ব_backend_id = split_backend_id;
    }

    ফেরত GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_বাফার_type_t * bufts,
        পূর্ণসংখ্যা n_backends,
        size_t graph_আকার,
        যৌক্তিক parallel,
        যৌক্তিক op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    কাঠামো ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, আকার(কাঠামো ggml_backend_sched));

    ধ্রুবক অক্ষর * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    ধ্রুবক অক্ষর * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_আকার);
    sched->hv_tensor_backend_ids = (পূর্ণসংখ্যা *) malloc(sched->hash_set.size * আকার(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * আকার(কাঠামো ggml_tensor *));

    ধ্রুবক size_t ggml_sched_max_splits = graph_আকার; // at most there is one split for each node in the graph
    ধ্রুবক size_t nodes_আকার = graph_আকার + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->গ্রন্থি_backend_ids = (পূর্ণসংখ্যা *) calloc(nodes_আকার, আকার(sched->গ্রন্থি_backend_ids[0]));
    sched->leaf_backend_ids = (পূর্ণসংখ্যা *) calloc(nodes_আকার, আকার(sched->leaf_backend_ids[0]));
    sched->পূর্ব_গ্রন্থি_backend_ids = (পূর্ণসংখ্যা *) calloc(nodes_আকার, আকার(sched->পূর্ব_গ্রন্থি_backend_ids[0]));
    sched->পূর্ব_leaf_backend_ids = (পূর্ণসংখ্যা *) calloc(nodes_আকার, আকার(sched->পূর্ব_leaf_backend_ids[0]));

    sched->graph_আকার      = 0;
    sched->পূর্ব_graph_আকার = 0;

    sched->context_বাফার_আকার = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*আকার(কাঠামো ggml_tensor) + ggml_graph_overhead_custom(graph_আকার, মিথ্যা);
    sched->context_বাফার = (অক্ষর *) malloc(sched->context_বাফার_আকার);

    ধ্রুবক পূর্ণসংখ্যা initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, আকার(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (কাঠামো ggml_tensor **) calloc(sched->graph_inputs_capacity, আকার(কাঠামো ggml_tensor *));

    জন্য (পূর্ণসংখ্যা b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_বাফার_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        যদি (sched->n_copies > 1) {
            জন্য (পূর্ণসংখ্যা c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    ফেরত sched;
}

শূন্য ggml_backend_sched_free(ggml_backend_sched_t sched) {
    যদি (sched == ফাঁকা) {
        ফেরত;
    }
    জন্য (পূর্ণসংখ্যা b = 0; b < sched->n_backends; b++) {
        জন্য (পূর্ণসংখ্যা c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    জন্য (পূর্ণসংখ্যা i = 0; i < sched->splits_capacity; i++) {
        মুক্ত(sched->splits[i].inputs);
    }
    মুক্ত(sched->splits);
    মুক্ত(sched->graph_inputs);
    মুক্ত(sched->hv_tensor_backend_ids);
    মুক্ত(sched->hv_tensor_copies);
    মুক্ত(sched->গ্রন্থি_backend_ids);
    মুক্ত(sched->leaf_backend_ids);
    মুক্ত(sched->পূর্ব_গ্রন্থি_backend_ids);
    মুক্ত(sched->পূর্ব_leaf_backend_ids);
    মুক্ত(sched->context_বাফার);
    মুক্ত(sched->graph.nodes);
    মুক্ত(sched->graph.leafs);
    মুক্ত(sched);
}

শূন্য ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    যদি (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * আকার(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * আকার(কাঠামো ggml_tensor *));
        sched->is_reset = সত্য;
    }
    sched->is_alloc = মিথ্যা;
}

শূন্য ggml_backend_sched_reserve_আকার(ggml_backend_sched_t sched, কাঠামো ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((পূর্ণসংখ্যা)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_আকার(sched->galloc, &sched->graph, sched->গ্রন্থি_backend_ids, sched->leaf_backend_ids, sizes);
}

যৌক্তিক ggml_backend_sched_reserve(ggml_backend_sched_t sched, কাঠামো ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((পূর্ণসংখ্যা)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    যদি (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->গ্রন্থি_backend_ids, sched->leaf_backend_ids)) {
        ফেরত মিথ্যা;
    }

    ggml_backend_sched_reset(sched);

    ফেরত সত্য;
}

যৌক্তিক ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, কাঠামো ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((পূর্ণসংখ্যা)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->পরবর্তী_copy;
    sched->পরবর্তী_copy = (sched->পরবর্তী_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    যদি (!ggml_backend_sched_alloc_splits(sched)) {
        ফেরত মিথ্যা;
    }

    sched->is_alloc = সত্য;

    ফেরত সত্য;
}

গণনা ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, কাঠামো ggml_cgraph * graph) {
    গণনা ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    ফেরত err;
}

গণনা ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, কাঠামো ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    যদি (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    যদি (!sched->is_alloc) {
        যদি (!ggml_backend_sched_alloc_graph(sched, graph)) {
            ফেরত GGML_STATUS_ALLOC_FAILED;
        }
    }

    ফেরত ggml_backend_sched_compute_splits(sched);
}

শূন্য ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    জন্য (পূর্ণসংখ্যা i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    যদি (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->পরবর্তী_copy = 0;
    }
}

শূন্য ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, শূন্য * user_তথ্য) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_তথ্য = user_তথ্য;
}

শূন্য ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, শূন্য * user_তথ্য) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_তথ্য = user_তথ্য;
}

শূন্য ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, শূন্য * user_তথ্য) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_তথ্য = user_তথ্য;
}

পূর্ণসংখ্যা ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ফেরত sched->n_splits;
}

পূর্ণসংখ্যা ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ফেরত sched->n_copies;
}

পূর্ণসংখ্যা ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ফেরত sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, পূর্ণসংখ্যা i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    ফেরত sched->backends[i];
}

ggml_backend_বাফার_type_t ggml_backend_sched_get_বাফার_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    পূর্ণসংখ্যা backend_সূচক = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_সূচক >= 0 && backend_সূচক < sched->n_backends);

    ফেরত sched->bufts[backend_সূচক];
}

size_t ggml_backend_sched_get_বাফার_আকার(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    পূর্ণসংখ্যা backend_সূচক = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_সূচক >= 0 && backend_সূচক < sched->n_backends);

    ফেরত ggml_gallocr_get_বাফার_আকার(sched->galloc, backend_সূচক);
}

শূন্য ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, কাঠামো ggml_tensor * গ্রন্থি, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    পূর্ণসংখ্যা backend_সূচক = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_সূচক >= 0 && backend_সূচক < sched->n_backends);
    tensor_backend_id(গ্রন্থি) = backend_সূচক;
    SET_CAUSE(গ্রন্থি, "usr");
    sched->is_reset = মিথ্যা;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, কাঠামো ggml_tensor * গ্রন্থি) {
    GGML_ASSERT(sched);
    পূর্ণসংখ্যা backend_সূচক = tensor_backend_id(গ্রন্থি);
    যদি (backend_সূচক == -1) {
        ফেরত ফাঁকা;
    }
    ফেরত sched->backends[backend_সূচক];
}

// utils

যৌক্তিক ggml_op_alloc_আকার_may_expand(গণনা ggml_op op) {
    নির্বাচন (op) {
        ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
        ক্ষেত্র GGML_OP_MUL_MAT:
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
        ক্ষেত্র GGML_OP_CUMSUM:
        ক্ষেত্র GGML_OP_ARGSORT:
        ক্ষেত্র GGML_OP_শীর্ষ_K:
        ক্ষেত্র GGML_OP_ARGMAX:
            ফেরত সত্য;
        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;
    }
}

গণনা ggml_status ggml_backend_view_init(কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->বাফার == ফাঁকা);
    GGML_ASSERT(tensor->view_src != ফাঁকা);
    GGML_ASSERT(tensor->view_src->বাফার != ফাঁকা);
    GGML_ASSERT(tensor->view_src->তথ্য != ফাঁকা);

    tensor->বাফার = tensor->view_src->বাফার;
    tensor->তথ্য = (অক্ষর *)tensor->view_src->তথ্য + tensor->view_offs;
    ফেরত ggml_backend_বাফার_init_tensor(tensor->বাফার, tensor);
}

গণনা ggml_status ggml_backend_tensor_alloc(ggml_backend_বাফার_t বাফার, কাঠামো ggml_tensor * tensor, শূন্য * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->বাফার == ফাঁকা);
    GGML_ASSERT(tensor->তথ্য == ফাঁকা);
    GGML_ASSERT(tensor->view_src == ফাঁকা);
    GGML_ASSERT(addr >= ggml_backend_বাফার_get_base(বাফার));
    GGML_ASSERT(ggml_backend_বাফার_is_meta(বাফার) ||
        (অক্ষর *) addr + ggml_backend_বাফার_get_alloc_আকার(বাফার, tensor) <=
        (অক্ষর *) ggml_backend_বাফার_get_base(বাফার) + ggml_backend_বাফার_get_আকার(বাফার));

    tensor->বাফার = বাফার;
    tensor->তথ্য = addr;
    ফেরত ggml_backend_বাফার_init_tensor(বাফার, tensor);
}

স্থির কাঠামো ggml_tensor * graph_copy_dup_tensor(কাঠামো ggml_hash_set hash_set, কাঠামো ggml_tensor ** গ্রন্থি_copies,
    কাঠামো ggml_context * ctx_allocated, কাঠামো ggml_context * ctx_unallocated, কাঠামো ggml_tensor * src) {

    GGML_ASSERT(src != ফাঁকা);
    GGML_ASSERT(src->তথ্য && "graph must be allocated");

    size_t id = ggml_hash_ঢোকাও(&hash_set, src);
    যদি (id == GGML_HASHSET_ALREADY_EXISTS) {
        ফেরত গ্রন্থি_copies[ggml_hash_find(&hash_set, src)];
    }

    কাঠামো ggml_tensor * dst = ggml_dup_tensor_layout(src->তথ্য && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    যদি (src->view_src != ফাঁকা) {
        dst->view_src = graph_copy_dup_tensor(hash_set, গ্রন্থি_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, আকার(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        কাঠামো ggml_tensor * s = src->src[i];
        যদি (s == ফাঁকা) {
            চালু;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, গ্রন্থি_copies, ctx_allocated, ctx_unallocated, s);
    }

    গ্রন্থি_copies[id] = dst;
    ফেরত dst;
}

স্থির শূন্য graph_copy_init_tensor(কাঠামো ggml_hash_set * hash_set, কাঠামো ggml_tensor ** গ্রন্থি_copies, যৌক্তিক * গ্রন্থি_init, কাঠামো ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    যদি (গ্রন্থি_init[id]) {
        ফেরত;
    }
    গ্রন্থি_init[id] = সত্য;

    কাঠামো ggml_tensor * dst = গ্রন্থি_copies[id];
    যদি (dst->view_src != ফাঁকা) {
        graph_copy_init_tensor(hash_set, গ্রন্থি_copies, গ্রন্থি_init, src->view_src);
        গণনা ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    অন্যথা {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        কাঠামো ggml_tensor * s = src->src[i];
        যদি (s == ফাঁকা) {
            চালু;
        }
        graph_copy_init_tensor(hash_set, গ্রন্থি_copies, গ্রন্থি_init, s);
    }
}

কাঠামো ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, কাঠামো ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    কাঠামো ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    কাঠামো ggml_tensor ** গ্রন্থি_copies = (ggml_tensor **) calloc(hash_set.size, আকার(গ্রন্থি_copies[0])); // NOLINT
    যৌক্তিক * গ্রন্থি_init = (যৌক্তিক *) calloc(hash_set.size, আকার(গ্রন্থি_init[0]));

    কাঠামো ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, মিথ্যা),
        /* .mem_buffer = */ ফাঁকা,
        /* .no_alloc   = */ সত্য
    };

    কাঠামো ggml_context * ctx_allocated = ggml_init(params);
    কাঠামো ggml_context * ctx_unallocated = ggml_init(params);

    যদি (ctx_allocated == ফাঁকা || ctx_unallocated == ফাঁকা) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        মুক্ত(গ্রন্থি_copies);
        মুক্ত(গ্রন্থি_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        ফেরত {
            /* .buffer           = */ ফাঁকা,
            /* .ctx_allocated    = */ ফাঁকা,
            /* .ctx_unallocated  = */ ফাঁকা,
            /* .graph            = */ ফাঁকা,
        };
    }

    // dup nodes
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, গ্রন্থি_copies, ctx_allocated, ctx_unallocated, গ্রন্থি);
    }

    // allocate nodes
    ggml_backend_বাফার_t বাফার = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    যদি (বাফার == ফাঁকা) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        মুক্ত(গ্রন্থি_copies);
        মুক্ত(গ্রন্থি_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        ফেরত {
            /* .buffer           = */ ফাঁকা,
            /* .ctx_allocated    = */ ফাঁকা,
            /* .ctx_unallocated  = */ ফাঁকা,
            /* .graph            = */ ফাঁকা,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, গ্রন্থি_copies, গ্রন্থি_init, গ্রন্থি);
    }

    // build graph copy
    কাঠামো ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, মিথ্যা);
    জন্য (পূর্ণসংখ্যা i = 0; i < graph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = graph->nodes[i];
        কাঠামো ggml_tensor * গ্রন্থি_copy = গ্রন্থি_copies[ggml_hash_find(&hash_set, গ্রন্থি)];
        graph_copy->nodes[i] = গ্রন্থি_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    মুক্ত(গ্রন্থি_copies);
    মুক্ত(গ্রন্থি_init);

    ফেরত {
        /* .buffer           = */ বাফার,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

শূন্য ggml_backend_graph_copy_free(কাঠামো ggml_backend_graph_copy copy) {
    ggml_backend_বাফার_free(copy.বাফার);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

যৌক্তিক ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, কাঠামো ggml_cgraph * graph, ggml_backend_eval_callback callback, শূন্য * user_তথ্য, কাঠামো ggml_tensor ধ্রুবক * ধ্রুবক * test_nodes, size_t অঙ্ক_test_nodes) {
    কাঠামো ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    যদি (copy.বাফার == ফাঁকা) {
        ফেরত মিথ্যা;
    }

    কাঠামো ggml_cgraph * g1 = graph;
    কাঠামো ggml_cgraph * g2 = copy.graph;

    প্রতিজ্ঞা(g1->n_nodes == g2->n_nodes);

    যদি (অঙ্ক_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        যৌক্তিক verified = মিথ্যা;
        জন্য (পূর্ণসংখ্যা i = 0; i < g1->n_nodes; i++) {
            জন্য (size_t j = 0; j < অঙ্ক_test_nodes; ++j) {
                যদি (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_তথ্য);
                    verified = সত্য;
                }
            }
        }
        GGML_ASSERT(verified);
    } অন্যথা {
        জন্য (পূর্ণসংখ্যা i = 0; i < g1->n_nodes; i++) {
            কাঠামো ggml_tensor * t1 = g1->nodes[i];
            কাঠামো ggml_tensor * t2 = g2->nodes[i];

            প্রতিজ্ঞা(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            কাঠামো ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            কাঠামো ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            যদি (ggml_is_view_op(t1->op)) {
                চালু;
            }

            // compare results, calculate rms etc
            যদি (!callback(i, t1, t2, user_তথ্য)) {
                বিরতি;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    ফেরত সত্য;
}

// CPU backend - buffer

স্থির শূন্য * ggml_backend_cpu_বাফার_get_base(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);
    uintptr_t তথ্য = (uintptr_t)বাফার->context;

    // align the buffer
    যদি (তথ্য % TENSOR_ALIGNMENT != 0) {
        তথ্য = GGML_PAD(তথ্য, TENSOR_ALIGNMENT);
    }

    ফেরত (শূন্য *)তথ্য;
}

স্থির শূন্য ggml_backend_cpu_বাফার_free_বাফার(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);
    ggml_aligned_free(বাফার->context, বাফার->size);
}

স্থির শূন্য ggml_backend_cpu_বাফার_memset_tensor(ggml_backend_বাফার_t বাফার, কাঠামো ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((অক্ষর *)tensor->তথ্য + offset, value, size);

    GGML_UNUSED(বাফার);
}

স্থির শূন্য ggml_backend_cpu_বাফার_set_tensor(ggml_backend_বাফার_t বাফার, কাঠামো ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((অক্ষর *)tensor->তথ্য + offset, তথ্য, size);

    GGML_UNUSED(বাফার);
}

স্থির শূন্য ggml_backend_cpu_বাফার_get_tensor(ggml_backend_বাফার_t বাফার, ধ্রুবক কাঠামো ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(তথ্য, (ধ্রুবক অক্ষর *)tensor->তথ্য + offset, size);

    GGML_UNUSED(বাফার);
}

স্থির যৌক্তিক ggml_backend_cpu_বাফার_cpy_tensor(ggml_backend_বাফার_t বাফার, ধ্রুবক কাঠামো ggml_tensor * src, কাঠামো ggml_tensor * dst) {
    GGML_ASSERT(src);
    যদি (ggml_backend_বাফার_is_host(src->বাফার)) {
        memcpy(dst->তথ্য, src->তথ্য, ggml_nbytes(src));
        ফেরত সত্য;
    }
    ফেরত মিথ্যা;

    GGML_UNUSED(বাফার);
}

স্থির শূন্য ggml_backend_cpu_বাফার_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    GGML_ASSERT(বাফার);
    memset(বাফার->context, value, বাফার->size);
}

স্থির ধ্রুবক কাঠামো ggml_backend_বাফার_i ggml_backend_cpu_বাফার_i = {
    /* .free_buffer     = */ ggml_backend_cpu_বাফার_free_বাফার,
    /* .get_base        = */ ggml_backend_cpu_বাফার_get_base,
    /* .init_tensor     = */ ফাঁকা, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_বাফার_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_বাফার_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_বাফার_get_tensor,
    /* .set_tensor_2d   = */ ফাঁকা,
    /* .get_tensor_2d   = */ ফাঁকা,
    /* .cpy_tensor      = */ ggml_backend_cpu_বাফার_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_বাফার_clear,
    /* .reset           = */ ফাঁকা,
};

স্থির ধ্রুবক কাঠামো ggml_backend_বাফার_i ggml_backend_cpu_বাফার_from_ptr_i = {
    /* .free_buffer     = */ ফাঁকা, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_বাফার_get_base,
    /* .init_tensor     = */ ফাঁকা, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_বাফার_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_বাফার_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_বাফার_get_tensor,
    /* .set_tensor_2d   = */ ফাঁকা,
    /* .get_tensor_2d   = */ ফাঁকা,
    /* .cpy_tensor      = */ ggml_backend_cpu_বাফার_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_বাফার_clear,
    /* .reset           = */ ফাঁকা,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

স্থির ধ্রুবক অক্ষর * ggml_backend_cpu_বাফার_type_get_name(ggml_backend_বাফার_type_t buft) {
    ফেরত "CPU";

    GGML_UNUSED(buft);
}

স্থির ggml_backend_বাফার_t ggml_backend_cpu_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size) {
    শূন্য * তথ্য = ggml_aligned_malloc(size);

    যদি (তথ্য == ফাঁকা) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        ফেরত ফাঁকা;
    }

    ফেরত ggml_backend_বাফার_init(buft, ggml_backend_cpu_বাফার_i, তথ্য, size);
}

স্থির size_t ggml_backend_cpu_বাফার_type_get_alignment(ggml_backend_বাফার_type_t buft) {
    ফেরত TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

স্থির যৌক্তিক ggml_backend_cpu_বাফার_type_is_host(ggml_backend_বাফার_type_t buft) {
    ফেরত সত্য;

    GGML_UNUSED(buft);
}

ggml_backend_বাফার_type_t ggml_backend_cpu_বাফার_type(শূন্য) {
    স্থির কাঠামো ggml_backend_বাফার_type ggml_backend_cpu_বাফার_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_বাফার_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_বাফার_type_alloc_বাফার,
            /* .alloc_buffer_n      = */ ফাঁকা,
            /* .get_alignment       = */ ggml_backend_cpu_বাফার_type_get_alignment,
            /* .get_max_size        = */ ফাঁকা, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ফাঁকা, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ফাঁকা,
            /* .is_host             = */ ggml_backend_cpu_বাফার_type_is_host,
        },
        /* .device  = */ ফাঁকা, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ফাঁকা,
    };

    ফেরত &ggml_backend_cpu_বাফার_type;
}

স্থির ধ্রুবক অক্ষর * ggml_backend_cpu_বাফার_from_ptr_type_get_name(ggml_backend_বাফার_type_t buft) {
    ফেরত "CPU_Mapped";

    GGML_UNUSED(buft);
}

স্থির ggml_backend_বাফার_type_t ggml_backend_cpu_বাফার_from_ptr_type(শূন্য) {
    স্থির কাঠামো ggml_backend_বাফার_type ggml_backend_cpu_বাফার_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_বাফার_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_বাফার_type_alloc_বাফার,
            /* .alloc_buffer_n      = */ ফাঁকা,
            /* .get_alignment       = */ ggml_backend_cpu_বাফার_type_get_alignment,
            /* .get_max_size        = */ ফাঁকা, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ফাঁকা, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ফাঁকা,
            /* .is_host             = */ ggml_backend_cpu_বাফার_type_is_host,
        },
        /* .device  = */ ফাঁকা, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ফাঁকা,
    };

    ফেরত &ggml_backend_cpu_বাফার_type;
}

ggml_backend_বাফার_t ggml_backend_cpu_বাফার_from_ptr(শূন্য * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    ফেরত ggml_backend_বাফার_init(ggml_backend_cpu_বাফার_from_ptr_type(), ggml_backend_cpu_বাফার_from_ptr_i, ptr, size);
}
