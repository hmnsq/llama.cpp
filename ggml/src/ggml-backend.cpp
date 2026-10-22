// Note: porting this file to C++ is a work in progress

#నిర్వచితమైతే _WIN32
#నిర్వచించు WIN32_LEAN_AND_సగటు
#నిర్వచితం_కాకపోతే NOMINMAX
#   define NOMINMAX
#అయితే_ముగింపు
#అన్తర్భావ <windows.h>
#అయితే_ముగింపు

#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-alloc.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <limits.h>
#అన్తర్భావ <stdarg.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <algorithm>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <vector>

#నిర్వచితమైతే __APPLE__
#అన్తర్భావ <sys/types.h>
#అన్తర్భావ <sys/sysctl.h>
#అయితే_ముగింపు


// backend buffer type

స్థిరరాశి అక్షరం * ggml_backend_buft_name(ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(buft);
    తిరిగి buft->iface.get_name(buft);
}

ggml_backend_బఫరు_t ggml_backend_buft_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    ఒకవేళ (size == 0) {
        // return a dummy buffer for zero-sized allocations
        తిరిగి ggml_backend_బఫరు_init(buft, {}, ఖాళీ, 0);
    }
    తిరిగి buft->iface.alloc_బఫరు(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
నిర్మాణం ggml_backend_buft_alloc_బఫరు_n_plan_అంశం {
    size_t size;  // total bytes for this buffer
    పూర్ణాంకం    మొదటి; // first tensor index (inclusive)
    పూర్ణాంకం    చివరి;  // last tensor index (exclusive)
};

ఉపయోగించి ggml_backend_buft_alloc_బఫరు_n_plan_t = std::vector<ggml_backend_buft_alloc_బఫరు_n_plan_అంశం>;

స్థిర ggml_backend_buft_alloc_బఫరు_n_plan_t ggml_backend_buft_alloc_బఫరు_n_plan(
        ggml_backend_బఫరు_type_t buft, నిర్మాణం ggml_tensor ** tensors, పూర్ణాంకం n_tensors) {
    ggml_backend_buft_alloc_బఫరు_n_plan_t plan;

    స్థిరరాశి size_t alignment = ggml_backend_buft_get_alignment(buft);
    స్థిరరాశి size_t max_size  = ggml_backend_buft_get_max_కొలత(buft);

    size_t cur_buf_కొలత = 0;
    పూర్ణాంకం    మొదటి        = 0;

    కొరకు (పూర్ణాంకం i = 0; i < n_tensors; i++) {
        size_t this_కొలత = 0;
        నిర్మాణం ggml_tensor * t = tensors[i];
        ఒకవేళ (t->దత్తాంశం == ఖాళీ && t->view_src == ఖాళీ) {
            this_కొలత = GGML_PAD(ggml_backend_buft_get_alloc_కొలత(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        ఒకవేళ (cur_buf_కొలత > 0 && (cur_buf_కొలత + this_కొలత) > max_size) {
            plan.push_back({ cur_buf_కొలత, మొదటి, i });
            cur_buf_కొలత = this_కొలత;
            మొదటి        = i;
        } లేకపోతే {
            cur_buf_కొలత += this_కొలత;
        }
    }

    ఒకవేళ (cur_buf_కొలత > 0) {
        plan.push_back({ cur_buf_కొలత, మొదటి, n_tensors });
    }

    తిరిగి plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
స్థిర ggml_backend_బఫరు_t ggml_backend_buft_alloc_బఫరు_n_default(ggml_backend_బఫరు_type_t buft, నిర్మాణం ggml_tensor ** tensors, పూర్ణాంకం n_tensors) {
    స్థిరరాశి ggml_backend_buft_alloc_బఫరు_n_plan_t plan = ggml_backend_buft_alloc_బఫరు_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_బఫరు_t> buffers;
    buffers.reserve(plan.size());

    కొరకు (స్థిరరాశి ggml_backend_buft_alloc_బఫరు_n_plan_అంశం & అంశం : plan) {
        ggml_backend_బఫరు_t బఫరు = ggml_backend_buft_alloc_బఫరు(buft, అంశం.size);
        ఒకవేళ (బఫరు == ఖాళీ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), అంశం.size);
            కొరకు (ggml_backend_బఫరు_t b : buffers) {
                ggml_backend_బఫరు_free(b);
            }
            తిరిగి ఖాళీ;
        }

        నిర్మాణం ggml_tallocr tallocr = ggml_tallocr_new(బఫరు);

        // allocate tensors in the current buffer
        నిర్మాణం ggml_tensor * t_failed = ఖాళీ;
        కొరకు (పూర్ణాంకం j = అంశం.మొదటి; j < అంశం.చివరి; j++) {
            నిర్మాణం ggml_tensor * t = tensors[j];
            ఒకవేళ (t->దత్తాంశం == ఖాళీ) {
                ఒకవేళ (t->view_src == ఖాళీ) {
                    ఒకవేళ (ggml_tallocr_alloc(&tallocr, t) != GGML_స్థితి_SUCCESS) {
                        t_failed = t;
                        ఆపు;
                    }
                } లేకపోతే ఒకవేళ (t->బఫరు == ఖాళీ) {
                    ఒకవేళ (ggml_backend_view_init(t) != GGML_స్థితి_SUCCESS) {
                        t_failed = t;
                        ఆపు;
                    }
                }
            } లేకపోతే {
                ఒకవేళ (t->view_src != ఖాళీ && t->బఫరు == ఖాళీ) {
                    // view of a pre-allocated tensor
                    ఒకవేళ (ggml_backend_view_init(t) != GGML_స్థితి_SUCCESS) {
                        t_failed = t;
                        ఆపు;
                    }
                }
            }
        }
        ఒకవేళ (t_failed != ఖాళీ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            కొరకు (ggml_backend_బఫరు_t b : buffers) {
                ggml_backend_బఫరు_free(b);
            }
            ggml_backend_బఫరు_free(బఫరు);
            తిరిగి ఖాళీ;
        }

        buffers.push_back(బఫరు);
    }

    ఒకవేళ (buffers.ఖాళీలు()) {
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (buffers.size() == 1) {
        తిరిగి buffers[0];
    }

    తిరిగి ggml_backend_multi_బఫరు_alloc_బఫరు(buffers.దత్తాంశం(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
స్థిర size_t ggml_backend_buft_get_alloc_కొలత_n_default(ggml_backend_బఫరు_type_t buft, నిర్మాణం ggml_tensor ** tensors, పూర్ణాంకం n_tensors) {
    స్థిరరాశి ggml_backend_buft_alloc_బఫరు_n_plan_t plan = ggml_backend_buft_alloc_బఫరు_n_plan(buft, tensors, n_tensors);

    size_t మొత్తం = 0;
    కొరకు (స్థిరరాశి ggml_backend_buft_alloc_బఫరు_n_plan_అంశం & అంశం : plan) {
        మొత్తం += అంశం.size;
    }
    తిరిగి మొత్తం;
}

ggml_backend_బఫరు_t ggml_backend_buft_alloc_బఫరు_n(ggml_backend_బఫరు_type_t buft, నిర్మాణం ggml_tensor ** tensors, పూర్ణాంకం n_tensors) {
    GGML_ASSERT(buft);
    ఒకవేళ (buft->iface.alloc_బఫరు_n) {
        తిరిగి buft->iface.alloc_బఫరు_n(buft, tensors, n_tensors);
    }
    తిరిగి ggml_backend_buft_alloc_బఫరు_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(buft);
    తిరిగి buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    ఒకవేళ (buft->iface.get_max_కొలత) {
        తిరిగి buft->iface.get_max_కొలత(buft);
    }
    తిరిగి SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    ఒకవేళ (buft->iface.get_alloc_కొలత) {
        size_t size = buft->iface.get_alloc_కొలత(buft, tensor);
        ప్రతిజ్ఞా(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_ఖాళీలు(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_కొలత_may_expand(tensor->op));

        తిరిగి size;
    }
    తిరిగి ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_కొలత_n(ggml_backend_బఫరు_type_t buft, నిర్మాణం ggml_tensor ** tensors, పూర్ణాంకం n_tensors) {
    GGML_ASSERT(buft);
    ఒకవేళ (buft->iface.get_alloc_కొలత_n) {
        తిరిగి buft->iface.get_alloc_కొలత_n(buft, tensors, n_tensors);
    }
    తిరిగి ggml_backend_buft_get_alloc_కొలత_n_default(buft, tensors, n_tensors);
}

తార్కిక ggml_backend_buft_is_host(ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(buft);
    ఒకవేళ (buft->iface.is_host) {
        తిరిగి buft->iface.is_host(buft);
    }
    తిరిగి అబద్ధం;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(buft);
    తిరిగి buft->device;
}

// backend buffer

ggml_backend_బఫరు_t ggml_backend_బఫరు_init(
               ggml_backend_బఫరు_type_t buft,
        నిర్మాణం ggml_backend_బఫరు_i      iface,
               శూన్య *                     context,
               size_t                     size) {
    ggml_backend_బఫరు_t బఫరు = కొత్త ggml_backend_బఫరు {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_బఫరు_USAGE_ANY
    };

    తిరిగి బఫరు;
}

స్థిరరాశి అక్షరం * ggml_backend_బఫరు_name(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి ggml_backend_buft_name(ggml_backend_బఫరు_get_type(బఫరు));
}

శూన్య ggml_backend_బఫరు_free(ggml_backend_బఫరు_t బఫరు) {
    ఒకవేళ (బఫరు == ఖాళీ) {
        తిరిగి;
    }

    ఒకవేళ (బఫరు->iface.free_బఫరు != ఖాళీ) {
        బఫరు->iface.free_బఫరు(బఫరు);
    }
    తొలగించు బఫరు;
}

size_t ggml_backend_బఫరు_get_కొలత(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);
    తిరిగి బఫరు->size;
}

శూన్య * ggml_backend_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);
    // get_base is optional if the buffer is zero-sized
    ఒకవేళ (!ggml_backend_బఫరు_is_meta(బఫరు) && బఫరు->size == 0) {
        తిరిగి ఖాళీ;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    ఒకవేళ (!బఫరు->iface.get_పునాది) {
        తిరిగి ఖాళీ;
    }

    శూన్య * పునాది = బఫరు->iface.get_పునాది(బఫరు);

    GGML_ASSERT(పునాది != ఖాళీ && "backend buffer base cannot be NULL");

    తిరిగి పునాది;
}

గణన ggml_స్థితి ggml_backend_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(బఫరు);
    // init_tensor is optional
    ఒకవేళ (బఫరు->iface.init_tensor) {
        తిరిగి బఫరు->iface.init_tensor(బఫరు, tensor);
    }
    తిరిగి GGML_స్థితి_SUCCESS;
}

శూన్య ggml_backend_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    GGML_ASSERT(బఫరు);
    // clear is optional if the buffer is zero-sized
    ఒకవేళ (బఫరు->size == 0) {
        తిరిగి;
    }

    బఫరు->iface.clear(బఫరు, value);
}

size_t ggml_backend_బఫరు_get_alignment(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి ggml_backend_buft_get_alignment(ggml_backend_బఫరు_get_type(బఫరు));
}

size_t ggml_backend_బఫరు_get_max_కొలత(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి ggml_backend_buft_get_max_కొలత(ggml_backend_బఫరు_get_type(బఫరు));
}

size_t ggml_backend_బఫరు_get_alloc_కొలత(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_backend_buft_get_alloc_కొలత(ggml_backend_బఫరు_get_type(బఫరు), tensor);
}

తార్కిక ggml_backend_బఫరు_is_host(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి ggml_backend_buft_is_host(ggml_backend_బఫరు_get_type(బఫరు));
}

శూన్య ggml_backend_బఫరు_సమితి_usage(ggml_backend_బఫరు_t బఫరు, గణన ggml_backend_బఫరు_usage usage) {
    GGML_ASSERT(బఫరు);
    బఫరు->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    ఒకవేళ (ggml_backend_బఫరు_is_multi_బఫరు(బఫరు)) {
        ggml_backend_multi_బఫరు_సమితి_usage(బఫరు, usage);
    } లేకపోతే ఒకవేళ (ggml_backend_బఫరు_is_meta(బఫరు)) {
        ggml_backend_meta_బఫరు_సమితి_usage(బఫరు, usage);
    }
}

గణన ggml_backend_బఫరు_usage ggml_backend_బఫరు_get_usage(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);
    తిరిగి బఫరు->usage;
}

ggml_backend_బఫరు_type_t ggml_backend_బఫరు_get_type(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);
    తిరిగి బఫరు->buft;
}

శూన్య ggml_backend_బఫరు_reset(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);
    ఒకవేళ (బఫరు->iface.reset) {
        బఫరు->iface.reset(బఫరు);
    }
}

తార్కిక ggml_backend_బఫరు_copy_tensor(స్థిరరాశి నిర్మాణం ggml_tensor * src, నిర్మాణం ggml_tensor * dst) {
    ggml_backend_బఫరు_t dst_buf = dst->view_src ? dst->view_src->బఫరు : dst->బఫరు;
    ఒకవేళ (dst_buf->iface.cpy_tensor) {
        తిరిగి dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    తిరిగి అబద్ధం;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    ఒకవేళ (backend == ఖాళీ) {
        తిరిగి ఖాళీ;
    }
    తిరిగి backend->guid;
}

స్థిరరాశి అక్షరం * ggml_backend_name(ggml_backend_t backend) {
    ఒకవేళ (backend == ఖాళీ) {
        తిరిగి "NULL";
    }
    తిరిగి backend->iface.get_name(backend);
}

శూన్య ggml_backend_free(ggml_backend_t backend) {
    ఒకవేళ (backend == ఖాళీ) {
        తిరిగి;
    }

    backend->iface.విడుదల(backend);
}

ggml_backend_బఫరు_type_t ggml_backend_get_default_బఫరు_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    తిరిగి ggml_backend_dev_బఫరు_type(backend->device);
}

ggml_backend_బఫరు_t ggml_backend_alloc_బఫరు(ggml_backend_t backend, size_t size) {
    తిరిగి ggml_backend_buft_alloc_బఫరు(ggml_backend_get_default_బఫరు_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    తిరిగి ggml_backend_buft_get_alignment(ggml_backend_get_default_బఫరు_type(backend));
}

size_t ggml_backend_get_max_కొలత(ggml_backend_t backend) {
    తిరిగి ggml_backend_buft_get_max_కొలత(ggml_backend_get_default_బఫరు_type(backend));
}

శూన్య ggml_backend_tensor_సమితి_async(ggml_backend_t backend, నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");
    GGML_ASSERT(స్థానభ్రంశం + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    ఒకవేళ (backend->iface.సమితి_tensor_async == ఖాళీ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_సమితి(tensor, దత్తాంశం, స్థానభ్రంశం, size);
    } లేకపోతే {
        backend->iface.సమితి_tensor_async(backend, tensor, దత్తాంశం, స్థానభ్రంశం, size);
    }
}

శూన్య ggml_backend_tensor_get_async(ggml_backend_t backend, స్థిరరాశి నిర్మాణం ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");
    GGML_ASSERT(స్థానభ్రంశం + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    ఒకవేళ (backend->iface.get_tensor_async == ఖాళీ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, దత్తాంశం, స్థానభ్రంశం, size);
    } లేకపోతే {
        backend->iface.get_tensor_async(backend, tensor, దత్తాంశం, స్థానభ్రంశం, size);
    }
}

శూన్య ggml_backend_tensor_సమితి_2d_async(ggml_backend_t backend, నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");

    ఒకవేళ (n_copies <= 1 || backend->iface.సమితి_tensor_2d_async == ఖాళీ) {
        కొరకు (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_సమితి_async(backend, tensor, (స్థిరరాశి అక్షరం *) దత్తాంశం + i*stride_దత్తాంశం, స్థానభ్రంశం + i*stride_tensor, size);
        }
        తిరిగి;
    }
    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");
    GGML_ASSERT(స్థానభ్రంశం + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.సమితి_tensor_2d_async(backend, tensor, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం);
}

శూన్య ggml_backend_tensor_get_2d_async(ggml_backend_t backend, స్థిరరాశి నిర్మాణం ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");

    ఒకవేళ (n_copies <= 1 || backend->iface.get_tensor_2d_async == ఖాళీ) {
        కొరకు (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (అక్షరం *) దత్తాంశం + i*stride_దత్తాంశం, స్థానభ్రంశం + i*stride_tensor, size);
        }
        తిరిగి;
    }
    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");
    GGML_ASSERT(స్థానభ్రంశం + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం);
}

శూన్య ggml_backend_tensor_సమితి(నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_బఫరు_t buf = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;
    GGML_ASSERT(buf != ఖాళీ && "tensor buffer not set");

    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");
    GGML_ASSERT(స్థానభ్రంశం + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.సమితి_tensor(buf, tensor, దత్తాంశం, స్థానభ్రంశం, size);
}

శూన్య ggml_backend_tensor_get(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_బఫరు_t buf = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;
    GGML_ASSERT(buf != ఖాళీ && "tensor buffer not set");

    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");
    GGML_ASSERT(స్థానభ్రంశం + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, దత్తాంశం, స్థానభ్రంశం, size);
}

శూన్య ggml_backend_tensor_సమితి_2d(నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    GGML_ASSERT(tensor);
    ggml_backend_బఫరు_t buf = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;
    GGML_ASSERT(buf != ఖాళీ && "tensor buffer not set");

    ఒకవేళ (n_copies <= 1 || buf->iface.సమితి_tensor_2d == ఖాళీ) {
        కొరకు (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_సమితి(tensor, (స్థిరరాశి అక్షరం *) దత్తాంశం + i*stride_దత్తాంశం, స్థానభ్రంశం + i*stride_tensor, size);
        }
        తిరిగి;
    }
    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");
    GGML_ASSERT(స్థానభ్రంశం + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.సమితి_tensor_2d(buf, tensor, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం);
}

శూన్య ggml_backend_tensor_get_2d(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    GGML_ASSERT(tensor);
    ggml_backend_బఫరు_t buf = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;
    GGML_ASSERT(buf != ఖాళీ && "tensor buffer not set");

    ఒకవేళ (n_copies <= 1 || buf->iface.get_tensor_2d == ఖాళీ) {
        కొరకు (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (అక్షరం *) దత్తాంశం + i*stride_దత్తాంశం, స్థానభ్రంశం + i*stride_tensor, size);
        }
        తిరిగి;
    }
    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");
    GGML_ASSERT(స్థానభ్రంశం + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం);
}

శూన్య ggml_backend_tensor_memset(నిర్మాణం ggml_tensor * tensor, uint8_t value, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_బఫరు_t buf = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;

    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    GGML_ASSERT(buf != ఖాళీ && "tensor buffer not set");
    GGML_ASSERT(tensor->దత్తాంశం != ఖాళీ && "tensor not allocated");
    GGML_ASSERT(స్థానభ్రంశం + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != ఖాళీ && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, స్థానభ్రంశం, size);
}

శూన్య ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ఒకవేళ (backend->iface.synchronize == ఖాళీ) {
        తిరిగి;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_లేఖాచిత్రం_plan_t ggml_backend_లేఖాచిత్రం_plan_create(ggml_backend_t backend, నిర్మాణం ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.లేఖాచిత్రం_plan_create != ఖాళీ);

    తిరిగి backend->iface.లేఖాచిత్రం_plan_create(backend, cgraph);
}

శూన్య ggml_backend_లేఖాచిత్రం_plan_free(ggml_backend_t backend, ggml_backend_లేఖాచిత్రం_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.లేఖాచిత్రం_plan_free != ఖాళీ);

    backend->iface.లేఖాచిత్రం_plan_free(backend, plan);
}

గణన ggml_స్థితి ggml_backend_లేఖాచిత్రం_plan_compute(ggml_backend_t backend, ggml_backend_లేఖాచిత్రం_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.లేఖాచిత్రం_plan_compute != ఖాళీ);

    తిరిగి backend->iface.లేఖాచిత్రం_plan_compute(backend, plan);
}

గణన ggml_స్థితి ggml_backend_లేఖాచిత్రం_compute(ggml_backend_t backend, నిర్మాణం ggml_cgraph * cgraph) {
    గణన ggml_స్థితి err = ggml_backend_లేఖాచిత్రం_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    తిరిగి err;
}

గణన ggml_స్థితి ggml_backend_లేఖాచిత్రం_compute_async(ggml_backend_t backend, నిర్మాణం ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    తిరిగి backend->iface.లేఖాచిత్రం_compute(backend, cgraph);
}

తార్కిక ggml_backend_supports_op(ggml_backend_t backend, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    GGML_ASSERT(backend);
    తిరిగి ggml_backend_dev_supports_op(backend->device, op);
}

తార్కిక ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(backend);
    తిరిగి ggml_backend_dev_supports_buft(backend->device, buft);
}

తార్కిక ggml_backend_offload_op(ggml_backend_t backend, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    GGML_ASSERT(backend);
    తిరిగి ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    తిరిగి backend->device;
}

// backend copy

శూన్య ggml_backend_tensor_copy(స్థిరరాశి నిర్మాణం ggml_tensor * src, నిర్మాణం ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    ఒకవేళ (src == dst) {
        తిరిగి;
    }

    ఒకవేళ (ggml_backend_బఫరు_is_host(src->బఫరు)) {
        ggml_backend_tensor_సమితి(dst, src->దత్తాంశం, 0, ggml_nbytes(src));
    } లేకపోతే ఒకవేళ (ggml_backend_బఫరు_is_host(dst->బఫరు)) {
        ggml_backend_tensor_get(src, dst->దత్తాంశం, 0, ggml_nbytes(src));
    } లేకపోతే ఒకవేళ (!ggml_backend_బఫరు_copy_tensor(src, dst)) {
#నిర్వచితం_కాకపోతే NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_బఫరు_name(src->బఫరు), ggml_backend_బఫరు_name(dst->బఫరు));
#అయితే_ముగింపు // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        శూన్య * దత్తాంశం = జ్ఞాపకం_ఇవ్వు(nbytes);
        ggml_backend_tensor_get(src, దత్తాంశం, 0, nbytes);
        ggml_backend_tensor_సమితి(dst, దత్తాంశం, 0, nbytes);
        విడుదల(దత్తాంశం);
    }
}

శూన్య ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, స్థిరరాశి నిర్మాణం ggml_tensor * src, నిర్మాణం ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    ఒకవేళ (src == dst) {
        తిరిగి;
    }

    GGML_ASSERT(backend_dst);
    ఒకవేళ (backend_dst->iface.cpy_tensor_async != ఖాళీ) {
        ఒకవేళ (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            తిరిగి;
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
    ఒకవేళ (device == ఖాళీ || device->iface.event_new == ఖాళీ) {
        తిరిగి ఖాళీ;
    }
    తిరిగి device->iface.event_new(device);
}

శూన్య ggml_backend_event_free(ggml_backend_event_t event) {
    ఒకవేళ (event == ఖాళీ) {
        తిరిగి;
    }
    event->device->iface.event_free(event->device, event);
}

శూన్య ggml_backend_event_రికార్డు(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_రికార్డు != ఖాళీ);

    backend->iface.event_రికార్డు(backend, event);
}

శూన్య ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

శూన్య ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != ఖాళీ);

    backend->iface.event_wait(backend, event);
}

స్థిర శూన్య ggml_backend_లేఖాచిత్రం_optimize(ggml_backend_t backend, నిర్మాణం ggml_cgraph * cgraph, నిర్మాణం ggml_backend_లేఖాచిత్రం_optimize_params * params) {
    GGML_ASSERT(backend);
    ఒకవేళ (backend->iface.లేఖాచిత్రం_optimize != ఖాళీ) {
        backend->iface.లేఖాచిత్రం_optimize(backend, cgraph, params);
    }
}

// Backend device

స్థిరరాశి అక్షరం * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    తిరిగి device->iface.get_name(device);
}

స్థిరరాశి అక్షరం * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    తిరిగి device->iface.get_description(device);
}

శూన్య ggml_backend_dev_జ్ఞాపకం(ggml_backend_dev_t device, size_t * విడుదల, size_t * మొత్తం) {
    GGML_ASSERT(device);
    device->iface.get_జ్ఞాపకం(device, విడుదల, మొత్తం);
}

గణన ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    తిరిగి device->iface.get_type(device);
}

శూన్య ggml_backend_dev_get_props(ggml_backend_dev_t device, నిర్మాణం ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, పరిమాణం(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    తిరిగి device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, స్థిరరాశి అక్షరం * params) {
    GGML_ASSERT(device);
    తిరిగి device->iface.init_backend(device, params);
}

ggml_backend_బఫరు_type_t ggml_backend_dev_బఫరు_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    తిరిగి device->iface.get_బఫరు_type(device);
}

ggml_backend_బఫరు_type_t ggml_backend_dev_host_బఫరు_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ఒకవేళ (device->iface.get_host_బఫరు_type == ఖాళీ) {
        తిరిగి ఖాళీ;
    }

    తిరిగి device->iface.get_host_బఫరు_type(device);
}

ggml_backend_బఫరు_t ggml_backend_dev_బఫరు_from_host_ptr(ggml_backend_dev_t device, శూన్య * ptr, size_t size, size_t max_tensor_కొలత) {
    GGML_ASSERT(device);
    తిరిగి device->iface.బఫరు_from_host_ptr(device, ptr, size, max_tensor_కొలత);
}

తార్కిక ggml_backend_dev_supports_op(ggml_backend_dev_t device, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    GGML_ASSERT(device);
    తిరిగి device->iface.supports_op(device, op);
}

తార్కిక ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_బఫరు_type_t buft) {
    GGML_ASSERT(device);
    తిరిగి device->iface.supports_buft(device, buft);
}

తార్కిక ggml_backend_dev_offload_op(ggml_backend_dev_t device, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    GGML_ASSERT(device);
    ఒకవేళ (device->iface.offload_op != ఖాళీ) {
        తిరిగి device->iface.offload_op(device, op);
    }

    తిరిగి అబద్ధం;
}

// Backend (reg)

స్థిరరాశి అక్షరం * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    తిరిగి reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_లెక్క(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    తిరిగి reg->iface.get_device_లెక్క(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t సూచీ) {
    GGML_ASSERT(reg);
    తిరిగి reg->iface.get_device(reg, సూచీ);
}

శూన్య * ggml_backend_reg_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name) {
    GGML_ASSERT(reg);
    ఒకవేళ (!reg->iface.get_proc_చిరునామా) {
        తిరిగి ఖాళీ;
    }
    తిరిగి reg->iface.get_proc_చిరునామా(reg, name);
}

// multi-buffer buffer

నిర్మాణం ggml_backend_multi_బఫరు_context {
    ggml_backend_బఫరు_t * buffers;
    size_t n_buffers;
};

స్థిర శూన్య ggml_backend_multi_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);
    ggml_backend_multi_బఫరు_context * ctx = (ggml_backend_multi_బఫరు_context *) బఫరు->context;
    కొరకు (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_బఫరు_free(ctx->buffers[i]);
    }

    విడుదల(ctx->buffers);
    విడుదల(ctx);
}

స్థిర శూన్య ggml_backend_multi_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    GGML_ASSERT(బఫరు);
    ggml_backend_multi_బఫరు_context * ctx = (ggml_backend_multi_బఫరు_context *) బఫరు->context;
    కొరకు (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_బఫరు_clear(ctx->buffers[i], value);
    }
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_బఫరు_i ggml_backend_multi_బఫరు_i = {
    /* .free_buffer     = */ ggml_backend_multi_బఫరు_free_బఫరు,
    /* .get_base        = */ ఖాళీ,
    /* .init_tensor     = */ ఖాళీ,
    /* .memset_tensor   = */ ఖాళీ,
    /* .set_tensor      = */ ఖాళీ,
    /* .get_tensor      = */ ఖాళీ,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ఖాళీ,
    /* .clear           = */ ggml_backend_multi_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};

ggml_backend_బఫరు_t ggml_backend_multi_బఫరు_alloc_బఫరు(ggml_backend_బఫరు_t * buffers, size_t n_buffers) {
    ggml_backend_multi_బఫరు_context * ctx = (ggml_backend_multi_బఫరు_context *) జ్ఞాపకం_ఇవ్వు(పరిమాణం(నిర్మాణం ggml_backend_multi_బఫరు_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_బఫరు_t *) జ్ఞాపకం_ఇవ్వు(n_buffers * పరిమాణం(ggml_backend_బఫరు_t));

    GGML_ASSERT(ctx->buffers != ఖాళీ);

    size_t మొత్తం_కొలత = 0;
    కొరకు (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        మొత్తం_కొలత += ggml_backend_బఫరు_get_కొలత(buffers[i]);
    }

    తిరిగి ggml_backend_బఫరు_init(buffers[0]->buft, ggml_backend_multi_బఫరు_i, ctx, మొత్తం_కొలత);
}

తార్కిక ggml_backend_బఫరు_is_multi_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);
    తిరిగి బఫరు->iface.free_బఫరు == ggml_backend_multi_బఫరు_free_బఫరు;
}

శూన్య ggml_backend_multi_బఫరు_సమితి_usage(ggml_backend_బఫరు_t బఫరు, గణన ggml_backend_బఫరు_usage usage) {
    GGML_ASSERT(బఫరు);
    GGML_ASSERT(ggml_backend_బఫరు_is_multi_బఫరు(బఫరు));
    ggml_backend_multi_బఫరు_context * ctx = (ggml_backend_multi_బఫరు_context *) బఫరు->context;
    కొరకు (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_బఫరు_సమితి_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
స్థిర నిర్మాణం ggml_tensor * ggml_dup_tensor_layout(నిర్మాణం ggml_context * ctx, స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    నిర్మాణం ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    తిరిగి dup;
}

స్థిర తార్కిక ggml_is_view_op(గణన ggml_op op) {
    తిరిగి op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_క్రమమార్పు || op == GGML_OP_TRANSPOSE;
}

// scheduler

#నిర్వచితం_కాకపోతే GGML_SCHED_MAX_BACKENDS
#నిర్వచించు GGML_SCHED_MAX_BACKENDS 16
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే GGML_SCHED_MAX_విభజించు_INPUTS
#నిర్వచించు GGML_SCHED_MAX_విభజించు_INPUTS 30
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే GGML_SCHED_MAX_COPIES
#నిర్వచించు GGML_SCHED_MAX_COPIES 4
#అయితే_ముగింపు

నిర్మాణం ggml_backend_sched_విభజించు {
    పూర్ణాంకం backend_id;
    పూర్ణాంకం i_ప్రారంభం;
    పూర్ణాంకం i_చివర;
    నిర్మాణం ggml_tensor ** inputs;
    పూర్ణాంకం n_inputs;
    పూర్ణాంకం inputs_సామర్థ్యం;
    // graph view of this split
    నిర్మాణం ggml_cgraph graph;
};

నిర్మాణం ggml_backend_sched {
    తార్కిక is_reset; // true if the scheduler has been reset since the last graph split
    తార్కిక is_alloc;

    పూర్ణాంకం n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_బఫరు_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    నిర్మాణం ggml_hash_సమితి  hash_సమితి;
    పూర్ణాంకం                 * hv_tensor_backend_ids; // [hash_set.size]
    నిర్మాణం ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    పూర్ణాంకం * కణుపు_backend_ids; // [graph_size]
    పూర్ణాంకం * ఆకు_backend_ids; // [graph_size]

    పూర్ణాంకం * prev_కణుపు_backend_ids; // [graph_size]
    పూర్ణాంకం * prev_ఆకు_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    నిర్మాణం ggml_cgraph graph;

    // graph splits
    నిర్మాణం ggml_backend_sched_విభజించు * splits;
    పూర్ణాంకం n_splits;
    పూర్ణాంకం splits_సామర్థ్యం;

    // pipeline parallelism support
    పూర్ణాంకం n_copies;
    పూర్ణాంకం cur_copy;
    పూర్ణాంకం తదుపరి_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    నిర్మాణం ggml_tensor ** లేఖాచిత్రం_inputs;
    పూర్ణాంకం n_లేఖాచిత్రం_inputs;
    పూర్ణాంకం లేఖాచిత్రం_inputs_సామర్థ్యం;

    నిర్మాణం ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    శూన్య * callback_eval_user_దత్తాంశం;

    ggml_backend_sched_copy_callback callback_copy;
    శూన్య * callback_copy_user_దత్తాంశం;

    ggml_backend_sched_realloc_callback callback_realloc;
    శూన్య * callback_realloc_user_దత్తాంశం;

    అక్షరం * context_బఫరు;
    size_t context_బఫరు_కొలత;

    తార్కిక op_offload;

    పూర్ణాంకం debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    పూర్ణాంకం లేఖాచిత్రం_కొలత;
    పూర్ణాంకం prev_లేఖాచిత్రం_కొలత;

    పూర్ణాంకం debug_realloc;
};

#నిర్వచించు hash_id(tensor) ggml_hash_కనుగొను_or_చొప్పించు(&sched->hash_సమితి, tensor)
#నిర్వచించు tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#నిర్వచించు tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#నిర్వచించు tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

స్థిర శూన్య ggml_backend_sched_విభజించు_inputs_grow(నిర్మాణం ggml_backend_sched_విభజించు * విభజించు) {
    పూర్ణాంకం new_cap = GGML_SCHED_MAX_విభజించు_INPUTS;
    ఒకవేళ (విభజించు->inputs_సామర్థ్యం > 0) {
        new_cap = 2*విభజించు->inputs_సామర్థ్యం;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, విభజించు->inputs_సామర్థ్యం, new_cap);
    }
    స్వయంచాలక * pnew = (నిర్మాణం ggml_tensor **) realloc((శూన్య *) విభజించు->inputs, new_cap * పరిమాణం(నిర్మాణం ggml_tensor *));
    ఒకవేళ (pnew == ఖాళీ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * పరిమాణం(నిర్మాణం ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    విభజించు->inputs = pnew;
    విభజించు->inputs_సామర్థ్యం = new_cap;
}

స్థిర శూన్య ggml_backend_sched_లేఖాచిత్రం_inputs_grow(ggml_backend_sched_t sched) {
    పూర్ణాంకం new_cap = GGML_SCHED_MAX_విభజించు_INPUTS;
    ఒకవేళ (sched->లేఖాచిత్రం_inputs_సామర్థ్యం > 0) {
        new_cap = 2*sched->లేఖాచిత్రం_inputs_సామర్థ్యం;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->లేఖాచిత్రం_inputs_సామర్థ్యం, new_cap);
    }
    స్వయంచాలక * pnew = (నిర్మాణం ggml_tensor **) realloc((శూన్య *) sched->లేఖాచిత్రం_inputs, new_cap * పరిమాణం(నిర్మాణం ggml_tensor *));
    ఒకవేళ (pnew == ఖాళీ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * పరిమాణం(నిర్మాణం ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->లేఖాచిత్రం_inputs = pnew;
    sched->లేఖాచిత్రం_inputs_సామర్థ్యం = new_cap;
}

// returns the priority of the backend, lower id is higher priority
స్థిర పూర్ణాంకం ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    కొరకు (పూర్ణాంకం i = 0; i < sched->n_backends; i++) {
        ఒకవేళ (sched->backends[i] == backend) {
            తిరిగి i;
        }
    }
    తిరిగి -1;
}

స్థిర పూర్ణాంకం ggml_backend_sched_backend_from_బఫరు(ggml_backend_sched_t sched, స్థిరరాశి నిర్మాణం ggml_tensor * tensor, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    ggml_backend_బఫరు_t బఫరు = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;
    ఒకవేళ (బఫరు == ఖాళీ) {
        తిరిగి -1;
    }

    // find highest prio backend that supports the buffer type and the op
    కొరకు (పూర్ణాంకం i = 0; i < sched->n_backends; i++) {
        ఒకవేళ (ggml_backend_supports_buft(sched->backends[i], బఫరు->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            తిరిగి i;
        }
    }

#నిర్వచితం_కాకపోతే NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_బఫరు_name(బఫరు), tensor->name);
#అయితే_ముగింపు

    తిరిగి -1;
}

#ఒకవేళ 0
#నిర్వచించు GGML_SCHED_MAX_SPLITS_DEBUG 4096
స్థిర అక్షరం causes[GGML_DEFAULT_లేఖాచిత్రం_కొలత*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_విభజించు_INPUTS][128]; // debug only
#నిర్వచించు సమితి_CAUSE(కణుపు, ...) sprintf(causes[hash_id(కణుపు)], __VA_ARGS__)
#నిర్వచించు GET_CAUSE(కణుపు) causes[hash_id(కణుపు)]
#లేకపోతే
#నిర్వచించు సమితి_CAUSE(కణుపు, ...)
#నిర్వచించు GET_CAUSE(కణుపు) ""
#అయితే_ముగింపు

// returns the backend that should be used for the node based on the current locations
స్థిర పూర్ణాంకం ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, నిర్మాణం ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    పూర్ణాంకం cur_backend_id = ggml_backend_sched_backend_from_బఫరు(sched, tensor, tensor);
    ఒకవేళ (cur_backend_id != -1) {
        సమితి_CAUSE(tensor, "1.dst");
        తిరిగి cur_backend_id;
    }

    // view_src
    ఒకవేళ (tensor->view_src != ఖాళీ) {
        cur_backend_id = ggml_backend_sched_backend_from_బఫరు(sched, tensor->view_src, tensor);
        ఒకవేళ (cur_backend_id != -1) {
            సమితి_CAUSE(tensor, "1.vsrc");
            తిరిగి cur_backend_id;
        }
    }

    ఒకవేళ (tensor->బఫరు || (tensor->view_src && tensor->view_src->బఫరు)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_బఫరు_t బఫరు = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_బఫరు_name(బఫరు), ggml_op_name(tensor->op));
    }

    // graph input
    ఒకవేళ (tensor->flags & GGML_TENSOR_FLAG_ఆదానం) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        సమితి_CAUSE(tensor, "1.inp");
        తిరిగి cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    తార్కిక allow = నిజం;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    ఒకవేళ (allow) {
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            స్థిరరాశి నిర్మాణం ggml_tensor * src = tensor->src[i];
            ఒకవేళ (src == ఖాళీ) {
                కొనసాగు;
            }
            ఒకవేళ (src->బఫరు != ఖాళీ && src->బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_WEIGHTS) {
                పూర్ణాంకం src_backend_id = ggml_backend_sched_backend_from_బఫరు(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                ఒకవేళ (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_బఫరు_is_host(src->బఫరు)) {
                    కొరకు (పూర్ణాంకం b = 0; b < src_backend_id; b++) {
                        ఒకవేళ (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            సమితి_CAUSE(tensor, "1.off");
                            తిరిగి b;
                        }
                    }
                }
                సమితి_CAUSE(tensor, "1.wgt%d", i);
                తిరిగి src_backend_id;
            }
        }
    }

    తిరిగి -1;
}

స్థిర అక్షరం * fmt_కొలత(size_t size) {
    స్థిర అక్షరం బఫరు[128];
    ఒకవేళ (size >= 1024*1024) {
        snprintf(బఫరు, పరిమాణం(బఫరు), "%zuM", size/1024/1024);
    } లేకపోతే {
        snprintf(బఫరు, పరిమాణం(బఫరు), "%zuK", size/1024);
    }
    తిరిగి బఫరు;
}

స్థిర శూన్య ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, నిర్మాణం ggml_cgraph * graph) {
    పూర్ణాంకం cur_విభజించు = 0;
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; i++) {
        ఒకవేళ (cur_విభజించు < sched->n_splits && i == sched->splits[cur_విభజించు].i_ప్రారంభం) {
            ggml_backend_t విభజించు_backend = sched->backends[sched->splits[cur_విభజించు].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_విభజించు, ggml_backend_name(విభజించు_backend),
                sched->splits[cur_విభజించు].n_inputs);
            కొరకు (పూర్ణాంకం j = 0; j < sched->splits[cur_విభజించు].n_inputs; j++) {
                ఒకవేళ (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_విభజించు].inputs[j]->name,
                    fmt_కొలత(ggml_nbytes(sched->splits[cur_విభజించు].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_విభజించు++;
        }
        నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
        ఒకవేళ (ggml_is_view_op(కణుపు->op)) {
            కొనసాగు;
        }
        ఒకవేళ (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, కణుపు);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(కణుపు), కణుపు->name,
                fmt_కొలత(ggml_nbytes(కణుపు)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(కణుపు),
                graph->use_counts[ggml_hash_కనుగొను(&graph->సందర్శించిన_hash_సమితి, కణుపు)], కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
                నిర్మాణం ggml_tensor * src = కణుపు->src[j];
                ఒకవేళ (src == ఖాళీ) {
                    కొనసాగు;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_కొలత(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

స్థిర తార్కిక ggml_backend_sched_బఫరు_supported(ggml_backend_sched_t sched, నిర్మాణం ggml_tensor * t, పూర్ణాంకం backend_id) {
    ggml_backend_బఫరు_t buf = t->view_src ? t->view_src->బఫరు : t->బఫరు;
    ggml_backend_బఫరు_type_t buft = ఖాళీ;

    ఒకవేళ (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } లేకపోతే {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        పూర్ణాంకం tensor_backend_id = tensor_backend_id(t);
        ఒకవేళ (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        ఒకవేళ (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    తిరిగి buft != ఖాళీ && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

స్థిర శూన్య ggml_backend_sched_సమితి_if_supported(ggml_backend_sched_t sched, నిర్మాణం ggml_tensor * కణుపు, పూర్ణాంకం cur_backend_id, పూర్ణాంకం * కణుపు_backend_id) {
    ఒకవేళ (ggml_backend_supports_op(sched->backends[cur_backend_id], కణుపు)) {
        *కణుపు_backend_id = cur_backend_id;
        సమితి_CAUSE(కణుపు, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
శూన్య ggml_backend_sched_విభజించు_లేఖాచిత్రం(ggml_backend_sched_t sched, నిర్మాణం ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_లేఖాచిత్రం_inputs = 0;
    sched->is_reset = అబద్ధం;

    నిర్మాణం ggml_init_params params = {
        /* .mem_size =   */ sched->context_బఫరు_కొలత,
        /* .mem_buffer = */ sched->context_బఫరు,
        /* .no_alloc =   */ నిజం
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    ఒకవేళ (sched->ctx == ఖాళీ) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_లేఖాచిత్రం_తదుపరి_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_leafs; i++) {
        నిర్మాణం ggml_tensor * ఆకు = graph->leafs[i];
        పూర్ణాంకం * ఆకు_backend_id = &tensor_backend_id(ఆకు);
        // do not overwrite user assignments
        ఒకవేళ (*ఆకు_backend_id == -1) {
            *ఆకు_backend_id = ggml_backend_sched_backend_id_from_cur(sched, ఆకు);
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
        పూర్ణాంకం * కణుపు_backend_id = &tensor_backend_id(కణుపు);
        // do not overwrite user assignments
        ఒకవేళ (*కణుపు_backend_id == -1) {
            *కణుపు_backend_id = ggml_backend_sched_backend_id_from_cur(sched, కణుపు);

#ఒకవేళ 0
            // src
            ఒకవేళ (కణుపు->op == GGML_OP_NONE) {
                కొనసాగు;
            }

            కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
                నిర్మాణం ggml_tensor * src = కణుపు->src[j];
                ఒకవేళ (src == ఖాళీ) {
                    కొనసాగు;
                }
                పూర్ణాంకం * src_backend_id = &tensor_backend_id(src);
                ఒకవేళ (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#అయితే_ముగింపు
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        పూర్ణాంకం cur_backend_id = -1;
        కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; i++) {
            నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
            ఒకవేళ (ggml_is_view_op(కణుపు->op)) {
                కొనసాగు;
            }
            పూర్ణాంకం * కణుపు_backend_id = &tensor_backend_id(కణుపు);
            ఒకవేళ (*కణుపు_backend_id != -1) {
                ఒకవేళ (*కణుపు_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } లేకపోతే {
                    cur_backend_id = *కణుపు_backend_id;
                }
            } లేకపోతే ఒకవేళ (cur_backend_id != -1) {
                ggml_backend_sched_సమితి_if_supported(sched, కణుపు, cur_backend_id, కణుపు_backend_id);
            }
        }
    }
    // expand gpu up
    {
        పూర్ణాంకం cur_backend_id = -1;
        కొరకు (పూర్ణాంకం i = graph->n_nodes - 1; i >= 0; i--) {
            నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
            ఒకవేళ (ggml_is_view_op(కణుపు->op)) {
                కొనసాగు;
            }
            పూర్ణాంకం * కణుపు_backend_id = &tensor_backend_id(కణుపు);
            ఒకవేళ (*కణుపు_backend_id != -1) {
                ఒకవేళ (*కణుపు_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } లేకపోతే {
                    cur_backend_id = *కణుపు_backend_id;
                }
            } లేకపోతే ఒకవేళ (cur_backend_id != -1) {
                ggml_backend_sched_సమితి_if_supported(sched, కణుపు, cur_backend_id, కణుపు_backend_id);
            }
        }
    }
    // expand rest down
    {
        పూర్ణాంకం cur_backend_id = -1;
        కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; i++) {
            నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
            ఒకవేళ (ggml_is_view_op(కణుపు->op)) {
                కొనసాగు;
            }
            పూర్ణాంకం * కణుపు_backend_id = &tensor_backend_id(కణుపు);
            ఒకవేళ (*కణుపు_backend_id != -1) {
                cur_backend_id = *కణుపు_backend_id;
            } లేకపోతే ఒకవేళ (cur_backend_id != -1) {
                ggml_backend_sched_సమితి_if_supported(sched, కణుపు, cur_backend_id, కణుపు_backend_id);
            }
        }
    }
    // expand rest up
    {
        పూర్ణాంకం cur_backend_id = -1;
        కొరకు (పూర్ణాంకం i = graph->n_nodes - 1; i >= 0; i--) {
            నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
            ఒకవేళ (ggml_is_view_op(కణుపు->op)) {
                కొనసాగు;
            }
            పూర్ణాంకం * కణుపు_backend_id = &tensor_backend_id(కణుపు);
            ఒకవేళ (*కణుపు_backend_id != -1) {
                cur_backend_id = *కణుపు_backend_id;
            } లేకపోతే ఒకవేళ (cur_backend_id != -1) {
                ggml_backend_sched_సమితి_if_supported(sched, కణుపు, cur_backend_id, కణుపు_backend_id);
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
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
        ఒకవేళ (ggml_is_view_op(కణుపు->op)) {
            కొనసాగు;
        }
        పూర్ణాంకం * కణుపు_backend_id = &tensor_backend_id(కణుపు);
        ఒకవేళ (*కణుపు_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            పూర్ణాంకం n_supported_ఉత్తమ = -1;
            కొరకు (పూర్ణాంకం b = 0; b < sched->n_backends; b++) {
                ఒకవేళ (ggml_backend_supports_op(sched->backends[b], కణుపు)) {
                    పూర్ణాంకం n_supported = 0;
                    కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
                        నిర్మాణం ggml_tensor * src = కణుపు->src[j];
                        ఒకవేళ (src == ఖాళీ) {
                            కొనసాగు;
                        }
                        ఒకవేళ ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_బఫరు_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    ఒకవేళ (n_supported > n_supported_ఉత్తమ) {
                        n_supported_ఉత్తమ = n_supported;
                        *కణుపు_backend_id = b;
                        సమితి_CAUSE(కణుపు, "3.best");
                    }
                }
            }
        } లేకపోతే {
            // assigned node: upgrade to higher prio backend if possible
            కొరకు (పూర్ణాంకం b = 0; b < *కణుపు_backend_id; b++) {
                ఒకవేళ (sched->bufts[b] == sched->bufts[*కణుపు_backend_id] && ggml_backend_supports_op(sched->backends[b], కణుపు)) {
                    తార్కిక supported = నిజం;
                    కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
                        నిర్మాణం ggml_tensor * src = కణుపు->src[j];
                        ఒకవేళ (src == ఖాళీ) {
                            కొనసాగు;
                        }
                        ఒకవేళ (!ggml_backend_sched_బఫరు_supported(sched, src, b)) {
                            supported = అబద్ధం;
                            ఆపు;
                        }
                    }
                    ఒకవేళ (supported) {
                        *కణుపు_backend_id = b;
                        సమితి_CAUSE(కణుపు, "3.upg");
                        ఆపు;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
        పూర్ణాంకం * cur_backend_id = &tensor_backend_id(కణుపు);
        ఒకవేళ (కణుపు->view_src != ఖాళీ && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(కణుపు->view_src);
            సమితి_CAUSE(కణుపు, "4.vsrc");
        }
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            నిర్మాణం ggml_tensor * src = కణుపు->src[j];
            ఒకవేళ (src == ఖాళీ) {
                కొనసాగు;
            }
            పూర్ణాంకం * src_backend_id = &tensor_backend_id(src);
            ఒకవేళ (*src_backend_id == -1) {
                ఒకవేళ (src->view_src != ఖాళీ) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    సమితి_CAUSE(src, "4.vsrc");
                } లేకపోతే {
                    *src_backend_id = *cur_backend_id;
                    సమితి_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        కొరకు (పూర్ణాంకం b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_సమితి_if_supported(sched, కణుపు, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        పూర్ణాంకం i_విభజించు = 0;
        నిర్మాణం ggml_backend_sched_విభజించు * విభజించు = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        పూర్ణాంకం i = 0;
        కొరకు (; i < graph->n_nodes; i++) {
            నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
            ఒకవేళ (!ggml_is_view_op(కణుపు->op)) {
                విభజించు->backend_id = tensor_backend_id(కణుపు);
                ఆపు;
            }
        }
        విభజించు->i_ప్రారంభం = 0;
        విభజించు->n_inputs = 0;
        పూర్ణాంకం cur_backend_id = విభజించు->backend_id;
        కొరకు (; i < graph->n_nodes; i++) {
            నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];

            ఒకవేళ (ggml_is_view_op(కణుపు->op)) {
                కొనసాగు;
            }

            స్థిరరాశి పూర్ణాంకం కణుపు_backend_id = tensor_backend_id(కణుపు);

            GGML_ASSERT(కణుపు_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            తార్కిక need_new_విభజించు = అబద్ధం;
            ఒకవేళ (కణుపు_backend_id == cur_backend_id && విభజించు->n_inputs > 0) {
                కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
                    నిర్మాణం ggml_tensor * src = కణుపు->src[j];
                    ఒకవేళ (src == ఖాళీ) {
                        కొనసాగు;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    ఒకవేళ (src->బఫరు != ఖాళీ && src->బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_WEIGHTS) {
                        పూర్ణాంకం src_backend_id = tensor_backend_id(src);
                        ఒకవేళ (src_backend_id != cur_backend_id && !ggml_backend_sched_బఫరు_supported(sched, src, cur_backend_id)) {
                            need_new_విభజించు = నిజం;
                            ఆపు;
                        }
                    }
                }
            }

            ఒకవేళ (కణుపు_backend_id != cur_backend_id || need_new_విభజించు) {
                విభజించు->i_చివర = i;
                i_విభజించు++;
                ఒకవేళ (i_విభజించు >= sched->splits_సామర్థ్యం) {
                    పూర్ణాంకం old_cap = sched->splits_సామర్థ్యం;
                    sched->splits_సామర్థ్యం *= 2;
                    sched->splits = (ggml_backend_sched_విభజించు *)
                        realloc(sched->splits, sched->splits_సామర్థ్యం * పరిమాణం(నిర్మాణం ggml_backend_sched_విభజించు));
                    GGML_ASSERT(sched->splits != ఖాళీ);
                    కొరకు (పూర్ణాంకం k = old_cap; k < sched->splits_సామర్థ్యం; k++) {
                        memset(&sched->splits[k], 0, పరిమాణం(నిర్మాణం ggml_backend_sched_విభజించు));
                    }
                }
                విభజించు = &sched->splits[i_విభజించు];
                విభజించు->backend_id = కణుపు_backend_id;
                విభజించు->i_ప్రారంభం = i;
                విభజించు->n_inputs = 0;
                cur_backend_id = కణుపు_backend_id;
            }

            // find inputs that are not on the same backend
            కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
                నిర్మాణం ggml_tensor * src = కణుపు->src[j];
                ఒకవేళ (src == ఖాళీ) {
                    కొనసాగు;
                }

                size_t src_id = hash_id(src);
                స్థిరరాశి పూర్ణాంకం src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                ఒకవేళ (src_backend_id != cur_backend_id && !ggml_backend_sched_బఫరు_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    ఒకవేళ (tensor_id_copy(src_id, cur_backend_id, 0) == ఖాళీ) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        కొరకు (పూర్ణాంకం c = 0; c < sched->n_copies; c++) {
                            నిర్మాణం ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            ఒకవేళ (sched->n_copies > 1) {
                                ggml_సమితి_ఆదానం(tensor_copy);
                                ggml_సమితి_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            సమితి_CAUSE(tensor_copy, "4.cpy");
                        }
                        పూర్ణాంకం n_inputs = విభజించు->n_inputs++;
                        ఒకవేళ (n_inputs >= విభజించు->inputs_సామర్థ్యం) {
                            ggml_backend_sched_విభజించు_inputs_grow(విభజించు);
                        }
                        విభజించు->inputs[n_inputs] = src;
                    }
                    కణుపు->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        విభజించు->i_చివర = graph->n_nodes;
        sched->n_splits = i_విభజించు + 1;
    }

    ఒకవేళ (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    ఒకవేళ (sched->n_copies > 1) {
        కొరకు (పూర్ణాంకం i = 0; i < graph->n_leafs; i++) {
            నిర్మాణం ggml_tensor * ఆకు = graph->leafs[i];
            ఒకవేళ ((ఆకు->flags & GGML_TENSOR_FLAG_ఆదానం) == 0) {
                కొనసాగు;
            }

            స్థిరరాశి size_t ఆకు_id = hash_id(ఆకు);
            స్థిరరాశి పూర్ణాంకం ఆకు_backend_id = tensor_backend_id(ఆకు);
            GGML_ASSERT(ఆకు_backend_id != -1); // all leafs should be assigned by now

            ఒకవేళ (tensor_id_copy(ఆకు_id, ఆకు_backend_id, 0) == ఖాళీ) {
                ggml_backend_t backend = sched->backends[ఆకు_backend_id];
                కొరకు (పూర్ణాంకం c = 0; c < sched->n_copies; c++) {
                    నిర్మాణం ggml_tensor * tensor_copy;
                    ఒకవేళ (c == sched->cur_copy) {
                        tensor_copy = ఆకు; // use the original tensor as the current copy
                    } లేకపోతే {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, ఆకు);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), ఆకు->name, c);
                    }
                    ggml_సమితి_ఆదానం(tensor_copy);
                    ggml_సమితి_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(ఆకు_id, ఆకు_backend_id, c) = tensor_copy;
                    సమితి_CAUSE(tensor_copy, "6.cpy");
                }
            }

            పూర్ణాంకం n_లేఖాచిత్రం_inputs = sched->n_లేఖాచిత్రం_inputs++;
            ఒకవేళ (n_లేఖాచిత్రం_inputs >= sched->లేఖాచిత్రం_inputs_సామర్థ్యం) {
                ggml_backend_sched_లేఖాచిత్రం_inputs_grow(sched);
            }
            sched->లేఖాచిత్రం_inputs[n_లేఖాచిత్రం_inputs] = ఆకు;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        పూర్ణాంకం * tmp = sched->కణుపు_backend_ids;
        sched->కణుపు_backend_ids = sched->prev_కణుపు_backend_ids;
        sched->prev_కణుపు_backend_ids = tmp;

        tmp = sched->ఆకు_backend_ids;
        sched->ఆకు_backend_ids = sched->prev_ఆకు_backend_ids;
        sched->prev_ఆకు_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    నిర్మాణం ggml_backend_లేఖాచిత్రం_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](శూన్య * user_దత్తాంశం, ggml_tensor * tensor, ggml_tensor * until) {
            స్వయంచాలక & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_దత్తాంశం;
            std::vector<ggml_tensor *> & keep = deps[until];
            ఒకవేళ (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    కొరకు (పూర్ణాంకం i = 0; i < sched->n_splits; i++) {
        నిర్మాణం ggml_backend_sched_విభజించు * విభజించు = &sched->splits[i];
        విభజించు->graph = ggml_లేఖాచిత్రం_view(graph, విభజించు->i_ప్రారంభం, విభజించు->i_చివర);

        ggml_backend_లేఖాచిత్రం_optimize(sched->backends[విభజించు->backend_id], &విభజించు->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    పూర్ణాంకం n_dep_nodes = 0;
    కొరకు (స్థిరరాశి స్వయంచాలక & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    పూర్ణాంకం మొత్తం_inputs = sched->n_లేఖాచిత్రం_inputs;
    కొరకు (పూర్ణాంకం i = 0; i < sched->n_splits; i++) {
        మొత్తం_inputs += sched->splits[i].n_inputs;
    }
    పూర్ణాంకం లేఖాచిత్రం_కొలత = std::max(graph->n_nodes, graph->n_leafs) + మొత్తం_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_లేఖాచిత్రం_కొలత = sched->లేఖాచిత్రం_కొలత;
    sched->లేఖాచిత్రం_కొలత = లేఖాచిత్రం_కొలత;

    ఒకవేళ (sched->graph.size < లేఖాచిత్రం_కొలత) {
        sched->graph.size = లేఖాచిత్రం_కొలత;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, లేఖాచిత్రం_కొలత * పరిమాణం(నిర్మాణం ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, లేఖాచిత్రం_కొలత * పరిమాణం(నిర్మాణం ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != ఖాళీ);
        GGML_ASSERT(sched->graph.leafs != ఖాళీ);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    నిర్మాణం ggml_cgraph * లేఖాచిత్రం_copy = &sched->graph;

    పూర్ణాంకం n_dep_nodes_added = 0;

    కొరకు (పూర్ణాంకం i = 0; i < sched->n_splits; i++) {
        నిర్మాణం ggml_backend_sched_విభజించు * విభజించు = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        కొరకు (పూర్ణాంకం j = 0; j < విభజించు->n_inputs; j++) {
            ప్రతిజ్ఞా(లేఖాచిత్రం_copy->size > (లేఖాచిత్రం_copy->n_nodes + 1));

            నిర్మాణం ggml_tensor * ఆదానం = విభజించు->inputs[j];
            స్థిరరాశి size_t ఆదానం_id = hash_id(ఆదానం);
            నిర్మాణం ggml_tensor * ఆదానం_cpy = tensor_id_copy(ఆదానం_id, విభజించు->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            నిర్మాణం ggml_tensor * ఆదానం_dep = ggml_view_tensor(sched->ctx, ఆదానం);
            ఆదానం_dep->src[0] = ఆదానం;
            sched->కణుపు_backend_ids[లేఖాచిత్రం_copy->n_nodes] = sched->hv_tensor_backend_ids[ఆదానం_id];
            లేఖాచిత్రం_copy->nodes[లేఖాచిత్రం_copy->n_nodes++] = ఆదానం_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->కణుపు_backend_ids[లేఖాచిత్రం_copy->n_nodes] = విభజించు->backend_id;
            లేఖాచిత్రం_copy->nodes[లేఖాచిత్రం_copy->n_nodes++] = ఆదానం_cpy;
        }

        కొరకు (పూర్ణాంకం j = విభజించు->i_ప్రారంభం; j < విభజించు->i_చివర; j++) {
            ప్రతిజ్ఞా(లేఖాచిత్రం_copy->size > లేఖాచిత్రం_copy->n_nodes);
            sched->కణుపు_backend_ids[లేఖాచిత్రం_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            లేఖాచిత్రం_copy->nodes[లేఖాచిత్రం_copy->n_nodes++] = graph->nodes[j];

            ఒకవేళ (alloc_deps.ఖాళీలు()) {
                కొనసాగు;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            స్వయంచాలక it = alloc_deps.find(graph->nodes[j]);
            ఒకవేళ (it != alloc_deps.end()) {
                స్థిరరాశి std::vector<ggml_tensor *> & keep = it->second;
                కొరకు (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    నిర్మాణం ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    కొరకు (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    ప్రతిజ్ఞా(లేఖాచిత్రం_copy->size > లేఖాచిత్రం_copy->n_nodes);
                    sched->కణుపు_backend_ids[లేఖాచిత్రం_copy->n_nodes] = విభజించు->backend_id;
                    లేఖాచిత్రం_copy->nodes[లేఖాచిత్రం_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    ఒకవేళ (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        కొరకు (పూర్ణాంకం i = 0; i < sched->n_లేఖాచిత్రం_inputs; i++) {
            నిర్మాణం ggml_tensor * ఆదానం = sched->లేఖాచిత్రం_inputs[i];
            size_t id = hash_id(ఆదానం);
            పూర్ణాంకం backend_id = tensor_backend_id(ఆదానం);
            కొరకు (పూర్ణాంకం c = 0; c < sched->n_copies; c++) {
                నిర్మాణం ggml_tensor * ఆదానం_cpy = tensor_id_copy(id, backend_id, c);
                sched->ఆకు_backend_ids[లేఖాచిత్రం_copy->n_leafs] = backend_id;
                ప్రతిజ్ఞా(లేఖాచిత్రం_copy->size > లేఖాచిత్రం_copy->n_leafs);
                లేఖాచిత్రం_copy->leafs[లేఖాచిత్రం_copy->n_leafs++] = ఆదానం_cpy;
            }
        }

        కొరకు (పూర్ణాంకం i = 0; i < sched->n_splits; i++) {
            నిర్మాణం ggml_backend_sched_విభజించు * విభజించు = &sched->splits[i];
            పూర్ణాంకం backend_id = విభజించు->backend_id;
            కొరకు (పూర్ణాంకం j = 0; j < విభజించు->n_inputs; j++) {
                నిర్మాణం ggml_tensor * ఆదానం = విభజించు->inputs[j];
                size_t id = hash_id(ఆదానం);
                కొరకు (పూర్ణాంకం c = 0; c < sched->n_copies; c++) {
                    నిర్మాణం ggml_tensor * ఆదానం_cpy = tensor_id_copy(id, backend_id, c);
                    sched->ఆకు_backend_ids[లేఖాచిత్రం_copy->n_leafs] = backend_id;
                    ప్రతిజ్ఞా(లేఖాచిత్రం_copy->size > లేఖాచిత్రం_copy->n_leafs);
                    లేఖాచిత్రం_copy->leafs[లేఖాచిత్రం_copy->n_leafs++] = ఆదానం_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_leafs; i++) {
        నిర్మాణం ggml_tensor * ఆకు = graph->leafs[i];
        sched->ఆకు_backend_ids[లేఖాచిత్రం_copy->n_leafs] = tensor_backend_id(ఆకు);
        ప్రతిజ్ఞా(లేఖాచిత్రం_copy->size > లేఖాచిత్రం_copy->n_leafs);
        లేఖాచిత్రం_copy->leafs[లేఖాచిత్రం_copy->n_leafs++] = ఆకు;
    }

    // set ids for all splits
    కొరకు (పూర్ణాంకం i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_లేఖాచిత్రం_తదుపరి_uid();
    }
}

స్థిర తార్కిక ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    తార్కిక backend_ids_మారింది = అబద్ధం;
    కొరకు (పూర్ణాంకం i = 0; i < sched->graph.n_nodes; i++) {
        ఒకవేళ (sched->కణుపు_backend_ids[i] != sched->prev_కణుపు_backend_ids[i] &&
            sched->bufts[sched->కణుపు_backend_ids[i]] != sched->bufts[sched->prev_కణుపు_backend_ids[i]]) {
            backend_ids_మారింది = నిజం;
            ఆపు;
        }
    }
    ఒకవేళ (!backend_ids_మారింది) {
        కొరకు (పూర్ణాంకం i = 0; i < sched->graph.n_leafs; i++) {
            ఒకవేళ (sched->ఆకు_backend_ids[i] != sched->prev_ఆకు_backend_ids[i] &&
                sched->bufts[sched->ఆకు_backend_ids[i]] != sched->bufts[sched->prev_ఆకు_backend_ids[i]]) {
                backend_ids_మారింది = నిజం;
                ఆపు;
            }
        }
    }

    // allocate graph
    ఒకవేళ (backend_ids_మారింది || !ggml_gallocr_alloc_లేఖాచిత్రం(sched->galloc, &sched->graph)) {
#నిర్వచితం_కాకపోతే NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_మారింది);
#అయితే_ముగింపు

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        స్థిరరాశి తార్కిక లేఖాచిత్రం_కొలత_మారింది = sched->prev_లేఖాచిత్రం_కొలత != sched->లేఖాచిత్రం_కొలత;

        ఒకవేళ (sched->callback_realloc != ఖాళీ) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_మారింది, లేఖాచిత్రం_కొలత_మారింది, sched->callback_realloc_user_దత్తాంశం);
        }

        ఒకవేళ (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            స్థిరరాశి తార్కిక unexpected = !backend_ids_మారింది && !లేఖాచిత్రం_కొలత_మారింది;

            ఒకవేళ (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->లేఖాచిత్రం_కొలత, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        కొరకు (పూర్ణాంకం i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ఒకవేళ (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->కణుపు_backend_ids, sched->ఆకు_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (!ggml_gallocr_alloc_లేఖాచిత్రం(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_backend_sched_is_host_బరువు(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    తిరిగి t->బఫరు != ఖాళీ &&
        ggml_backend_బఫరు_get_usage(t->బఫరు) == GGML_BACKEND_బఫరు_USAGE_WEIGHTS &&
        ggml_backend_బఫరు_is_host(t->బఫరు);
}

స్థిర శూన్య ggml_backend_sched_copy_ఆదానం(ggml_backend_sched_t sched, నిర్మాణం ggml_backend_sched_విభజించు * విభజించు, నిర్మాణం ggml_tensor * ఆదానం) {
    స్థిరరాశి పూర్ణాంకం విభజించు_backend_id = విభజించు->backend_id;
    ggml_backend_t విభజించు_backend = sched->backends[విభజించు_backend_id];
    ggml_backend_t ఆదానం_backend = ggml_backend_sched_get_tensor_backend(sched, ఆదానం);
    నిర్మాణం ggml_tensor * ఆదానం_cpy = tensor_copy(ఆదానం, విభజించు_backend_id, sched->cur_copy);

    ఒకవేళ (ఆదానం->flags & GGML_TENSOR_FLAG_ఆదానం) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        ఒకవేళ (sched->events[విభజించు_backend_id][sched->cur_copy] != ఖాళీ) {
            ggml_backend_event_synchronize(sched->events[విభజించు_backend_id][sched->cur_copy]);
        } లేకపోతే {
            ggml_backend_synchronize(విభజించు_backend);
        }
        ggml_backend_tensor_copy(ఆదానం, ఆదానం_cpy);
        తిరిగి;
    }

    // wait for the split backend to finish using the input before overwriting it
    ఒకవేళ (sched->events[విభజించు_backend_id][sched->cur_copy] != ఖాళీ) {
        ggml_backend_event_wait(విభజించు_backend, sched->events[విభజించు_backend_id][sched->cur_copy]);
    } లేకపోతే {
        ggml_backend_synchronize(విభజించు_backend);
    }

    ఒకవేళ (sched->callback_copy != ఖాళీ && ggml_backend_sched_is_host_బరువు(ఆదానం) &&
        sched->callback_copy(విభజించు_backend, ఆదానం, ఆదానం_cpy, &విభజించు->graph, sched->callback_copy_user_దత్తాంశం)) {
        తిరిగి;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    ఒకవేళ (!విభజించు_backend->iface.cpy_tensor_async || !విభజించు_backend->iface.cpy_tensor_async(ఆదానం_backend, విభజించు_backend, ఆదానం, ఆదానం_cpy)) {
        ggml_backend_synchronize(ఆదానం_backend);
        ఒకవేళ (sched->events[విభజించు_backend_id][sched->cur_copy] != ఖాళీ) {
            ggml_backend_event_synchronize(sched->events[విభజించు_backend_id][sched->cur_copy]);
        } లేకపోతే {
            ggml_backend_synchronize(విభజించు_backend);
        }
        ggml_backend_tensor_copy(ఆదానం, ఆదానం_cpy);
    }
}

స్థిర గణన ggml_స్థితి ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    నిర్మాణం ggml_backend_sched_విభజించు * splits = sched->splits;

    పూర్ణాంకం prev_backend_id = -1;

    కొరకు (పూర్ణాంకం విభజించు_id = 0; విభజించు_id < sched->n_splits; విభజించు_id++) {
        నిర్మాణం ggml_backend_sched_విభజించు * విభజించు = &splits[విభజించు_id];
        పూర్ణాంకం విభజించు_backend_id = విభజించు->backend_id;
        ggml_backend_t విభజించు_backend = sched->backends[విభజించు_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        ఒకవేళ (విభజించు->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != విభజించు_backend_id) {
            ఒకవేళ (sched->events[prev_backend_id][sched->cur_copy] != ఖాళీ) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } లేకపోతే {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        కొరకు (పూర్ణాంకం ఆదానం_id = 0; ఆదానం_id < విభజించు->n_inputs; ఆదానం_id++) {
            ఒకవేళ (!ggml_backend_sched_is_host_బరువు(విభజించు->inputs[ఆదానం_id])) {
                ggml_backend_sched_copy_ఆదానం(sched, విభజించు, విభజించు->inputs[ఆదానం_id]);
            }
        }
        కొరకు (పూర్ణాంకం ఆదానం_id = 0; ఆదానం_id < విభజించు->n_inputs; ఆదానం_id++) {
            ఒకవేళ (ggml_backend_sched_is_host_బరువు(విభజించు->inputs[ఆదానం_id])) {
                ggml_backend_sched_copy_ఆదానం(sched, విభజించు, విభజించు->inputs[ఆదానం_id]);
            }
        }

        ఒకవేళ (!sched->callback_eval) {
            గణన ggml_స్థితి ec = ggml_backend_లేఖాచిత్రం_compute_async(విభజించు_backend, &విభజించు->graph);
            ఒకవేళ (ec != GGML_స్థితి_SUCCESS) {
                తిరిగి ec;
            }
        } లేకపోతే {
            // similar to ggml_backend_compare_graph_backend
            కొరకు (పూర్ణాంకం j0 = 0; j0 < విభజించు->graph.n_nodes; j0++) {
                నిర్మాణం ggml_tensor * t = విభజించు->graph.nodes[j0];

                // check if the user needs data from this node
                తార్కిక need = sched->callback_eval(t, నిజం, sched->callback_eval_user_దత్తాంశం);

                పూర్ణాంకం j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                అయితే (!need && j1 < విభజించు->graph.n_nodes - 1) {
                    t = విభజించు->graph.nodes[++j1];
                    need = sched->callback_eval(t, నిజం, sched->callback_eval_user_దత్తాంశం);
                }

                నిర్మాణం ggml_cgraph gv = ggml_లేఖాచిత్రం_view(&విభజించు->graph, j0, j1 + 1);

                గణన ggml_స్థితి ec = ggml_backend_లేఖాచిత్రం_compute_async(విభజించు_backend, &gv);
                ఒకవేళ (ec != GGML_స్థితి_SUCCESS) {
                    తిరిగి ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(విభజించు_backend);

                ఒకవేళ (need && !sched->callback_eval(t, అబద్ధం, sched->callback_eval_user_దత్తాంశం)) {
                    ఆపు;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        ఒకవేళ (sched->events[విభజించు_backend_id][sched->cur_copy] != ఖాళీ) {
            ggml_backend_event_రికార్డు(sched->events[విభజించు_backend_id][sched->cur_copy], విభజించు_backend);
        }

        prev_backend_id = విభజించు_backend_id;
    }

    తిరిగి GGML_స్థితి_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_బఫరు_type_t * bufts,
        పూర్ణాంకం n_backends,
        size_t లేఖాచిత్రం_కొలత,
        తార్కిక parallel,
        తార్కిక op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    నిర్మాణం ggml_backend_sched * sched = (ggml_backend_sched *) సున్నా_జ్ఞాపకం_ఇవ్వు(1, పరిమాణం(నిర్మాణం ggml_backend_sched));

    స్థిరరాశి అక్షరం * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#నిర్వచితమైతే GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#అయితే_ముగింపు
    స్థిరరాశి అక్షరం * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_సమితి    = ggml_hash_సమితి_new(లేఖాచిత్రం_కొలత);
    sched->hv_tensor_backend_ids = (పూర్ణాంకం *) జ్ఞాపకం_ఇవ్వు(sched->hash_సమితి.size * పరిమాణం(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) జ్ఞాపకం_ఇవ్వు(sched->hash_సమితి.size * sched->n_backends * sched->n_copies * పరిమాణం(నిర్మాణం ggml_tensor *));

    స్థిరరాశి size_t ggml_sched_max_splits = లేఖాచిత్రం_కొలత; // at most there is one split for each node in the graph
    స్థిరరాశి size_t nodes_కొలత = లేఖాచిత్రం_కొలత + ggml_sched_max_splits*GGML_SCHED_MAX_విభజించు_INPUTS*2;
    sched->కణుపు_backend_ids = (పూర్ణాంకం *) సున్నా_జ్ఞాపకం_ఇవ్వు(nodes_కొలత, పరిమాణం(sched->కణుపు_backend_ids[0]));
    sched->ఆకు_backend_ids = (పూర్ణాంకం *) సున్నా_జ్ఞాపకం_ఇవ్వు(nodes_కొలత, పరిమాణం(sched->ఆకు_backend_ids[0]));
    sched->prev_కణుపు_backend_ids = (పూర్ణాంకం *) సున్నా_జ్ఞాపకం_ఇవ్వు(nodes_కొలత, పరిమాణం(sched->prev_కణుపు_backend_ids[0]));
    sched->prev_ఆకు_backend_ids = (పూర్ణాంకం *) సున్నా_జ్ఞాపకం_ఇవ్వు(nodes_కొలత, పరిమాణం(sched->prev_ఆకు_backend_ids[0]));

    sched->లేఖాచిత్రం_కొలత      = 0;
    sched->prev_లేఖాచిత్రం_కొలత = 0;

    sched->context_బఫరు_కొలత = ggml_sched_max_splits*GGML_SCHED_MAX_విభజించు_INPUTS*2*పరిమాణం(నిర్మాణం ggml_tensor) + ggml_లేఖాచిత్రం_overhead_custom(లేఖాచిత్రం_కొలత, అబద్ధం);
    sched->context_బఫరు = (అక్షరం *) జ్ఞాపకం_ఇవ్వు(sched->context_బఫరు_కొలత);

    స్థిరరాశి పూర్ణాంకం initial_splits_సామర్థ్యం = 16;
    sched->splits = (ggml_backend_sched_విభజించు *) సున్నా_జ్ఞాపకం_ఇవ్వు(initial_splits_సామర్థ్యం, పరిమాణం(sched->splits[0]));
    sched->splits_సామర్థ్యం = initial_splits_సామర్థ్యం;

    sched->లేఖాచిత్రం_inputs_సామర్థ్యం = GGML_SCHED_MAX_విభజించు_INPUTS;
    sched->లేఖాచిత్రం_inputs = (నిర్మాణం ggml_tensor **) సున్నా_జ్ఞాపకం_ఇవ్వు(sched->లేఖాచిత్రం_inputs_సామర్థ్యం, పరిమాణం(నిర్మాణం ggml_tensor *));

    కొరకు (పూర్ణాంకం b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_బఫరు_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        ఒకవేళ (sched->n_copies > 1) {
            కొరకు (పూర్ణాంకం c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    తిరిగి sched;
}

శూన్య ggml_backend_sched_free(ggml_backend_sched_t sched) {
    ఒకవేళ (sched == ఖాళీ) {
        తిరిగి;
    }
    కొరకు (పూర్ణాంకం b = 0; b < sched->n_backends; b++) {
        కొరకు (పూర్ణాంకం c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_సమితి_free(&sched->hash_సమితి);
    కొరకు (పూర్ణాంకం i = 0; i < sched->splits_సామర్థ్యం; i++) {
        విడుదల(sched->splits[i].inputs);
    }
    విడుదల(sched->splits);
    విడుదల(sched->లేఖాచిత్రం_inputs);
    విడుదల(sched->hv_tensor_backend_ids);
    విడుదల(sched->hv_tensor_copies);
    విడుదల(sched->కణుపు_backend_ids);
    విడుదల(sched->ఆకు_backend_ids);
    విడుదల(sched->prev_కణుపు_backend_ids);
    విడుదల(sched->prev_ఆకు_backend_ids);
    విడుదల(sched->context_బఫరు);
    విడుదల(sched->graph.nodes);
    విడుదల(sched->graph.leafs);
    విడుదల(sched);
}

శూన్య ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    ఒకవేళ (!sched->is_reset) {
        ggml_hash_సమితి_reset(&sched->hash_సమితి);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_సమితి.size * పరిమాణం(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_సమితి.size * sched->n_backends * sched->n_copies * పరిమాణం(నిర్మాణం ggml_tensor *));
        sched->is_reset = నిజం;
    }
    sched->is_alloc = అబద్ధం;
}

శూన్య ggml_backend_sched_reserve_కొలత(ggml_backend_sched_t sched, నిర్మాణం ggml_cgraph * measure_లేఖాచిత్రం, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((పూర్ణాంకం)sched->hash_సమితి.size >= measure_లేఖాచిత్రం->n_nodes + measure_లేఖాచిత్రం->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_విభజించు_లేఖాచిత్రం(sched, measure_లేఖాచిత్రం);

    ggml_gallocr_reserve_n_కొలత(sched->galloc, &sched->graph, sched->కణుపు_backend_ids, sched->ఆకు_backend_ids, sizes);
}

తార్కిక ggml_backend_sched_reserve(ggml_backend_sched_t sched, నిర్మాణం ggml_cgraph * measure_లేఖాచిత్రం) {
    GGML_ASSERT(sched);
    GGML_ASSERT((పూర్ణాంకం)sched->hash_సమితి.size >= measure_లేఖాచిత్రం->n_nodes + measure_లేఖాచిత్రం->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_విభజించు_లేఖాచిత్రం(sched, measure_లేఖాచిత్రం);

    ఒకవేళ (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->కణుపు_backend_ids, sched->ఆకు_backend_ids)) {
        తిరిగి అబద్ధం;
    }

    ggml_backend_sched_reset(sched);

    తిరిగి నిజం;
}

తార్కిక ggml_backend_sched_alloc_లేఖాచిత్రం(ggml_backend_sched_t sched, నిర్మాణం ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((పూర్ణాంకం)sched->hash_సమితి.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->తదుపరి_copy;
    sched->తదుపరి_copy = (sched->తదుపరి_copy + 1) % sched->n_copies;

    ggml_backend_sched_విభజించు_లేఖాచిత్రం(sched, graph);

    ఒకవేళ (!ggml_backend_sched_alloc_splits(sched)) {
        తిరిగి అబద్ధం;
    }

    sched->is_alloc = నిజం;

    తిరిగి నిజం;
}

గణన ggml_స్థితి ggml_backend_sched_లేఖాచిత్రం_compute(ggml_backend_sched_t sched, నిర్మాణం ggml_cgraph * graph) {
    గణన ggml_స్థితి err = ggml_backend_sched_లేఖాచిత్రం_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    తిరిగి err;
}

గణన ggml_స్థితి ggml_backend_sched_లేఖాచిత్రం_compute_async(ggml_backend_sched_t sched, నిర్మాణం ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    ఒకవేళ (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    ఒకవేళ (!sched->is_alloc) {
        ఒకవేళ (!ggml_backend_sched_alloc_లేఖాచిత్రం(sched, graph)) {
            తిరిగి GGML_స్థితి_ALLOC_FAILED;
        }
    }

    తిరిగి ggml_backend_sched_compute_splits(sched);
}

శూన్య ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    కొరకు (పూర్ణాంకం i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    ఒకవేళ (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->తదుపరి_copy = 0;
    }
}

శూన్య ggml_backend_sched_సమితి_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, శూన్య * user_దత్తాంశం) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_దత్తాంశం = user_దత్తాంశం;
}

శూన్య ggml_backend_sched_సమితి_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, శూన్య * user_దత్తాంశం) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_దత్తాంశం = user_దత్తాంశం;
}

శూన్య ggml_backend_sched_సమితి_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, శూన్య * user_దత్తాంశం) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_దత్తాంశం = user_దత్తాంశం;
}

పూర్ణాంకం ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    తిరిగి sched->n_splits;
}

పూర్ణాంకం ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    తిరిగి sched->n_copies;
}

పూర్ణాంకం ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    తిరిగి sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, పూర్ణాంకం i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    తిరిగి sched->backends[i];
}

ggml_backend_బఫరు_type_t ggml_backend_sched_get_బఫరు_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    పూర్ణాంకం backend_సూచీ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_సూచీ >= 0 && backend_సూచీ < sched->n_backends);

    తిరిగి sched->bufts[backend_సూచీ];
}

size_t ggml_backend_sched_get_బఫరు_కొలత(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    పూర్ణాంకం backend_సూచీ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_సూచీ >= 0 && backend_సూచీ < sched->n_backends);

    తిరిగి ggml_gallocr_get_బఫరు_కొలత(sched->galloc, backend_సూచీ);
}

శూన్య ggml_backend_sched_సమితి_tensor_backend(ggml_backend_sched_t sched, నిర్మాణం ggml_tensor * కణుపు, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    పూర్ణాంకం backend_సూచీ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_సూచీ >= 0 && backend_సూచీ < sched->n_backends);
    tensor_backend_id(కణుపు) = backend_సూచీ;
    సమితి_CAUSE(కణుపు, "usr");
    sched->is_reset = అబద్ధం;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, నిర్మాణం ggml_tensor * కణుపు) {
    GGML_ASSERT(sched);
    పూర్ణాంకం backend_సూచీ = tensor_backend_id(కణుపు);
    ఒకవేళ (backend_సూచీ == -1) {
        తిరిగి ఖాళీ;
    }
    తిరిగి sched->backends[backend_సూచీ];
}

// utils

తార్కిక ggml_op_alloc_కొలత_may_expand(గణన ggml_op op) {
    ఎంపిక (op) {
        స్థితి GGML_OP_FLASH_ATTN_EXT:
        స్థితి GGML_OP_MUL_MAT:
        స్థితి GGML_OP_MUL_MAT_ID:
        స్థితి GGML_OP_CUMSUM:
        స్థితి GGML_OP_ARGSORT:
        స్థితి GGML_OP_శిఖరం_K:
        స్థితి GGML_OP_ARGMAX:
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

గణన ggml_స్థితి ggml_backend_view_init(నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->బఫరు == ఖాళీ);
    GGML_ASSERT(tensor->view_src != ఖాళీ);
    GGML_ASSERT(tensor->view_src->బఫరు != ఖాళీ);
    GGML_ASSERT(tensor->view_src->దత్తాంశం != ఖాళీ);

    tensor->బఫరు = tensor->view_src->బఫరు;
    tensor->దత్తాంశం = (అక్షరం *)tensor->view_src->దత్తాంశం + tensor->view_offs;
    తిరిగి ggml_backend_బఫరు_init_tensor(tensor->బఫరు, tensor);
}

గణన ggml_స్థితి ggml_backend_tensor_alloc(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor, శూన్య * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->బఫరు == ఖాళీ);
    GGML_ASSERT(tensor->దత్తాంశం == ఖాళీ);
    GGML_ASSERT(tensor->view_src == ఖాళీ);
    GGML_ASSERT(addr >= ggml_backend_బఫరు_get_పునాది(బఫరు));
    GGML_ASSERT(ggml_backend_బఫరు_is_meta(బఫరు) ||
        (అక్షరం *) addr + ggml_backend_బఫరు_get_alloc_కొలత(బఫరు, tensor) <=
        (అక్షరం *) ggml_backend_బఫరు_get_పునాది(బఫరు) + ggml_backend_బఫరు_get_కొలత(బఫరు));

    tensor->బఫరు = బఫరు;
    tensor->దత్తాంశం = addr;
    తిరిగి ggml_backend_బఫరు_init_tensor(బఫరు, tensor);
}

స్థిర నిర్మాణం ggml_tensor * లేఖాచిత్రం_copy_dup_tensor(నిర్మాణం ggml_hash_సమితి hash_సమితి, నిర్మాణం ggml_tensor ** కణుపు_copies,
    నిర్మాణం ggml_context * ctx_allocated, నిర్మాణం ggml_context * ctx_unallocated, నిర్మాణం ggml_tensor * src) {

    GGML_ASSERT(src != ఖాళీ);
    GGML_ASSERT(src->దత్తాంశం && "graph must be allocated");

    size_t id = ggml_hash_చొప్పించు(&hash_సమితి, src);
    ఒకవేళ (id == GGML_HASHSET_ALREADY_EXISTS) {
        తిరిగి కణుపు_copies[ggml_hash_కనుగొను(&hash_సమితి, src)];
    }

    నిర్మాణం ggml_tensor * dst = ggml_dup_tensor_layout(src->దత్తాంశం && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    ఒకవేళ (src->view_src != ఖాళీ) {
        dst->view_src = లేఖాచిత్రం_copy_dup_tensor(hash_సమితి, కణుపు_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    జ్ఞాపకం_నకలు(dst->op_params, src->op_params, పరిమాణం(dst->op_params));
    ggml_సమితి_name(dst, src->name);

    // copy src
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        నిర్మాణం ggml_tensor * s = src->src[i];
        ఒకవేళ (s == ఖాళీ) {
            కొనసాగు;
        }
        dst->src[i] = లేఖాచిత్రం_copy_dup_tensor(hash_సమితి, కణుపు_copies, ctx_allocated, ctx_unallocated, s);
    }

    కణుపు_copies[id] = dst;
    తిరిగి dst;
}

స్థిర శూన్య లేఖాచిత్రం_copy_init_tensor(నిర్మాణం ggml_hash_సమితి * hash_సమితి, నిర్మాణం ggml_tensor ** కణుపు_copies, తార్కిక * కణుపు_init, నిర్మాణం ggml_tensor * src) {
    size_t id = ggml_hash_కనుగొను(hash_సమితి, src);
    ఒకవేళ (కణుపు_init[id]) {
        తిరిగి;
    }
    కణుపు_init[id] = నిజం;

    నిర్మాణం ggml_tensor * dst = కణుపు_copies[id];
    ఒకవేళ (dst->view_src != ఖాళీ) {
        లేఖాచిత్రం_copy_init_tensor(hash_సమితి, కణుపు_copies, కణుపు_init, src->view_src);
        గణన ggml_స్థితి స్థితి_2 = ggml_backend_view_init(dst);
        GGML_ASSERT(స్థితి_2 == GGML_స్థితి_SUCCESS);
    }
    లేకపోతే {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        నిర్మాణం ggml_tensor * s = src->src[i];
        ఒకవేళ (s == ఖాళీ) {
            కొనసాగు;
        }
        లేఖాచిత్రం_copy_init_tensor(hash_సమితి, కణుపు_copies, కణుపు_init, s);
    }
}

నిర్మాణం ggml_backend_లేఖాచిత్రం_copy ggml_backend_లేఖాచిత్రం_copy(ggml_backend_t backend, నిర్మాణం ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    నిర్మాణం ggml_hash_సమితి hash_సమితి = ggml_hash_సమితి_new(graph->సందర్శించిన_hash_సమితి.size);
    నిర్మాణం ggml_tensor ** కణుపు_copies = (ggml_tensor **) సున్నా_జ్ఞాపకం_ఇవ్వు(hash_సమితి.size, పరిమాణం(కణుపు_copies[0])); // NOLINT
    తార్కిక * కణుపు_init = (తార్కిక *) సున్నా_జ్ఞాపకం_ఇవ్వు(hash_సమితి.size, పరిమాణం(కణుపు_init[0]));

    నిర్మాణం ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_సమితి.size + ggml_లేఖాచిత్రం_overhead_custom(graph->size, అబద్ధం),
        /* .mem_buffer = */ ఖాళీ,
        /* .no_alloc   = */ నిజం
    };

    నిర్మాణం ggml_context * ctx_allocated = ggml_init(params);
    నిర్మాణం ggml_context * ctx_unallocated = ggml_init(params);

    ఒకవేళ (ctx_allocated == ఖాళీ || ctx_unallocated == ఖాళీ) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_సమితి_free(&hash_సమితి);
        విడుదల(కణుపు_copies);
        విడుదల(కణుపు_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        తిరిగి {
            /* .buffer           = */ ఖాళీ,
            /* .ctx_allocated    = */ ఖాళీ,
            /* .ctx_unallocated  = */ ఖాళీ,
            /* .graph            = */ ఖాళీ,
        };
    }

    // dup nodes
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
        లేఖాచిత్రం_copy_dup_tensor(hash_సమితి, కణుపు_copies, ctx_allocated, ctx_unallocated, కణుపు);
    }

    // allocate nodes
    ggml_backend_బఫరు_t బఫరు = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    ఒకవేళ (బఫరు == ఖాళీ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_సమితి_free(&hash_సమితి);
        విడుదల(కణుపు_copies);
        విడుదల(కణుపు_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        తిరిగి {
            /* .buffer           = */ ఖాళీ,
            /* .ctx_allocated    = */ ఖాళీ,
            /* .ctx_unallocated  = */ ఖాళీ,
            /* .graph            = */ ఖాళీ,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
        లేఖాచిత్రం_copy_init_tensor(&hash_సమితి, కణుపు_copies, కణుపు_init, కణుపు);
    }

    // build graph copy
    నిర్మాణం ggml_cgraph * లేఖాచిత్రం_copy = ggml_new_లేఖాచిత్రం_custom(ctx_allocated, graph->size, అబద్ధం);
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = graph->nodes[i];
        నిర్మాణం ggml_tensor * కణుపు_copy = కణుపు_copies[ggml_hash_కనుగొను(&hash_సమితి, కణుపు)];
        లేఖాచిత్రం_copy->nodes[i] = కణుపు_copy;
    }
    లేఖాచిత్రం_copy->n_nodes = graph->n_nodes;

    ggml_hash_సమితి_free(&hash_సమితి);
    విడుదల(కణుపు_copies);
    విడుదల(కణుపు_init);

    తిరిగి {
        /* .buffer           = */ బఫరు,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ లేఖాచిత్రం_copy,
    };
}

శూన్య ggml_backend_లేఖాచిత్రం_copy_free(నిర్మాణం ggml_backend_లేఖాచిత్రం_copy copy) {
    ggml_backend_బఫరు_free(copy.బఫరు);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

తార్కిక ggml_backend_compare_లేఖాచిత్రం_backend(ggml_backend_t backend1, ggml_backend_t backend2, నిర్మాణం ggml_cgraph * graph, ggml_backend_eval_callback callback, శూన్య * user_దత్తాంశం, నిర్మాణం ggml_tensor స్థిరరాశి * స్థిరరాశి * test_nodes, size_t అంకె_test_nodes) {
    నిర్మాణం ggml_backend_లేఖాచిత్రం_copy copy = ggml_backend_లేఖాచిత్రం_copy(backend2, graph);
    ఒకవేళ (copy.బఫరు == ఖాళీ) {
        తిరిగి అబద్ధం;
    }

    నిర్మాణం ggml_cgraph * g1 = graph;
    నిర్మాణం ggml_cgraph * g2 = copy.graph;

    ప్రతిజ్ఞా(g1->n_nodes == g2->n_nodes);

    ఒకవేళ (అంకె_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_లేఖాచిత్రం_compute(backend1, g1);
        ggml_backend_లేఖాచిత్రం_compute(backend2, g2);

        తార్కిక verified = అబద్ధం;
        కొరకు (పూర్ణాంకం i = 0; i < g1->n_nodes; i++) {
            కొరకు (size_t j = 0; j < అంకె_test_nodes; ++j) {
                ఒకవేళ (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_దత్తాంశం);
                    verified = నిజం;
                }
            }
        }
        GGML_ASSERT(verified);
    } లేకపోతే {
        కొరకు (పూర్ణాంకం i = 0; i < g1->n_nodes; i++) {
            నిర్మాణం ggml_tensor * t1 = g1->nodes[i];
            నిర్మాణం ggml_tensor * t2 = g2->nodes[i];

            ప్రతిజ్ఞా(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            నిర్మాణం ggml_cgraph g1v = ggml_లేఖాచిత్రం_view(g1, i, i + 1);
            నిర్మాణం ggml_cgraph g2v = ggml_లేఖాచిత్రం_view(g2, i, i + 1);

            ggml_backend_లేఖాచిత్రం_compute(backend1, &g1v);
            ggml_backend_లేఖాచిత్రం_compute(backend2, &g2v);

            ఒకవేళ (ggml_is_view_op(t1->op)) {
                కొనసాగు;
            }

            // compare results, calculate rms etc
            ఒకవేళ (!callback(i, t1, t2, user_దత్తాంశం)) {
                ఆపు;
            }
        }
    }
    ggml_backend_లేఖాచిత్రం_copy_free(copy);

    తిరిగి నిజం;
}

// CPU backend - buffer

స్థిర శూన్య * ggml_backend_cpu_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);
    uintptr_t దత్తాంశం = (uintptr_t)బఫరు->context;

    // align the buffer
    ఒకవేళ (దత్తాంశం % TENSOR_ALIGNMENT != 0) {
        దత్తాంశం = GGML_PAD(దత్తాంశం, TENSOR_ALIGNMENT);
    }

    తిరిగి (శూన్య *)దత్తాంశం;
}

స్థిర శూన్య ggml_backend_cpu_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);
    ggml_aligned_free(బఫరు->context, బఫరు->size);
}

స్థిర శూన్య ggml_backend_cpu_బఫరు_memset_tensor(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor, uint8_t value, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(tensor);
    memset((అక్షరం *)tensor->దత్తాంశం + స్థానభ్రంశం, value, size);

    GGML_వాడని(బఫరు);
}

స్థిర శూన్య ggml_backend_cpu_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(tensor);
    జ్ఞాపకం_నకలు((అక్షరం *)tensor->దత్తాంశం + స్థానభ్రంశం, దత్తాంశం, size);

    GGML_వాడని(బఫరు);
}

స్థిర శూన్య ggml_backend_cpu_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి నిర్మాణం ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(tensor);
    జ్ఞాపకం_నకలు(దత్తాంశం, (స్థిరరాశి అక్షరం *)tensor->దత్తాంశం + స్థానభ్రంశం, size);

    GGML_వాడని(బఫరు);
}

స్థిర తార్కిక ggml_backend_cpu_బఫరు_cpy_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి నిర్మాణం ggml_tensor * src, నిర్మాణం ggml_tensor * dst) {
    GGML_ASSERT(src);
    ఒకవేళ (ggml_backend_బఫరు_is_host(src->బఫరు)) {
        జ్ఞాపకం_నకలు(dst->దత్తాంశం, src->దత్తాంశం, ggml_nbytes(src));
        తిరిగి నిజం;
    }
    తిరిగి అబద్ధం;

    GGML_వాడని(బఫరు);
}

స్థిర శూన్య ggml_backend_cpu_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    GGML_ASSERT(బఫరు);
    memset(బఫరు->context, value, బఫరు->size);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_బఫరు_i ggml_backend_cpu_బఫరు_i = {
    /* .free_buffer     = */ ggml_backend_cpu_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_cpu_బఫరు_get_పునాది,
    /* .init_tensor     = */ ఖాళీ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_బఫరు_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ggml_backend_cpu_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_బఫరు_i ggml_backend_cpu_బఫరు_from_ptr_i = {
    /* .free_buffer     = */ ఖాళీ, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_బఫరు_get_పునాది,
    /* .init_tensor     = */ ఖాళీ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_బఫరు_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ggml_backend_cpu_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cpu_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి "CPU";

    GGML_వాడని(buft);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_cpu_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    శూన్య * దత్తాంశం = ggml_aligned_malloc(size);

    ఒకవేళ (దత్తాంశం == ఖాళీ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        తిరిగి ఖాళీ;
    }

    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_cpu_బఫరు_i, దత్తాంశం, size);
}

స్థిర size_t ggml_backend_cpu_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి TENSOR_ALIGNMENT;

    GGML_వాడని(buft);
}

స్థిర తార్కిక ggml_backend_cpu_బఫరు_type_is_host(ggml_backend_బఫరు_type_t buft) {
    తిరిగి నిజం;

    GGML_వాడని(buft);
}

ggml_backend_బఫరు_type_t ggml_backend_cpu_బఫరు_type(శూన్య) {
    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_cpu_బఫరు_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_బఫరు_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_బఫరు_type_alloc_బఫరు,
            /* .alloc_buffer_n      = */ ఖాళీ,
            /* .get_alignment       = */ ggml_backend_cpu_బఫరు_type_get_alignment,
            /* .get_max_size        = */ ఖాళీ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ఖాళీ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ఖాళీ,
            /* .is_host             = */ ggml_backend_cpu_బఫరు_type_is_host,
        },
        /* .device  = */ ఖాళీ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ఖాళీ,
    };

    తిరిగి &ggml_backend_cpu_బఫరు_type;
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cpu_బఫరు_from_ptr_type_get_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి "CPU_Mapped";

    GGML_వాడని(buft);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_cpu_బఫరు_from_ptr_type(శూన్య) {
    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_cpu_బఫరు_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_బఫరు_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_బఫరు_type_alloc_బఫరు,
            /* .alloc_buffer_n      = */ ఖాళీ,
            /* .get_alignment       = */ ggml_backend_cpu_బఫరు_type_get_alignment,
            /* .get_max_size        = */ ఖాళీ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ఖాళీ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ఖాళీ,
            /* .is_host             = */ ggml_backend_cpu_బఫరు_type_is_host,
        },
        /* .device  = */ ఖాళీ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ఖాళీ,
    };

    తిరిగి &ggml_backend_cpu_బఫరు_type;
}

ggml_backend_బఫరు_t ggml_backend_cpu_బఫరు_from_ptr(శూన్య * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    తిరిగి ggml_backend_బఫరు_init(ggml_backend_cpu_బఫరు_from_ptr_type(), ggml_backend_cpu_బఫరు_from_ptr_i, ptr, size);
}
