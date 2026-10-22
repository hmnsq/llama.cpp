#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-blas.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"

#അന്തര്ഭാവ <future>
#അന്തര്ഭാവ <vector>
#അന്തര്ഭാവ <cstring>

#എങ്കിൽ defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#അല്ലെങ്കിൽ
#   include <cblas.h>
#endif

ഘടന ggml_backend_blas_context {
    പൂർണ്ണസംഖ്യ n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<അക്ഷരം[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<ശൂന്യം>> tasks;
#endif
};

സ്ഥിതം ശൂന്യം ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, ഘടന ggml_tensor * dst) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം എണ്ണൽ ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    സ്ഥിരം int64_t r2 = ne12/ne02;
    സ്ഥിരം int64_t r3 = ne13/ne03;

    സ്ഥിരം int64_t ne_plane      = ne01*ne00;
    സ്ഥിരം size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*വലുപ്പം(ദശാംശം);

    എങ്കിൽ (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(പുതിയ അക്ഷരം[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    ശൂന്യം * wdata = ctx->work_data.get();

    // convert src0 to float
    എങ്കിൽ (type != GGML_TYPE_F32) {
        സ്ഥിരം auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t സ്ഥിരം to_float = type_traits->to_float;

        വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
            വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                സ്ഥിരം ശൂന്യം  *       x      = (അക്ഷരം *)  src0->data + i02*nb02          + i03*nb03;
                      ദശാംശം * സ്ഥിരം wplane = (ദശാംശം *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                സ്ഥിരം പൂർണ്ണസംഖ്യ min_cols_per_thread = 4096;
                സ്ഥിരം പൂർണ്ണസംഖ്യ min_rows_per_thread = std::max((പൂർണ്ണസംഖ്യ)(min_cols_per_thread/ne00), 1);
                സ്ഥിരം പൂർണ്ണസംഖ്യ n_threads = std::max(std::min(ctx->n_threads, (പൂർണ്ണസംഖ്യ)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel വേണ്ടി അക്കം_threads(n_threads)
                വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((സ്ഥിരം അക്ഷരം *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#അല്ലെങ്കിൽ
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < n_threads; i++) {
                    സ്ഥിരം int64_t start =       i*ne01/n_threads;
                    സ്ഥിരം int64_t end   = (i + 1)*ne01/n_threads;
                    എങ്കിൽ (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            വേണ്ടി (int64_t i01 = start; i01 < end; i01++) {
                                to_float((സ്ഥിരം അക്ഷരം *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    സ്ഥിരം int64_t start = 0;
                    സ്ഥിരം int64_t end   = ne01/n_threads;
                    വേണ്ടി (int64_t i01 = start; i01 < end; i01++) {
                        to_float((സ്ഥിരം അക്ഷരം *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        വേണ്ടി (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#എങ്കിൽ defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_അക്കം_threads(ctx->n_threads);
#അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_അക്കം_threads(ctx->n_threads);
#അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_അക്കം_threads(ctx->n_threads);
#അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_MKL)
    mkl_set_അക്കം_threads(ctx->n_threads);
#endif

    വേണ്ടി (int64_t i13 = 0; i13 < ne13; i13++) {
        വേണ്ടി (int64_t i12 = 0; i12 < ne12; i12++) {
            സ്ഥിരം int64_t i03 = i13/r3;
            സ്ഥിരം int64_t i02 = i12/r2;

            സ്ഥിരം ദശാംശം * x = (ദശാംശം *) ((അക്ഷരം *) src0->data + i02*nb02 + i03*nb03);
            സ്ഥിരം ദശാംശം * y = (ദശാംശം *) ((അക്ഷരം *) src1->data + i12*nb12 + i13*nb13);
                  ദശാംശം * d = (ദശാംശം *) ((അക്ഷരം *)  dst->data + i12*nb2  + i13*nb3);

            എങ്കിൽ (type != GGML_TYPE_F32) {
                x = (ദശാംശം *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, ഘടന ggml_tensor * dst) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // Arguments to ggml_compute_forward_out_prod (expressed as major,minor)
    // src0: (k,n)
    // src1: (k,m)
    // dst:  (m,n)
    //
    // Arguments to sgemm (see https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/sgemm.f)
    // Also expressed as (major,minor)
    // a: (m,k): so src1 transposed
    // b: (k,n): so src0
    // c: (m,n)
    //
    // However, if ggml_is_transposed(src1) is true, then
    // src1->data already contains a transposed version, so sgemm mustn't
    // transpose it further.

    പൂർണ്ണസംഖ്യ n = src0->ne[0];
    പൂർണ്ണസംഖ്യ k = src0->ne[1];
    പൂർണ്ണസംഖ്യ m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    പൂർണ്ണസംഖ്യ lda;

    എങ്കിൽ (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } അല്ലെങ്കിൽ {
        transposeA = CblasNoTrans;
        lda = k;
    }

    ദശാംശം * a = (ദശാംശം *) ((അക്ഷരം *) src1->data);
    ദശാംശം * b = (ദശാംശം *) ((അക്ഷരം *) src0->data);
    ദശാംശം * c = (ദശാംശം *) ((അക്ഷരം *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_blas_get_name(ggml_backend_t backend) {
    തിരികെ "BLAS";

    GGML_UNUSED(backend);
}

സ്ഥിതം ശൂന്യം ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, ഘടന ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[i];

        എങ്കിൽ ((കണ്ണി->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            തുടരുക;
        }

        തിരഞ്ഞെടുപ്പ് (കണ്ണി->op) {
            അവസ്ഥ GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, കണ്ണി);
                നിർത്തുക;

            അവസ്ഥ GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, കണ്ണി);
                നിർത്തുക;

            അവസ്ഥ GGML_OP_NONE:
            അവസ്ഥ GGML_OP_RESHAPE:
            അവസ്ഥ GGML_OP_VIEW:
            അവസ്ഥ GGML_OP_PERMUTE:
            അവസ്ഥ GGML_OP_TRANSPOSE:
                നിർത്തുക;

            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(കണ്ണി));
        }
    }

    തിരികെ GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

സ്ഥിതം ഘടന ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ ശൂന്യ,
    /* .get_tensor_async        = */ ശൂന്യ,
    /* .set_tensor_2d_async     = */ ശൂന്യ,
    /* .get_tensor_2d_async     = */ ശൂന്യ,
    /* .cpy_tensor_async        = */ ശൂന്യ,
    /* .synchronize             = */ ശൂന്യ,
    /* .graph_plan_create       = */ ശൂന്യ,
    /* .graph_plan_free         = */ ശൂന്യ,
    /* .graph_plan_update       = */ ശൂന്യ,
    /* .graph_plan_compute      = */ ശൂന്യ,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ ശൂന്യ,
    /* .event_wait              = */ ശൂന്യ,
    /* .graph_optimize          = */ ശൂന്യ,
};

സ്ഥിതം ggml_guid_t ggml_backend_blas_guid(ശൂന്യം) {
    സ്ഥിതം ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    തിരികെ &guid;
}

ggml_backend_t ggml_backend_blas_init(ശൂന്യം) {
    ggml_backend_blas_context * ctx = പുതിയ ggml_backend_blas_context;

    ggml_backend_t backend = പുതിയ ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#എങ്കിൽ defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    എങ്കിൽ (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#എങ്കിൽ defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    തിരികെ backend;
}

തർക്കം ggml_backend_is_blas(ggml_backend_t backend) {
    തിരികെ backend != ശൂന്യ && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

ശൂന്യം ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, പൂർണ്ണസംഖ്യ n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    തിരികെ "BLAS";

    GGML_UNUSED(dev);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #എങ്കിൽ defined(GGML_BLAS_USE_ACCELERATE)
        തിരികെ "Accelerate";
    #അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_MKL)
        തിരികെ "MKL";
    #അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_AOCL)
        തിരികെ "AOCL-BLAS";
    #അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_BLIS)
        തിരികെ "BLIS";
    #അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_NVPL)
        തിരികെ "NVPL";
    #അല്ലെങ്കിലെങ്കിൽ defined(GGML_BLAS_USE_OPENBLAS)
        തിരികെ "OpenBLAS";
    #അല്ലെങ്കിൽ
        തിരികെ "BLAS";
    #endif

    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * സ്വതന്ത്രം, size_t * total) {
    // no memory to report
    *സ്വതന്ത്രം  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

സ്ഥിതം എണ്ണൽ ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    തിരികെ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, ഘടന ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ അസത്യം,
        /* .host_buffer           = */ അസത്യം,
        /* .buffer_from_host_ptr  = */ സത്യം,
        /* .events                = */ അസത്യം,
        /* .mmap_support          = */ സത്യം,
    };
}

സ്ഥിതം ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, സ്ഥിരം അക്ഷരം * params) {
    തിരികെ ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    തിരികെ ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ശൂന്യം * ptr, size_t size, size_t max_tensor_size) {
    തിരികെ ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

സ്ഥിതം തർക്കം ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];

    തിരഞ്ഞെടുപ്പ് (op->op) {
        അവസ്ഥ GGML_OP_NONE:
        അവസ്ഥ GGML_OP_RESHAPE:
        അവസ്ഥ GGML_OP_VIEW:
        അവസ്ഥ GGML_OP_PERMUTE:
        അവസ്ഥ GGML_OP_TRANSPOSE:
            തിരികെ സത്യം;

        അവസ്ഥ GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
            സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];

            സ്ഥിരം int64_t ne10 = src1->ne[0];

            സ്ഥിരം int64_t ne0 = op->ne[0];
            സ്ഥിരം int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            സ്ഥിരം int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            എങ്കിൽ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                തിരികെ അസത്യം;
            }

            തിരികെ ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != ശൂന്യ);
        }

        അവസ്ഥ GGML_OP_OUT_PROD:
            തിരികെ op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != ശൂന്യ);

        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;

    }

    GGML_UNUSED(dev);
}

സ്ഥിതം തർക്കം ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    തിരികെ ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ ശൂന്യ,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ ശൂന്യ,
    /* .event_new            = */ ശൂന്യ,
    /* .event_free           = */ ശൂന്യ,
    /* .event_synchronize    = */ ശൂന്യ,
};

// backend reg interface

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    തിരികെ "BLAS";

    GGML_UNUSED(reg);
}

സ്ഥിതം size_t ggml_backend_blas_reg_get_device_എണ്ണം(ggml_backend_reg_t reg) {
    തിരികെ 1;

    GGML_UNUSED(reg);
}

സ്ഥിതം ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    സ്ഥിതം ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    തിരികെ &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

സ്ഥിതം ശൂന്യം * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, സ്ഥിരം അക്ഷരം * name) {
    എങ്കിൽ (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_blas_set_n_threads;
    }
    തിരികെ ശൂന്യ;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_എണ്ണം,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(ശൂന്യം) {
    സ്ഥിതം ഘടന ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ ശൂന്യ,
    };

    തിരികെ &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
