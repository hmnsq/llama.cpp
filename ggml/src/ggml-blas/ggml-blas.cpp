#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-blas.h"
#அந்தர்பாவ "ggml-backend-impl.h"

#அந்தர்பாவ <future>
#அந்தர்பாவ <vector>
#அந்தர்பாவ <cstring>

#எனில் defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#அல்லதெனில் defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#அல்லதெனில் defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#அல்லதெனில் defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#இல்லையெனில்
#   include <cblas.h>
#endif

கட்டமைப்பு ggml_backend_blas_context {
    முழு n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<எழுத்து[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<வெறுமை>> tasks;
#endif
};

நிலையான வெறுமை ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, கட்டமைப்பு ggml_tensor * dst) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = dst->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி எண்ணிக்கை ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == அளவு(தசம));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    மாறிலி int64_t r2 = ne12/ne02;
    மாறிலி int64_t r3 = ne13/ne03;

    மாறிலி int64_t ne_plane      = ne01*ne00;
    மாறிலி size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*அளவு(தசம);

    எனில் (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(புதிய எழுத்து[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    வெறுமை * wdata = ctx->work_data.get();

    // convert src0 to float
    எனில் (type != GGML_TYPE_F32) {
        மாறிலி auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t மாறிலி to_float = type_traits->to_float;

        க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
            க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                மாறிலி வெறுமை  *       x      = (எழுத்து *)  src0->data + i02*nb02          + i03*nb03;
                      தசம * மாறிலி wplane = (தசம *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                மாறிலி முழு min_cols_per_thread = 4096;
                மாறிலி முழு min_rows_per_thread = std::max((முழு)(min_cols_per_thread/ne00), 1);
                மாறிலி முழு n_threads = std::max(std::min(ctx->n_threads, (முழு)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel க்கு இலக்கம்_threads(n_threads)
                க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((மாறிலி எழுத்து *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#இல்லையெனில்
                க்கு (முழு i = 1; i < n_threads; i++) {
                    மாறிலி int64_t start =       i*ne01/n_threads;
                    மாறிலி int64_t end   = (i + 1)*ne01/n_threads;
                    எனில் (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            க்கு (int64_t i01 = start; i01 < end; i01++) {
                                to_float((மாறிலி எழுத்து *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    மாறிலி int64_t start = 0;
                    மாறிலி int64_t end   = ne01/n_threads;
                    க்கு (int64_t i01 = start; i01 < end; i01++) {
                        to_float((மாறிலி எழுத்து *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        க்கு (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#எனில் defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_இலக்கம்_threads(ctx->n_threads);
#அல்லதெனில் defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_இலக்கம்_threads(ctx->n_threads);
#அல்லதெனில் defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_இலக்கம்_threads(ctx->n_threads);
#அல்லதெனில் defined(GGML_BLAS_USE_MKL)
    mkl_set_இலக்கம்_threads(ctx->n_threads);
#endif

    க்கு (int64_t i13 = 0; i13 < ne13; i13++) {
        க்கு (int64_t i12 = 0; i12 < ne12; i12++) {
            மாறிலி int64_t i03 = i13/r3;
            மாறிலி int64_t i02 = i12/r2;

            மாறிலி தசம * x = (தசம *) ((எழுத்து *) src0->data + i02*nb02 + i03*nb03);
            மாறிலி தசம * y = (தசம *) ((எழுத்து *) src1->data + i12*nb12 + i13*nb13);
                  தசம * d = (தசம *) ((எழுத்து *)  dst->data + i12*nb2  + i13*nb3);

            எனில் (type != GGML_TYPE_F32) {
                x = (தசம *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

நிலையான வெறுமை ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, கட்டமைப்பு ggml_tensor * dst) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = dst->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == அளவு(தசம));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == அளவு(தசம));
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

    முழு n = src0->ne[0];
    முழு k = src0->ne[1];
    முழு m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    முழு lda;

    எனில் (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } இல்லையெனில் {
        transposeA = CblasNoTrans;
        lda = k;
    }

    தசம * a = (தசம *) ((எழுத்து *) src1->data);
    தசம * b = (தசம *) ((எழுத்து *) src0->data);
    தசம * c = (தசம *) ((எழுத்து *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

நிலையான மாறிலி எழுத்து * ggml_backend_blas_get_name(ggml_backend_t backend) {
    திரும்பு "BLAS";

    GGML_UNUSED(backend);
}

நிலையான வெறுமை ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, கட்டமைப்பு ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[i];

        எனில் ((முனை->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            தொடர்;
        }

        தேர்வு (முனை->op) {
            நிலை GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, முனை);
                நிறுத்து;

            நிலை GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, முனை);
                நிறுத்து;

            நிலை GGML_OP_NONE:
            நிலை GGML_OP_RESHAPE:
            நிலை GGML_OP_VIEW:
            நிலை GGML_OP_PERMUTE:
            நிலை GGML_OP_TRANSPOSE:
                நிறுத்து;

            இயல்பு:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(முனை));
        }
    }

    திரும்பு GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

நிலையான கட்டமைப்பு ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ வெற்று,
    /* .get_tensor_async        = */ வெற்று,
    /* .set_tensor_2d_async     = */ வெற்று,
    /* .get_tensor_2d_async     = */ வெற்று,
    /* .cpy_tensor_async        = */ வெற்று,
    /* .synchronize             = */ வெற்று,
    /* .graph_plan_create       = */ வெற்று,
    /* .graph_plan_free         = */ வெற்று,
    /* .graph_plan_update       = */ வெற்று,
    /* .graph_plan_compute      = */ வெற்று,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ வெற்று,
    /* .event_wait              = */ வெற்று,
    /* .graph_optimize          = */ வெற்று,
};

நிலையான ggml_guid_t ggml_backend_blas_guid(வெறுமை) {
    நிலையான ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    திரும்பு &guid;
}

ggml_backend_t ggml_backend_blas_init(வெறுமை) {
    ggml_backend_blas_context * ctx = புதிய ggml_backend_blas_context;

    ggml_backend_t backend = புதிய ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#எனில் defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    எனில் (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#எனில் defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    திரும்பு backend;
}

தருக்க ggml_backend_is_blas(ggml_backend_t backend) {
    திரும்பு backend != வெற்று && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

வெறுமை ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, முழு n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

நிலையான மாறிலி எழுத்து * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    திரும்பு "BLAS";

    GGML_UNUSED(dev);
}

நிலையான மாறிலி எழுத்து * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #எனில் defined(GGML_BLAS_USE_ACCELERATE)
        திரும்பு "Accelerate";
    #அல்லதெனில் defined(GGML_BLAS_USE_MKL)
        திரும்பு "MKL";
    #அல்லதெனில் defined(GGML_BLAS_USE_AOCL)
        திரும்பு "AOCL-BLAS";
    #அல்லதெனில் defined(GGML_BLAS_USE_BLIS)
        திரும்பு "BLIS";
    #அல்லதெனில் defined(GGML_BLAS_USE_NVPL)
        திரும்பு "NVPL";
    #அல்லதெனில் defined(GGML_BLAS_USE_OPENBLAS)
        திரும்பு "OpenBLAS";
    #இல்லையெனில்
        திரும்பு "BLAS";
    #endif

    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * விடுவி, size_t * total) {
    // no memory to report
    *விடுவி  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

நிலையான எண்ணிக்கை ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    திரும்பு GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, கட்டமைப்பு ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ பொய்,
        /* .host_buffer           = */ பொய்,
        /* .buffer_from_host_ptr  = */ உண்மை,
        /* .events                = */ பொய்,
        /* .mmap_support          = */ உண்மை,
    };
}

நிலையான ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, மாறிலி எழுத்து * params) {
    திரும்பு ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

நிலையான ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    திரும்பு ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

நிலையான ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, வெறுமை * ptr, size_t size, size_t max_tensor_size) {
    திரும்பு ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

நிலையான தருக்க ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];

    தேர்வு (op->op) {
        நிலை GGML_OP_NONE:
        நிலை GGML_OP_RESHAPE:
        நிலை GGML_OP_VIEW:
        நிலை GGML_OP_PERMUTE:
        நிலை GGML_OP_TRANSPOSE:
            திரும்பு உண்மை;

        நிலை GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
            மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];

            மாறிலி int64_t ne10 = src1->ne[0];

            மாறிலி int64_t ne0 = op->ne[0];
            மாறிலி int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            மாறிலி int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            எனில் (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                திரும்பு பொய்;
            }

            திரும்பு ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != வெற்று);
        }

        நிலை GGML_OP_OUT_PROD:
            திரும்பு op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != வெற்று);

        இயல்பு:
            திரும்பு பொய்;

    }

    GGML_UNUSED(dev);
}

நிலையான தருக்க ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    திரும்பு ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ வெற்று,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ வெற்று,
    /* .event_new            = */ வெற்று,
    /* .event_free           = */ வெற்று,
    /* .event_synchronize    = */ வெற்று,
};

// backend reg interface

நிலையான மாறிலி எழுத்து * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    திரும்பு "BLAS";

    GGML_UNUSED(reg);
}

நிலையான size_t ggml_backend_blas_reg_get_device_மொத்தம்(ggml_backend_reg_t reg) {
    திரும்பு 1;

    GGML_UNUSED(reg);
}

நிலையான ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    நிலையான ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    திரும்பு &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

நிலையான வெறுமை * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, மாறிலி எழுத்து * name) {
    எனில் (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_blas_set_n_threads;
    }
    திரும்பு வெற்று;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_மொத்தம்,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(வெறுமை) {
    நிலையான கட்டமைப்பு ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ வெற்று,
    };

    திரும்பு &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
