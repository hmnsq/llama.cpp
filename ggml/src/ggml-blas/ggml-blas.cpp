#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-blas.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ <future>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <cstring>

#ಯದಿ defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#ಅನ್ಯಥಾ
#   include <cblas.h>
#endif

ಸಂರಚನಾ ggml_backend_blas_context {
    ಪೂರ್ಣಾಙ್ಕ n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<ವರ್ಣ[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<ಶೂನ್ಯ>> tasks;
#endif
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, ಸಂರಚನಾ ggml_tensor * dst) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಧ್ರುವ ಗಣನಾ ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಪರಿಮಾಣ(ಭಿನ್ನ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ಧ್ರುವ int64_t r2 = ne12/ne02;
    ಧ್ರುವ int64_t r3 = ne13/ne03;

    ಧ್ರುವ int64_t ne_plane      = ne01*ne00;
    ಧ್ರುವ size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಯದಿ (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(ನೂತನ ವರ್ಣ[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    ಶೂನ್ಯ * wdata = ctx->work_data.get();

    // convert src0 to float
    ಯದಿ (type != GGML_TYPE_F32) {
        ಧ್ರುವ auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t ಧ್ರುವ to_float = type_traits->to_float;

        ಪ್ರತಿ (int64_t i03 = 0; i03 < ne03; i03++) {
            ಪ್ರತಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                ಧ್ರುವ ಶೂನ್ಯ  *       x      = (ವರ್ಣ *)  src0->data + i02*nb02          + i03*nb03;
                      ಭಿನ್ನ * ಧ್ರುವ wplane = (ಭಿನ್ನ *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ min_cols_per_thread = 4096;
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ min_rows_per_thread = std::max((ಪೂರ್ಣಾಙ್ಕ)(min_cols_per_thread/ne00), 1);
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_threads = std::max(std::min(ctx->n_threads, (ಪೂರ್ಣಾಙ್ಕ)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel ಪ್ರತಿ ಅಂಕೆ_threads(n_threads)
                ಪ್ರತಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((ಧ್ರುವ ವರ್ಣ *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#ಅನ್ಯಥಾ
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < n_threads; i++) {
                    ಧ್ರುವ int64_t start =       i*ne01/n_threads;
                    ಧ್ರುವ int64_t end   = (i + 1)*ne01/n_threads;
                    ಯದಿ (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            ಪ್ರತಿ (int64_t i01 = start; i01 < end; i01++) {
                                to_float((ಧ್ರುವ ವರ್ಣ *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    ಧ್ರುವ int64_t start = 0;
                    ಧ್ರುವ int64_t end   = ne01/n_threads;
                    ಪ್ರತಿ (int64_t i01 = start; i01 < end; i01++) {
                        to_float((ಧ್ರುವ ವರ್ಣ *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        ಪ್ರತಿ (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#ಯದಿ defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_ಅಂಕೆ_threads(ctx->n_threads);
#ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_ಅಂಕೆ_threads(ctx->n_threads);
#ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_ಅಂಕೆ_threads(ctx->n_threads);
#ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_MKL)
    mkl_set_ಅಂಕೆ_threads(ctx->n_threads);
#endif

    ಪ್ರತಿ (int64_t i13 = 0; i13 < ne13; i13++) {
        ಪ್ರತಿ (int64_t i12 = 0; i12 < ne12; i12++) {
            ಧ್ರುವ int64_t i03 = i13/r3;
            ಧ್ರುವ int64_t i02 = i12/r2;

            ಧ್ರುವ ಭಿನ್ನ * x = (ಭಿನ್ನ *) ((ವರ್ಣ *) src0->data + i02*nb02 + i03*nb03);
            ಧ್ರುವ ಭಿನ್ನ * y = (ಭಿನ್ನ *) ((ವರ್ಣ *) src1->data + i12*nb12 + i13*nb13);
                  ಭಿನ್ನ * d = (ಭಿನ್ನ *) ((ವರ್ಣ *)  dst->data + i12*nb2  + i13*nb3);

            ಯದಿ (type != GGML_TYPE_F32) {
                x = (ಭಿನ್ನ *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, ಸಂರಚನಾ ggml_tensor * dst) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ಪರಿಮಾಣ(ಭಿನ್ನ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಪರಿಮಾಣ(ಭಿನ್ನ));
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

    ಪೂರ್ಣಾಙ್ಕ n = src0->ne[0];
    ಪೂರ್ಣಾಙ್ಕ k = src0->ne[1];
    ಪೂರ್ಣಾಙ್ಕ m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    ಪೂರ್ಣಾಙ್ಕ lda;

    ಯದಿ (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } ಅನ್ಯಥಾ {
        transposeA = CblasNoTrans;
        lda = k;
    }

    ಭಿನ್ನ * a = (ಭಿನ್ನ *) ((ವರ್ಣ *) src1->data);
    ಭಿನ್ನ * b = (ಭಿನ್ನ *) ((ವರ್ಣ *) src0->data);
    ಭಿನ್ನ * c = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_blas_get_name(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ "BLAS";

    GGML_UNUSED(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, ಸಂರಚನಾ ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        ಯದಿ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಅನುವರ್ತ;
        }

        ವಿಕಲ್ಪ (ಗಂಟು->op) {
            ಪಕ್ಷ GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, ಗಂಟು);
                ಭಙ್ಗ;

            ಪಕ್ಷ GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, ಗಂಟು);
                ಭಙ್ಗ;

            ಪಕ್ಷ GGML_OP_NONE:
            ಪಕ್ಷ GGML_OP_RESHAPE:
            ಪಕ್ಷ GGML_OP_VIEW:
            ಪಕ್ಷ GGML_OP_PERMUTE:
            ಪಕ್ಷ GGML_OP_TRANSPOSE:
                ಭಙ್ಗ;

            ಸ್ವಭಾವ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(ಗಂಟು));
        }
    }

    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ ರಿಕ್ತ,
    /* .get_tensor_async        = */ ರಿಕ್ತ,
    /* .set_tensor_2d_async     = */ ರಿಕ್ತ,
    /* .get_tensor_2d_async     = */ ರಿಕ್ತ,
    /* .cpy_tensor_async        = */ ರಿಕ್ತ,
    /* .synchronize             = */ ರಿಕ್ತ,
    /* .graph_plan_create       = */ ರಿಕ್ತ,
    /* .graph_plan_free         = */ ರಿಕ್ತ,
    /* .graph_plan_update       = */ ರಿಕ್ತ,
    /* .graph_plan_compute      = */ ರಿಕ್ತ,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ ರಿಕ್ತ,
    /* .event_wait              = */ ರಿಕ್ತ,
    /* .graph_optimize          = */ ರಿಕ್ತ,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_blas_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    ಪ್ರತ್ಯಾಗಮ &guid;
}

ggml_backend_t ggml_backend_blas_init(ಶೂನ್ಯ) {
    ggml_backend_blas_context * ctx = ನೂತನ ggml_backend_blas_context;

    ggml_backend_t backend = ನೂತನ ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#ಯದಿ defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    ಯದಿ (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#ಯದಿ defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    ಪ್ರತ್ಯಾಗಮ backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_blas(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ backend != ರಿಕ್ತ && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

ಶೂನ್ಯ ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, ಪೂರ್ಣಾಙ್ಕ n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ "BLAS";

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #ಯದಿ defined(GGML_BLAS_USE_ACCELERATE)
        ಪ್ರತ್ಯಾಗಮ "Accelerate";
    #ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_MKL)
        ಪ್ರತ್ಯಾಗಮ "MKL";
    #ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_AOCL)
        ಪ್ರತ್ಯಾಗಮ "AOCL-BLAS";
    #ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_BLIS)
        ಪ್ರತ್ಯಾಗಮ "BLIS";
    #ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_NVPL)
        ಪ್ರತ್ಯಾಗಮ "NVPL";
    #ನೋಚೇದ್ಯದಿ defined(GGML_BLAS_USE_OPENBLAS)
        ಪ್ರತ್ಯಾಗಮ "OpenBLAS";
    #ಅನ್ಯಥಾ
        ಪ್ರತ್ಯಾಗಮ "BLAS";
    #endif

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * ಮುಕ್ತ, size_t * total) {
    // no memory to report
    *ಮುಕ್ತ  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಗಣನಾ ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, ಸಂರಚನಾ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ಅಸತ್ಯ,
        /* .host_buffer           = */ ಅಸತ್ಯ,
        /* .buffer_from_host_ptr  = */ ಸತ್ಯ,
        /* .events                = */ ಅಸತ್ಯ,
        /* .mmap_support          = */ ಸತ್ಯ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, ಧ್ರುವ ವರ್ಣ * params) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_size) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];

    ವಿಕಲ್ಪ (op->op) {
        ಪಕ್ಷ GGML_OP_NONE:
        ಪಕ್ಷ GGML_OP_RESHAPE:
        ಪಕ್ಷ GGML_OP_VIEW:
        ಪಕ್ಷ GGML_OP_PERMUTE:
        ಪಕ್ಷ GGML_OP_TRANSPOSE:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

        ಪಕ್ಷ GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
            ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];

            ಧ್ರುವ int64_t ne10 = src1->ne[0];

            ಧ್ರುವ int64_t ne0 = op->ne[0];
            ಧ್ರುವ int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            ಧ್ರುವ int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            ಯದಿ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != ರಿಕ್ತ);
        }

        ಪಕ್ಷ GGML_OP_OUT_PROD:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != ರಿಕ್ತ);

        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    }

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ ರಿಕ್ತ,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ ರಿಕ್ತ,
    /* .event_new            = */ ರಿಕ್ತ,
    /* .event_free           = */ ರಿಕ್ತ,
    /* .event_synchronize    = */ ರಿಕ್ತ,
};

// backend reg interface

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    ಪ್ರತ್ಯಾಗಮ "BLAS";

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ size_t ggml_backend_blas_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಪ್ರತ್ಯಾಗಮ 1;

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ಸ್ಥಿರ ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, ಧ್ರುವ ವರ್ಣ * name) {
    ಯದಿ (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_blas_set_n_threads;
    }
    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ ರಿಕ್ತ,
    };

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
