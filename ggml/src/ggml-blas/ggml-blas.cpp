#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-blas.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ <future>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <cstring>

#ಆದರೆ defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#ಇಲ್ಲದಿದ್ದರೆ
#   include <cblas.h>
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ ggml_backend_blas_context {
    ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು = GGML_DEFAULT_N_ಎಳೆಗಳು;
    std::unique_ptr<ಅಕ್ಷರ[]> work_ದತ್ತಾಂಶ;
    size_t work_ಗಾತ್ರ = 0;
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_USE_OPENMP
    std::vector<std::future<ಶೂನ್ಯ>> tasks;
#ಆದರೆ_ಅಂತ್ಯ
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, ರಚನೆ ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ಗಾತ್ರ(type));
    GGML_ASSERT(nb10 == ggml_type_ಗಾತ್ರ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ಸ್ಥಿರಾಂಕ int64_t r2 = ne12/ne02;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne13/ne03;

    ಸ್ಥಿರಾಂಕ int64_t ne_plane      = ne01*ne00;
    ಸ್ಥಿರಾಂಕ size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಆದರೆ (ctx->work_ಗಾತ್ರ < desired_wsize) {
        ctx->work_ದತ್ತಾಂಶ.reset(ಹೊಸ ಅಕ್ಷರ[desired_wsize]);
        ctx->work_ಗಾತ್ರ = desired_wsize;
    }
    ಶೂನ್ಯ * wdata = ctx->work_ದತ್ತಾಂಶ.get();

    // convert src0 to float
    ಆದರೆ (type != GGML_TYPE_F32) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * type_traits = ggml_get_type_traits(type);
        ggml_ಗಮ್ಯ_float_t ಸ್ಥಿರಾಂಕ ಗಮ್ಯ_float = type_traits->ಗಮ್ಯ_float;

        ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
            ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ  *       x      = (ಅಕ್ಷರ *)  src0->ದತ್ತಾಂಶ + i02*nb02          + i03*nb03;
                      ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ wplane = (ದಶಮಾಂಶ *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ min_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_thread = 4096;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ min_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = std::max((ಪೂರ್ಣಾಂಕ)(min_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_thread/ne00), 1);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು = std::max(std::min(ctx->n_ಎಳೆಗಳು, (ಪೂರ್ಣಾಂಕ)(ne01/min_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread)), 1);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENMP
                #ಸೂಚನೆ omp parallel ಗಾಗಿ ಅಂಕೆ_ಎಳೆಗಳು(n_ಎಳೆಗಳು)
                ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ಗಮ್ಯ_float((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#ಇಲ್ಲದಿದ್ದರೆ
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < n_ಎಳೆಗಳು; i++) {
                    ಸ್ಥಿರಾಂಕ int64_t start =       i*ne01/n_ಎಳೆಗಳು;
                    ಸ್ಥಿರಾಂಕ int64_t end   = (i + 1)*ne01/n_ಎಳೆಗಳು;
                    ಆದರೆ (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            ಗಾಗಿ (int64_t i01 = start; i01 < end; i01++) {
                                ಗಮ್ಯ_float((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    ಸ್ಥಿರಾಂಕ int64_t start = 0;
                    ಸ್ಥಿರಾಂಕ int64_t end   = ne01/n_ಎಳೆಗಳು;
                    ಗಾಗಿ (int64_t i01 = start; i01 < end; i01++) {
                        ಗಮ್ಯ_float((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_USE_OPENMP
        // wait for all tasks to finish
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#ಆದರೆ_ಅಂತ್ಯ
    }

#ಆದರೆ defined(GGML_BLAS_USE_OPENBLAS)
    openblas_ಗಣ_ಅಂಕೆ_ಎಳೆಗಳು(ctx->n_ಎಳೆಗಳು);
#ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_BLIS)
    bli_thread_ಗಣ_ಅಂಕೆ_ಎಳೆಗಳು(ctx->n_ಎಳೆಗಳು);
#ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_ಗಣ_ಅಂಕೆ_ಎಳೆಗಳು(ctx->n_ಎಳೆಗಳು);
#ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_MKL)
    mkl_ಗಣ_ಅಂಕೆ_ಎಳೆಗಳು(ctx->n_ಎಳೆಗಳು);
#ಆದರೆ_ಅಂತ್ಯ

    ಗಾಗಿ (int64_t i13 = 0; i13 < ne13; i13++) {
        ಗಾಗಿ (int64_t i12 = 0; i12 < ne12; i12++) {
            ಸ್ಥಿರಾಂಕ int64_t i03 = i13/r3;
            ಸ್ಥಿರಾಂಕ int64_t i02 = i12/r2;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i02*nb02 + i03*nb03);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i12*nb12 + i13*nb13);
                  ದಶಮಾಂಶ * d = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i12*nb2  + i13*nb3);

            ಆದರೆ (type != GGML_TYPE_F32) {
                x = (ದಶಮಾಂಶ *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(Cblas_ಅಡ್ಡಸಾಲು_Major, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_blas_ಹೊರಹೋಗು_prod(ggml_backend_blas_context * ctx, ರಚನೆ ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
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

    ಪೂರ್ಣಾಂಕ n = src0->ne[0];
    ಪೂರ್ಣಾಂಕ k = src0->ne[1];
    ಪೂರ್ಣಾಂಕ m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    ಪೂರ್ಣಾಂಕ lda;

    ಆದರೆ (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } ಇಲ್ಲದಿದ್ದರೆ {
        transposeA = CblasNoTrans;
        lda = k;
    }

    ದಶಮಾಂಶ * a = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ * b = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ * c = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ);

    cblas_sgemm(Cblas_ಅಡ್ಡಸಾಲು_Major, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_ಬಳಸದ(ctx);
}

// backend interface

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_blas_get_name(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ "BLAS";

    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    ಅಳಿಸು ctx;
    ಅಳಿಸು backend;
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_blas_ನಕ್ಷೆ_compute(ggml_backend_t backend, ರಚನೆ ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        ಆದರೆ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಮುಂದುವರಿ;
        }

        ಆಯ್ಕೆ (ಗಂಟು->op) {
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
                ggml_backend_blas_ಹೊರಹೋಗು_prod(ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_NONE:
            ಸ್ಥಿತಿ GGML_OP_RESHAPE:
            ಸ್ಥಿತಿ GGML_OP_VIEW:
            ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
            ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
                ನಿಲ್ಲಿಸು;

            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(ಗಂಟು));
        }
    }

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;

    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ ರಚನೆ ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ ಖಾಲಿ,
    /* .get_tensor_async        = */ ಖಾಲಿ,
    /* .set_tensor_2d_async     = */ ಖಾಲಿ,
    /* .get_tensor_2d_async     = */ ಖಾಲಿ,
    /* .cpy_tensor_async        = */ ಖಾಲಿ,
    /* .synchronize             = */ ಖಾಲಿ,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_blas_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ಖಾಲಿ,
    /* .event_wait              = */ ಖಾಲಿ,
    /* .graph_optimize          = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_blas_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    ಹಿಂತಿರುಗಿ &guid;
}

ggml_backend_t ggml_backend_blas_init(ಶೂನ್ಯ) {
    ggml_backend_blas_context * ctx = ಹೊಸ ggml_backend_blas_context;

    ggml_backend_t backend = ಹೊಸ ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#ಆದರೆ defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    ಆದರೆ (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_blas(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಖಾಲಿ && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

ಶೂನ್ಯ ggml_backend_blas_ಗಣ_n_ಎಳೆಗಳು(ggml_backend_t backend_blas, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
}

// device interface

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ "BLAS";

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #ಆದರೆ defined(GGML_BLAS_USE_ACCELERATE)
        ಹಿಂತಿರುಗಿ "Accelerate";
    #ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_MKL)
        ಹಿಂತಿರುಗಿ "MKL";
    #ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_AOCL)
        ಹಿಂತಿರುಗಿ "AOCL-BLAS";
    #ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_BLIS)
        ಹಿಂತಿರುಗಿ "BLIS";
    #ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_NVPL)
        ಹಿಂತಿರುಗಿ "NVPL";
    #ಅಥವಾಆದರೆ defined(GGML_BLAS_USE_OPENBLAS)
        ಹಿಂತಿರುಗಿ "OpenBLAS";
    #ಇಲ್ಲದಿದ್ದರೆ
        ಹಿಂತಿರುಗಿ "BLAS";
    #ಆದರೆ_ಅಂತ್ಯ

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_blas_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    // no memory to report
    *ಬಿಡುಗಡೆ  = 0;
    *ಒಟ್ಟು = 0;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, ರಚನೆ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);
    props->caps = {
        /* .async                 = */ ಸುಳ್ಳು,
        /* .host_buffer           = */ ಸುಳ್ಳು,
        /* .buffer_from_host_ptr  = */ ನಿಜ,
        /* .events                = */ ಸುಳ್ಳು,
        /* .mmap_support          = */ ನಿಜ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ಹಿಂತಿರುಗಿ ggml_backend_blas_init();

    GGML_ಬಳಸದ(dev);
    GGML_ಬಳಸದ(params);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_blas_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ ggml_backend_cpu_ಬಫರ್_type();

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_blas_device_ಬಫರ್_from_host_ptr(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ ggml_backend_cpu_ಬಫರ್_from_ptr(ptr, size);

    GGML_ಬಳಸದ(dev);
    GGML_ಬಳಸದ(max_tensor_ಗಾತ್ರ);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];

    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
            ಹಿಂತಿರುಗಿ ನಿಜ;

        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
            ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];

            ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0];

            ಸ್ಥಿರಾಂಕ int64_t ne0 = op->ne[0];
            ಸ್ಥಿರಾಂಕ int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            ಸ್ಥಿರಾಂಕ int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            ಆದರೆ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಹಿಂತಿರುಗಿ ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->ಗಮ್ಯ_float != ಖಾಲಿ);
        }

        ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_ಮಾತೃಕೆ(src0) &&
                   ggml_is_ಮಾತೃಕೆ(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->ಗಮ್ಯ_float != ಖಾಲಿ);

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    }

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_is_host(buft);

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_ಸ್ಮರಣೆ,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ಖಾಲಿ,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_ಬಫರ್_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ ಖಾಲಿ,
    /* .event_new            = */ ಖಾಲಿ,
    /* .event_free           = */ ಖಾಲಿ,
    /* .event_synchronize    = */ ಖಾಲಿ,
};

// backend reg interface

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ "BLAS";

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ size_t ggml_backend_blas_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ 1;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ಸೂಚ್ಯಂಕ == 0);

    ಸ್ಥಿರ ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ ಶೂನ್ಯಸೂಚಕ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_blas_device;

    GGML_ಬಳಸದ(reg);
    GGML_ಬಳಸದ(ಸೂಚ್ಯಂಕ);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_blas_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಆದರೆ (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_blas_ಗಣ_n_ಎಳೆಗಳು;
    }
    ಹಿಂತಿರುಗಿ ಖಾಲಿ;

    GGML_ಬಳಸದ(reg);
    GGML_ಬಳಸದ(name);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_ವಿಳಾಸ,
};

ggml_backend_reg_t ggml_backend_blas_reg(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ರಚನೆ ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ ಖಾಲಿ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
