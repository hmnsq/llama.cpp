#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-blas.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"

#ଅନ୍ତର୍ଭାଵ <future>
#ଅନ୍ତର୍ଭାଵ <vector>
#ଅନ୍ତର୍ଭାଵ <cstring>

#ଯଦି defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#ନଚେତଯଦି defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#ନଚେତଯଦି defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#ନଚେତଯଦି defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#ଅନ୍ୟଥା
#   include <cblas.h>
#endif

ସଂରଚନା ggml_backend_blas_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<ଅକ୍ଷର[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<ଶୂନ୍ୟ>> tasks;
#endif
};

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, ସଂରଚନା ggml_tensor * dst) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ ଗଣନା ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ଅଚଳ int64_t r2 = ne12/ne02;
    ଅଚଳ int64_t r3 = ne13/ne03;

    ଅଚଳ int64_t ne_plane      = ne01*ne00;
    ଅଚଳ size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*ଆକାର(ଦଶମିକ);

    ଯଦି (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(ନୂତନ ଅକ୍ଷର[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    ଶୂନ୍ୟ * wdata = ctx->work_data.get();

    // convert src0 to float
    ଯଦି (type != GGML_TYPE_F32) {
        ଅଚଳ auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t ଅଚଳ to_float = type_traits->to_float;

        ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
            ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                ଅଚଳ ଶୂନ୍ୟ  *       x      = (ଅକ୍ଷର *)  src0->data + i02*nb02          + i03*nb03;
                      ଦଶମିକ * ଅଚଳ wplane = (ଦଶମିକ *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା min_cols_per_thread = 4096;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା min_rows_per_thread = std::max((ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(min_cols_per_thread/ne00), 1);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads = std::max(std::min(ctx->n_threads, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel ପାଇଁ ଅଙ୍କ_threads(n_threads)
                ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((ଅଚଳ ଅକ୍ଷର *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#ଅନ୍ୟଥା
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < n_threads; i++) {
                    ଅଚଳ int64_t start =       i*ne01/n_threads;
                    ଅଚଳ int64_t end   = (i + 1)*ne01/n_threads;
                    ଯଦି (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            ପାଇଁ (int64_t i01 = start; i01 < end; i01++) {
                                to_float((ଅଚଳ ଅକ୍ଷର *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    ଅଚଳ int64_t start = 0;
                    ଅଚଳ int64_t end   = ne01/n_threads;
                    ପାଇଁ (int64_t i01 = start; i01 < end; i01++) {
                        to_float((ଅଚଳ ଅକ୍ଷର *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        ପାଇଁ (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#ଯଦି defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_ଅଙ୍କ_threads(ctx->n_threads);
#ନଚେତଯଦି defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_ଅଙ୍କ_threads(ctx->n_threads);
#ନଚେତଯଦି defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_ଅଙ୍କ_threads(ctx->n_threads);
#ନଚେତଯଦି defined(GGML_BLAS_USE_MKL)
    mkl_set_ଅଙ୍କ_threads(ctx->n_threads);
#endif

    ପାଇଁ (int64_t i13 = 0; i13 < ne13; i13++) {
        ପାଇଁ (int64_t i12 = 0; i12 < ne12; i12++) {
            ଅଚଳ int64_t i03 = i13/r3;
            ଅଚଳ int64_t i02 = i12/r2;

            ଅଚଳ ଦଶମିକ * x = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i02*nb02 + i03*nb03);
            ଅଚଳ ଦଶମିକ * y = (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + i12*nb12 + i13*nb13);
                  ଦଶମିକ * d = (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + i12*nb2  + i13*nb3);

            ଯଦି (type != GGML_TYPE_F32) {
                x = (ଦଶମିକ *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, ସଂରଚନା ggml_tensor * dst) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
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

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = src0->ne[0];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = src0->ne[1];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା lda;

    ଯଦି (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } ଅନ୍ୟଥା {
        transposeA = CblasNoTrans;
        lda = k;
    }

    ଦଶମିକ * a = (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data);
    ଦଶମିକ * b = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data);
    ଦଶମିକ * c = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_blas_get_name(ggml_backend_t backend) {
    ଫେରନ୍ତୁ "BLAS";

    GGML_UNUSED(backend);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, ସଂରଚନା ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];

        ଯଦି ((ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ଜାରି;
        }

        ବାଛ (ଗଣ୍ଠି->op) {
            ସ୍ଥିତି GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, ଗଣ୍ଠି);
                ଭାଙ୍ଗ;

            ସ୍ଥିତି GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, ଗଣ୍ଠି);
                ଭାଙ୍ଗ;

            ସ୍ଥିତି GGML_OP_NONE:
            ସ୍ଥିତି GGML_OP_RESHAPE:
            ସ୍ଥିତି GGML_OP_VIEW:
            ସ୍ଥିତି GGML_OP_PERMUTE:
            ସ୍ଥିତି GGML_OP_TRANSPOSE:
                ଭାଙ୍ଗ;

            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(ଗଣ୍ଠି));
        }
    }

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

ସ୍ଥିର ସଂରଚନା ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ ଖାଲି,
    /* .get_tensor_async        = */ ଖାଲି,
    /* .set_tensor_2d_async     = */ ଖାଲି,
    /* .get_tensor_2d_async     = */ ଖାଲି,
    /* .cpy_tensor_async        = */ ଖାଲି,
    /* .synchronize             = */ ଖାଲି,
    /* .graph_plan_create       = */ ଖାଲି,
    /* .graph_plan_free         = */ ଖାଲି,
    /* .graph_plan_update       = */ ଖାଲି,
    /* .graph_plan_compute      = */ ଖାଲି,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ ଖାଲି,
    /* .event_wait              = */ ଖାଲି,
    /* .graph_optimize          = */ ଖାଲି,
};

ସ୍ଥିର ggml_guid_t ggml_backend_blas_guid(ଶୂନ୍ୟ) {
    ସ୍ଥିର ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    ଫେରନ୍ତୁ &guid;
}

ggml_backend_t ggml_backend_blas_init(ଶୂନ୍ୟ) {
    ggml_backend_blas_context * ctx = ନୂତନ ggml_backend_blas_context;

    ggml_backend_t backend = ନୂତନ ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#ଯଦି defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    ଯଦି (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#ଯଦି defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    ଫେରନ୍ତୁ backend;
}

ତାର୍କିକ ggml_backend_is_blas(ggml_backend_t backend) {
    ଫେରନ୍ତୁ backend != ଖାଲି && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

ଶୂନ୍ୟ ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ "BLAS";

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #ଯଦି defined(GGML_BLAS_USE_ACCELERATE)
        ଫେରନ୍ତୁ "Accelerate";
    #ନଚେତଯଦି defined(GGML_BLAS_USE_MKL)
        ଫେରନ୍ତୁ "MKL";
    #ନଚେତଯଦି defined(GGML_BLAS_USE_AOCL)
        ଫେରନ୍ତୁ "AOCL-BLAS";
    #ନଚେତଯଦି defined(GGML_BLAS_USE_BLIS)
        ଫେରନ୍ତୁ "BLIS";
    #ନଚେତଯଦି defined(GGML_BLAS_USE_NVPL)
        ଫେରନ୍ତୁ "NVPL";
    #ନଚେତଯଦି defined(GGML_BLAS_USE_OPENBLAS)
        ଫେରନ୍ତୁ "OpenBLAS";
    #ଅନ୍ୟଥା
        ଫେରନ୍ତୁ "BLAS";
    #endif

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * ମୁକ୍ତ, size_t * total) {
    // no memory to report
    *ମୁକ୍ତ  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଗଣନା ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, ସଂରଚନା ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ମିଥ୍ୟା,
        /* .host_buffer           = */ ମିଥ୍ୟା,
        /* .buffer_from_host_ptr  = */ ସତ୍ୟ,
        /* .events                = */ ମିଥ୍ୟା,
        /* .mmap_support          = */ ସତ୍ୟ,
    };
}

ସ୍ଥିର ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params) {
    ଫେରନ୍ତୁ ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ଶୂନ୍ୟ * ptr, size_t size, size_t max_tensor_size) {
    ଫେରନ୍ତୁ ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];

    ବାଛ (op->op) {
        ସ୍ଥିତି GGML_OP_NONE:
        ସ୍ଥିତି GGML_OP_RESHAPE:
        ସ୍ଥିତି GGML_OP_VIEW:
        ସ୍ଥିତି GGML_OP_PERMUTE:
        ସ୍ଥିତି GGML_OP_TRANSPOSE:
            ଫେରନ୍ତୁ ସତ୍ୟ;

        ସ୍ଥିତି GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
            ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];

            ଅଚଳ int64_t ne10 = src1->ne[0];

            ଅଚଳ int64_t ne0 = op->ne[0];
            ଅଚଳ int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            ଅଚଳ int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            ଯଦି (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଫେରନ୍ତୁ ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != ଖାଲି);
        }

        ସ୍ଥିତି GGML_OP_OUT_PROD:
            ଫେରନ୍ତୁ op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != ଖାଲି);

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;

    }

    GGML_UNUSED(dev);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ ଖାଲି,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ ଖାଲି,
    /* .event_new            = */ ଖାଲି,
    /* .event_free           = */ ଖାଲି,
    /* .event_synchronize    = */ ଖାଲି,
};

// backend reg interface

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ "BLAS";

    GGML_UNUSED(reg);
}

ସ୍ଥିର size_t ggml_backend_blas_reg_get_device_ଗଣନା(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ 1;

    GGML_UNUSED(reg);
}

ସ୍ଥିର ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ସ୍ଥିର ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    ଫେରନ୍ତୁ &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, ଅଚଳ ଅକ୍ଷର * name) {
    ଯଦି (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_blas_set_n_threads;
    }
    ଫେରନ୍ତୁ ଖାଲି;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_ଗଣନା,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(ଶୂନ୍ୟ) {
    ସ୍ଥିର ସଂରଚନା ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ ଖାଲି,
    };

    ଫେରନ୍ତୁ &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
