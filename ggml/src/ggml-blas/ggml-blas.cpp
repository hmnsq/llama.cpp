#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-blas.h"
#అన్తర్భావ "ggml-backend-impl.h"

#అన్తర్భావ <future>
#అన్తర్భావ <vector>
#అన్తర్భావ <cstring>

#ఒకవేళ defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#లేదావేళ defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#లేదావేళ defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#లేదావేళ defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#లేకపోతే
#   include <cblas.h>
#అయితే_ముగింపు

నిర్మాణం ggml_backend_blas_context {
    పూర్ణాంకం n_దారాలు = GGML_DEFAULT_N_దారాలు;
    std::unique_ptr<అక్షరం[]> work_దత్తాంశం;
    size_t work_కొలత = 0;
#నిర్వచితం_కాకపోతే GGML_USE_OPENMP
    std::vector<std::future<శూన్య>> tasks;
#అయితే_ముగింపు
};

స్థిర శూన్య ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, నిర్మాణం ggml_tensor * dst) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి గణన ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_కొలత(type));
    GGML_ASSERT(nb10 == ggml_type_కొలత(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    స్థిరరాశి int64_t r2 = ne12/ne02;
    స్థిరరాశి int64_t r3 = ne13/ne03;

    స్థిరరాశి int64_t ne_plane      = ne01*ne00;
    స్థిరరాశి size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*పరిమాణం(దశాంశం);

    ఒకవేళ (ctx->work_కొలత < desired_wsize) {
        ctx->work_దత్తాంశం.reset(కొత్త అక్షరం[desired_wsize]);
        ctx->work_కొలత = desired_wsize;
    }
    శూన్య * wdata = ctx->work_దత్తాంశం.get();

    // convert src0 to float
    ఒకవేళ (type != GGML_TYPE_F32) {
        స్థిరరాశి స్వయంచాలక * type_traits = ggml_get_type_traits(type);
        ggml_గమ్యం_float_t స్థిరరాశి గమ్యం_float = type_traits->గమ్యం_float;

        కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
            కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                స్థిరరాశి శూన్య  *       x      = (అక్షరం *)  src0->దత్తాంశం + i02*nb02          + i03*nb03;
                      దశాంశం * స్థిరరాశి wplane = (దశాంశం *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                స్థిరరాశి పూర్ణాంకం min_నిలువుల_సంఖ్య_చొప్పున_thread = 4096;
                స్థిరరాశి పూర్ణాంకం min_అడ్డువరుసలు_చొప్పున_thread = std::max((పూర్ణాంకం)(min_నిలువుల_సంఖ్య_చొప్పున_thread/ne00), 1);
                స్థిరరాశి పూర్ణాంకం n_దారాలు = std::max(std::min(ctx->n_దారాలు, (పూర్ణాంకం)(ne01/min_అడ్డువరుసలు_చొప్పున_thread)), 1);

#నిర్వచితమైతే GGML_USE_OPENMP
                #సూచన omp parallel కొరకు అంకె_దారాలు(n_దారాలు)
                కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                    గమ్యం_float((స్థిరరాశి అక్షరం *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#లేకపోతే
                కొరకు (పూర్ణాంకం i = 1; i < n_దారాలు; i++) {
                    స్థిరరాశి int64_t start =       i*ne01/n_దారాలు;
                    స్థిరరాశి int64_t end   = (i + 1)*ne01/n_దారాలు;
                    ఒకవేళ (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            కొరకు (int64_t i01 = start; i01 < end; i01++) {
                                గమ్యం_float((స్థిరరాశి అక్షరం *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    స్థిరరాశి int64_t start = 0;
                    స్థిరరాశి int64_t end   = ne01/n_దారాలు;
                    కొరకు (int64_t i01 = start; i01 < end; i01++) {
                        గమ్యం_float((స్థిరరాశి అక్షరం *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#అయితే_ముగింపు
            }
        }

#నిర్వచితం_కాకపోతే GGML_USE_OPENMP
        // wait for all tasks to finish
        కొరకు (స్వయంచాలక & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#అయితే_ముగింపు
    }

#ఒకవేళ defined(GGML_BLAS_USE_OPENBLAS)
    openblas_సమితి_అంకె_దారాలు(ctx->n_దారాలు);
#లేదావేళ defined(GGML_BLAS_USE_BLIS)
    bli_thread_సమితి_అంకె_దారాలు(ctx->n_దారాలు);
#లేదావేళ defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_సమితి_అంకె_దారాలు(ctx->n_దారాలు);
#లేదావేళ defined(GGML_BLAS_USE_MKL)
    mkl_సమితి_అంకె_దారాలు(ctx->n_దారాలు);
#అయితే_ముగింపు

    కొరకు (int64_t i13 = 0; i13 < ne13; i13++) {
        కొరకు (int64_t i12 = 0; i12 < ne12; i12++) {
            స్థిరరాశి int64_t i03 = i13/r3;
            స్థిరరాశి int64_t i02 = i12/r2;

            స్థిరరాశి దశాంశం * x = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i02*nb02 + i03*nb03);
            స్థిరరాశి దశాంశం * y = (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + i12*nb12 + i13*nb13);
                  దశాంశం * d = (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + i12*nb2  + i13*nb3);

            ఒకవేళ (type != GGML_TYPE_F32) {
                x = (దశాంశం *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(Cblas_అడ్డువరుస_Major, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

స్థిర శూన్య ggml_backend_blas_బహిర్గతం_prod(ggml_backend_blas_context * ctx, నిర్మాణం ggml_tensor * dst) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
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

    పూర్ణాంకం n = src0->ne[0];
    పూర్ణాంకం k = src0->ne[1];
    పూర్ణాంకం m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    పూర్ణాంకం lda;

    ఒకవేళ (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } లేకపోతే {
        transposeA = CblasNoTrans;
        lda = k;
    }

    దశాంశం * a = (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం);
    దశాంశం * b = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం);
    దశాంశం * c = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం);

    cblas_sgemm(Cblas_అడ్డువరుస_Major, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_వాడని(ctx);
}

// backend interface

స్థిర స్థిరరాశి అక్షరం * ggml_backend_blas_get_name(ggml_backend_t backend) {
    తిరిగి "BLAS";

    GGML_వాడని(backend);
}

స్థిర శూన్య ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    తొలగించు ctx;
    తొలగించు backend;
}

స్థిర గణన ggml_స్థితి ggml_backend_blas_లేఖాచిత్రం_compute(ggml_backend_t backend, నిర్మాణం ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[i];

        ఒకవేళ ((కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            కొనసాగు;
        }

        ఎంపిక (కణుపు->op) {
            స్థితి GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_బహిర్గతం_PROD:
                ggml_backend_blas_బహిర్గతం_prod(ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_NONE:
            స్థితి GGML_OP_RESHAPE:
            స్థితి GGML_OP_VIEW:
            స్థితి GGML_OP_క్రమమార్పు:
            స్థితి GGML_OP_TRANSPOSE:
                ఆపు;

            అప్రమేయ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(కణుపు));
        }
    }

    తిరిగి GGML_స్థితి_SUCCESS;

    GGML_వాడని(backend);
}

స్థిర నిర్మాణం ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ ఖాళీ,
    /* .get_tensor_async        = */ ఖాళీ,
    /* .set_tensor_2d_async     = */ ఖాళీ,
    /* .get_tensor_2d_async     = */ ఖాళీ,
    /* .cpy_tensor_async        = */ ఖాళీ,
    /* .synchronize             = */ ఖాళీ,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_blas_లేఖాచిత్రం_compute,
    /* .event_record            = */ ఖాళీ,
    /* .event_wait              = */ ఖాళీ,
    /* .graph_optimize          = */ ఖాళీ,
};

స్థిర ggml_guid_t ggml_backend_blas_guid(శూన్య) {
    స్థిర ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    తిరిగి &guid;
}

ggml_backend_t ggml_backend_blas_init(శూన్య) {
    ggml_backend_blas_context * ctx = కొత్త ggml_backend_blas_context;

    ggml_backend_t backend = కొత్త ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#ఒకవేళ defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    ఒకవేళ (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#అయితే_ముగింపు

#ఒకవేళ defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#అయితే_ముగింపు

    తిరిగి backend;
}

తార్కిక ggml_backend_is_blas(ggml_backend_t backend) {
    తిరిగి backend != ఖాళీ && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

శూన్య ggml_backend_blas_సమితి_n_దారాలు(ggml_backend_t backend_blas, పూర్ణాంకం n_దారాలు) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_దారాలు = n_దారాలు;
}

// device interface

స్థిర స్థిరరాశి అక్షరం * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    తిరిగి "BLAS";

    GGML_వాడని(dev);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #ఒకవేళ defined(GGML_BLAS_USE_ACCELERATE)
        తిరిగి "Accelerate";
    #లేదావేళ defined(GGML_BLAS_USE_MKL)
        తిరిగి "MKL";
    #లేదావేళ defined(GGML_BLAS_USE_AOCL)
        తిరిగి "AOCL-BLAS";
    #లేదావేళ defined(GGML_BLAS_USE_BLIS)
        తిరిగి "BLIS";
    #లేదావేళ defined(GGML_BLAS_USE_NVPL)
        తిరిగి "NVPL";
    #లేదావేళ defined(GGML_BLAS_USE_OPENBLAS)
        తిరిగి "OpenBLAS";
    #లేకపోతే
        తిరిగి "BLAS";
    #అయితే_ముగింపు

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_blas_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    // no memory to report
    *విడుదల  = 0;
    *మొత్తం = 0;

    GGML_వాడని(dev);
}

స్థిర గణన ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    తిరిగి GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, నిర్మాణం ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);
    props->caps = {
        /* .async                 = */ అబద్ధం,
        /* .host_buffer           = */ అబద్ధం,
        /* .buffer_from_host_ptr  = */ నిజం,
        /* .events                = */ అబద్ధం,
        /* .mmap_support          = */ నిజం,
    };
}

స్థిర ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    తిరిగి ggml_backend_blas_init();

    GGML_వాడని(dev);
    GGML_వాడని(params);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_blas_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    తిరిగి ggml_backend_cpu_బఫరు_type();

    GGML_వాడని(dev);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_blas_device_బఫరు_from_host_ptr(ggml_backend_dev_t dev, శూన్య * ptr, size_t size, size_t max_tensor_కొలత) {
    తిరిగి ggml_backend_cpu_బఫరు_from_ptr(ptr, size);

    GGML_వాడని(dev);
    GGML_వాడని(max_tensor_కొలత);
}

స్థిర తార్కిక ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];

    ఎంపిక (op->op) {
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
            తిరిగి నిజం;

        స్థితి GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
            స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];

            స్థిరరాశి int64_t ne10 = src1->ne[0];

            స్థిరరాశి int64_t ne0 = op->ne[0];
            స్థిరరాశి int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            స్థిరరాశి int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            ఒకవేళ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                తిరిగి అబద్ధం;
            }

            తిరిగి ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->గమ్యం_float != ఖాళీ);
        }

        స్థితి GGML_OP_బహిర్గతం_PROD:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_మాతృక(src0) &&
                   ggml_is_మాతృక(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->గమ్యం_float != ఖాళీ);

        అప్రమేయ:
            తిరిగి అబద్ధం;

    }

    GGML_వాడని(dev);
}

స్థిర తార్కిక ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    తిరిగి ggml_backend_buft_is_host(buft);

    GGML_వాడని(dev);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_జ్ఞాపకం,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ఖాళీ,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_బఫరు_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ ఖాళీ,
    /* .event_new            = */ ఖాళీ,
    /* .event_free           = */ ఖాళీ,
    /* .event_synchronize    = */ ఖాళీ,
};

// backend reg interface

స్థిర స్థిరరాశి అక్షరం * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    తిరిగి "BLAS";

    GGML_వాడని(reg);
}

స్థిర size_t ggml_backend_blas_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    తిరిగి 1;

    GGML_వాడని(reg);
}

స్థిర ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t సూచీ) {
    GGML_ASSERT(సూచీ == 0);

    స్థిర ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ శూన్యసూచిక,
    };

    తిరిగి &ggml_backend_blas_device;

    GGML_వాడని(reg);
    GGML_వాడని(సూచీ);
}

స్థిర శూన్య * ggml_backend_blas_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name) {
    ఒకవేళ (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        తిరిగి (శూన్య *)ggml_backend_blas_సమితి_n_దారాలు;
    }
    తిరిగి ఖాళీ;

    GGML_వాడని(reg);
    GGML_వాడని(name);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_లెక్క,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_చిరునామా,
};

ggml_backend_reg_t ggml_backend_blas_reg(శూన్య) {
    స్థిర నిర్మాణం ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ ఖాళీ,
    };

    తిరిగి &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
