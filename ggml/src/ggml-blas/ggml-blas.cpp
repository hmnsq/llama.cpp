#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-blas.h"
#अन्तर्भाव "ggml-backend-impl.h"

#अन्तर्भाव <future>
#अन्तर्भाव <vector>
#अन्तर्भाव <cstring>

#यदि defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#नोचेद्यदि defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#नोचेद्यदि defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#नोचेद्यदि defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#अन्यथा
#   include <cblas.h>
#यदि_अन्तः

संरचना ggml_backend_blas_context {
    पूर्णाङ्क n_तन्तवः = GGML_DEFAULT_N_तन्तवः;
    std::unique_ptr<वर्ण[]> work_दत्त;
    size_t work_आकार = 0;
#यदि_अपरिभाषितम् GGML_USE_OPENMP
    std::vector<std::future<शून्य>> tasks;
#यदि_अन्तः
};

स्थिर शून्य ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, संरचना ggml_tensor * dst) {
    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव गणना ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(type));
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ध्रुव int64_t r2 = ne12/ne02;
    ध्रुव int64_t r3 = ne13/ne03;

    ध्रुव int64_t ne_plane      = ne01*ne00;
    ध्रुव size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*परिमाण(भिन्न);

    यदि (ctx->work_आकार < desired_wsize) {
        ctx->work_दत्त.reset(नूतन वर्ण[desired_wsize]);
        ctx->work_आकार = desired_wsize;
    }
    शून्य * wdata = ctx->work_दत्त.get();

    // convert src0 to float
    यदि (type != GGML_TYPE_F32) {
        ध्रुव स्वचालित * type_traits = ggml_get_type_traits(type);
        ggml_गन्तव्यम्_float_t ध्रुव गन्तव्यम्_float = type_traits->गन्तव्यम्_float;

        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                ध्रुव शून्य  *       x      = (वर्ण *)  src0->दत्त + i02*nb02          + i03*nb03;
                      भिन्न * ध्रुव wplane = (भिन्न *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                ध्रुव पूर्णाङ्क min_स्तम्भसङ्ख्या_प्रत्येकम्_thread = 4096;
                ध्रुव पूर्णाङ्क min_पङ्क्तिकाः_प्रत्येकम्_thread = std::max((पूर्णाङ्क)(min_स्तम्भसङ्ख्या_प्रत्येकम्_thread/ne00), 1);
                ध्रुव पूर्णाङ्क n_तन्तवः = std::max(std::min(ctx->n_तन्तवः, (पूर्णाङ्क)(ne01/min_पङ्क्तिकाः_प्रत्येकम्_thread)), 1);

#यदि_परिभाषितम् GGML_USE_OPENMP
                #निर्देशः omp parallel प्रति अङ्कः_तन्तवः(n_तन्तवः)
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    गन्तव्यम्_float((ध्रुव वर्ण *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#अन्यथा
                प्रति (पूर्णाङ्क i = 1; i < n_तन्तवः; i++) {
                    ध्रुव int64_t start =       i*ne01/n_तन्तवः;
                    ध्रुव int64_t end   = (i + 1)*ne01/n_तन्तवः;
                    यदि (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            प्रति (int64_t i01 = start; i01 < end; i01++) {
                                गन्तव्यम्_float((ध्रुव वर्ण *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    ध्रुव int64_t start = 0;
                    ध्रुव int64_t end   = ne01/n_तन्तवः;
                    प्रति (int64_t i01 = start; i01 < end; i01++) {
                        गन्तव्यम्_float((ध्रुव वर्ण *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#यदि_अन्तः
            }
        }

#यदि_अपरिभाषितम् GGML_USE_OPENMP
        // wait for all tasks to finish
        प्रति (स्वचालित & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#यदि_अन्तः
    }

#यदि defined(GGML_BLAS_USE_OPENBLAS)
    openblas_समुच्चयः_अङ्कः_तन्तवः(ctx->n_तन्तवः);
#नोचेद्यदि defined(GGML_BLAS_USE_BLIS)
    bli_thread_समुच्चयः_अङ्कः_तन्तवः(ctx->n_तन्तवः);
#नोचेद्यदि defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_समुच्चयः_अङ्कः_तन्तवः(ctx->n_तन्तवः);
#नोचेद्यदि defined(GGML_BLAS_USE_MKL)
    mkl_समुच्चयः_अङ्कः_तन्तवः(ctx->n_तन्तवः);
#यदि_अन्तः

    प्रति (int64_t i13 = 0; i13 < ne13; i13++) {
        प्रति (int64_t i12 = 0; i12 < ne12; i12++) {
            ध्रुव int64_t i03 = i13/r3;
            ध्रुव int64_t i02 = i12/r2;

            ध्रुव भिन्न * x = (भिन्न *) ((वर्ण *) src0->दत्त + i02*nb02 + i03*nb03);
            ध्रुव भिन्न * y = (भिन्न *) ((वर्ण *) src1->दत्त + i12*nb12 + i13*nb13);
                  भिन्न * d = (भिन्न *) ((वर्ण *)  dst->दत्त + i12*nb2  + i13*nb3);

            यदि (type != GGML_TYPE_F32) {
                x = (भिन्न *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

स्थिर शून्य ggml_backend_blas_बहिर्गमनम्_prod(ggml_backend_blas_context * ctx, संरचना ggml_tensor * dst) {
    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == परिमाण(भिन्न));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
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

    पूर्णाङ्क n = src0->ne[0];
    पूर्णाङ्क k = src0->ne[1];
    पूर्णाङ्क m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    पूर्णाङ्क lda;

    यदि (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } अन्यथा {
        transposeA = CblasNoTrans;
        lda = k;
    }

    भिन्न * a = (भिन्न *) ((वर्ण *) src1->दत्त);
    भिन्न * b = (भिन्न *) ((वर्ण *) src0->दत्त);
    भिन्न * c = (भिन्न *) ((वर्ण *) dst->दत्त);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_अप्रयुक्तम्(ctx);
}

// backend interface

स्थिर ध्रुव वर्ण * ggml_backend_blas_get_name(ggml_backend_t backend) {
    प्रत्यागम "BLAS";

    GGML_अप्रयुक्तम्(backend);
}

स्थिर शून्य ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    विलोपय ctx;
    विलोपय backend;
}

स्थिर गणना ggml_स्थितिः ggml_backend_blas_graph_compute(ggml_backend_t backend, संरचना ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[i];

        यदि ((ग्रन्थिः->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            अनुवर्त;
        }

        विकल्प (ग्रन्थिः->op) {
            पक्ष GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_बहिर्गमनम्_PROD:
                ggml_backend_blas_बहिर्गमनम्_prod(ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_NONE:
            पक्ष GGML_OP_RESHAPE:
            पक्ष GGML_OP_VIEW:
            पक्ष GGML_OP_PERMUTE:
            पक्ष GGML_OP_TRANSPOSE:
                भङ्ग;

            स्वभाव:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(ग्रन्थिः));
        }
    }

    प्रत्यागम GGML_स्थितिः_SUCCESS;

    GGML_अप्रयुक्तम्(backend);
}

स्थिर संरचना ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ रिक्त,
    /* .get_tensor_async        = */ रिक्त,
    /* .set_tensor_2d_async     = */ रिक्त,
    /* .get_tensor_2d_async     = */ रिक्त,
    /* .cpy_tensor_async        = */ रिक्त,
    /* .synchronize             = */ रिक्त,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ रिक्त,
    /* .event_wait              = */ रिक्त,
    /* .graph_optimize          = */ रिक्त,
};

स्थिर ggml_guid_t ggml_backend_blas_guid(शून्य) {
    स्थिर ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    प्रत्यागम &guid;
}

ggml_backend_t ggml_backend_blas_init(शून्य) {
    ggml_backend_blas_context * ctx = नूतन ggml_backend_blas_context;

    ggml_backend_t backend = नूतन ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#यदि defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    यदि (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#यदि_अन्तः

#यदि defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#यदि_अन्तः

    प्रत्यागम backend;
}

तार्किक ggml_backend_is_blas(ggml_backend_t backend) {
    प्रत्यागम backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

शून्य ggml_backend_blas_समुच्चयः_n_तन्तवः(ggml_backend_t backend_blas, पूर्णाङ्क n_तन्तवः) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_तन्तवः = n_तन्तवः;
}

// device interface

स्थिर ध्रुव वर्ण * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    प्रत्यागम "BLAS";

    GGML_अप्रयुक्तम्(dev);
}

स्थिर ध्रुव वर्ण * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #यदि defined(GGML_BLAS_USE_ACCELERATE)
        प्रत्यागम "Accelerate";
    #नोचेद्यदि defined(GGML_BLAS_USE_MKL)
        प्रत्यागम "MKL";
    #नोचेद्यदि defined(GGML_BLAS_USE_AOCL)
        प्रत्यागम "AOCL-BLAS";
    #नोचेद्यदि defined(GGML_BLAS_USE_BLIS)
        प्रत्यागम "BLIS";
    #नोचेद्यदि defined(GGML_BLAS_USE_NVPL)
        प्रत्यागम "NVPL";
    #नोचेद्यदि defined(GGML_BLAS_USE_OPENBLAS)
        प्रत्यागम "OpenBLAS";
    #अन्यथा
        प्रत्यागम "BLAS";
    #यदि_अन्तः

    GGML_अप्रयुक्तम्(dev);
}

स्थिर शून्य ggml_backend_blas_device_get_स्मृतिः(ggml_backend_dev_t dev, size_t * मुक्त, size_t * योग) {
    // no memory to report
    *मुक्त  = 0;
    *योग = 0;

    GGML_अप्रयुक्तम्(dev);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    प्रत्यागम GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_अप्रयुक्तम्(dev);
}

स्थिर शून्य ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_स्मृतिः(dev, &props->स्मृतिः_free, &props->स्मृतिः_योग);
    props->caps = {
        /* .async                 = */ असत्य,
        /* .host_buffer           = */ असत्य,
        /* .buffer_from_host_ptr  = */ सत्य,
        /* .events                = */ असत्य,
        /* .mmap_support          = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    प्रत्यागम ggml_backend_blas_init();

    GGML_अप्रयुक्तम्(dev);
    GGML_अप्रयुक्तम्(params);
}

स्थिर ggml_backend_बफर_type_t ggml_backend_blas_device_get_बफर_type(ggml_backend_dev_t dev) {
    प्रत्यागम ggml_backend_cpu_बफर_type();

    GGML_अप्रयुक्तम्(dev);
}

स्थिर ggml_backend_बफर_t ggml_backend_blas_device_बफर_from_host_ptr(ggml_backend_dev_t dev, शून्य * ptr, size_t size, size_t max_tensor_आकार) {
    प्रत्यागम ggml_backend_cpu_बफर_from_ptr(ptr, size);

    GGML_अप्रयुक्तम्(dev);
    GGML_अप्रयुक्तम्(max_tensor_आकार);
}

स्थिर तार्किक ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];

    विकल्प (op->op) {
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_RESHAPE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
            प्रत्यागम सत्य;

        पक्ष GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            ध्रुव संरचना ggml_tensor * src0 = op->src[0];
            ध्रुव संरचना ggml_tensor * src1 = op->src[1];

            ध्रुव int64_t ne10 = src1->ne[0];

            ध्रुव int64_t ne0 = op->ne[0];
            ध्रुव int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            ध्रुव int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            यदि (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                प्रत्यागम असत्य;
            }

            प्रत्यागम ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->गन्तव्यम्_float != रिक्त);
        }

        पक्ष GGML_OP_बहिर्गमनम्_PROD:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->गन्तव्यम्_float != रिक्त);

        स्वभाव:
            प्रत्यागम असत्य;

    }

    GGML_अप्रयुक्तम्(dev);
}

स्थिर तार्किक ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    प्रत्यागम ggml_backend_buft_is_host(buft);

    GGML_अप्रयुक्तम्(dev);
}

स्थिर ध्रुव संरचना ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_स्मृतिः,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_बफर_type,
    /* .get_host_buffer_type = */ रिक्त,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_बफर_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ रिक्त,
    /* .event_new            = */ रिक्त,
    /* .event_free           = */ रिक्त,
    /* .event_synchronize    = */ रिक्त,
};

// backend reg interface

स्थिर ध्रुव वर्ण * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    प्रत्यागम "BLAS";

    GGML_अप्रयुक्तम्(reg);
}

स्थिर size_t ggml_backend_blas_reg_get_device_गणनम्(ggml_backend_reg_t reg) {
    प्रत्यागम 1;

    GGML_अप्रयुक्तम्(reg);
}

स्थिर ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t सूचकांक) {
    GGML_ASSERT(सूचकांक == 0);

    स्थिर ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ शून्यसूचकः,
    };

    प्रत्यागम &ggml_backend_blas_device;

    GGML_अप्रयुक्तम्(reg);
    GGML_अप्रयुक्तम्(सूचकांक);
}

स्थिर शून्य * ggml_backend_blas_get_proc_स्थानाङ्कः(ggml_backend_reg_t reg, ध्रुव वर्ण * name) {
    यदि (std::वाक्यं_तुलय(name, "ggml_backend_set_n_threads") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_blas_समुच्चयः_n_तन्तवः;
    }
    प्रत्यागम रिक्त;

    GGML_अप्रयुक्तम्(reg);
    GGML_अप्रयुक्तम्(name);
}

स्थिर ध्रुव संरचना ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_गणनम्,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_स्थानाङ्कः,
};

ggml_backend_reg_t ggml_backend_blas_reg(शून्य) {
    स्थिर संरचना ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_संस्करणम्,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ रिक्त,
    };

    प्रत्यागम &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
