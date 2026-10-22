#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-blas.h"
#अन्तर्भाव "ggml-backend-impl.h"

#अन्तर्भाव <future>
#अन्तर्भाव <vector>
#अन्तर्भाव <cstring>

#यदि defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#अथवायदि defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#अथवायदि defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#अथवायदि defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#अन्यथा
#   include <cblas.h>
#endif

संरचना ggml_backend_blas_context {
    पूर्णाङ्क n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<वर्ण[]> work_data;
    माप_प्रकार work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<शून्य>> tasks;
#endif
};

स्थिर शून्य ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, संरचना ggml_tensor * dst) {
    अचर संरचना ggml_tensor * src0 = dst->src[0];
    अचर संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    अचर गणना ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    अचर int64_t r2 = ne12/ne02;
    अचर int64_t r3 = ne13/ne03;

    अचर int64_t ne_plane      = ne01*ne00;
    अचर माप_प्रकार  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*परिमाण(भिन्न);

    यदि (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(नूतन वर्ण[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    शून्य * wdata = ctx->work_data.get();

    // convert src0 to float
    यदि (type != GGML_TYPE_F32) {
        अचर स्वतः * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t अचर to_float = type_traits->to_float;

        लागि (int64_t i03 = 0; i03 < ne03; i03++) {
            लागि (int64_t i02 = 0; i02 < ne02; i02++) {
                अचर शून्य  *       x      = (वर्ण *)  src0->data + i02*nb02          + i03*nb03;
                      भिन्न * अचर wplane = (भिन्न *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                अचर पूर्णाङ्क min_cols_per_thread = 4096;
                अचर पूर्णाङ्क min_rows_per_thread = std::max((पूर्णाङ्क)(min_cols_per_thread/ne00), 1);
                अचर पूर्णाङ्क n_threads = std::max(std::min(ctx->n_threads, (पूर्णाङ्क)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel लागि अङ्क_threads(n_threads)
                लागि (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((अचर वर्ण *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#अन्यथा
                लागि (पूर्णाङ्क i = 1; i < n_threads; i++) {
                    अचर int64_t start =       i*ne01/n_threads;
                    अचर int64_t end   = (i + 1)*ne01/n_threads;
                    यदि (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            लागि (int64_t i01 = start; i01 < end; i01++) {
                                to_float((अचर वर्ण *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    अचर int64_t start = 0;
                    अचर int64_t end   = ne01/n_threads;
                    लागि (int64_t i01 = start; i01 < end; i01++) {
                        to_float((अचर वर्ण *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        लागि (स्वतः & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#यदि defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_अङ्क_threads(ctx->n_threads);
#अथवायदि defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_अङ्क_threads(ctx->n_threads);
#अथवायदि defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_अङ्क_threads(ctx->n_threads);
#अथवायदि defined(GGML_BLAS_USE_MKL)
    mkl_set_अङ्क_threads(ctx->n_threads);
#endif

    लागि (int64_t i13 = 0; i13 < ne13; i13++) {
        लागि (int64_t i12 = 0; i12 < ne12; i12++) {
            अचर int64_t i03 = i13/r3;
            अचर int64_t i02 = i12/r2;

            अचर भिन्न * x = (भिन्न *) ((वर्ण *) src0->data + i02*nb02 + i03*nb03);
            अचर भिन्न * y = (भिन्न *) ((वर्ण *) src1->data + i12*nb12 + i13*nb13);
                  भिन्न * d = (भिन्न *) ((वर्ण *)  dst->data + i12*nb2  + i13*nb3);

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

स्थिर शून्य ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, संरचना ggml_tensor * dst) {
    अचर संरचना ggml_tensor * src0 = dst->src[0];
    अचर संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

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

    भिन्न * a = (भिन्न *) ((वर्ण *) src1->data);
    भिन्न * b = (भिन्न *) ((वर्ण *) src0->data);
    भिन्न * c = (भिन्न *) ((वर्ण *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

स्थिर अचर वर्ण * ggml_backend_blas_get_name(ggml_backend_t backend) {
    फर्काऊ "BLAS";

    GGML_UNUSED(backend);
}

स्थिर शून्य ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

स्थिर गणना ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, संरचना ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    लागि (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * गाँठो = cgraph->nodes[i];

        यदि ((गाँठो->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            जारी;
        }

        विकल्प (गाँठो->op) {
            पक्ष GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, गाँठो);
                रोक;

            पक्ष GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, गाँठो);
                रोक;

            पक्ष GGML_OP_NONE:
            पक्ष GGML_OP_RESHAPE:
            पक्ष GGML_OP_VIEW:
            पक्ष GGML_OP_PERMUTE:
            पक्ष GGML_OP_TRANSPOSE:
                रोक;

            स्वभाव:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(गाँठो));
        }
    }

    फर्काऊ GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
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
    फर्काऊ &guid;
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
#endif

#यदि defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    फर्काऊ backend;
}

तार्किक ggml_backend_is_blas(ggml_backend_t backend) {
    फर्काऊ backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

शून्य ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, पूर्णाङ्क n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

स्थिर अचर वर्ण * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    फर्काऊ "BLAS";

    GGML_UNUSED(dev);
}

स्थिर अचर वर्ण * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #यदि defined(GGML_BLAS_USE_ACCELERATE)
        फर्काऊ "Accelerate";
    #अथवायदि defined(GGML_BLAS_USE_MKL)
        फर्काऊ "MKL";
    #अथवायदि defined(GGML_BLAS_USE_AOCL)
        फर्काऊ "AOCL-BLAS";
    #अथवायदि defined(GGML_BLAS_USE_BLIS)
        फर्काऊ "BLIS";
    #अथवायदि defined(GGML_BLAS_USE_NVPL)
        फर्काऊ "NVPL";
    #अथवायदि defined(GGML_BLAS_USE_OPENBLAS)
        फर्काऊ "OpenBLAS";
    #अन्यथा
        फर्काऊ "BLAS";
    #endif

    GGML_UNUSED(dev);
}

स्थिर शून्य ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * total) {
    // no memory to report
    *मुक्त  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    फर्काऊ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

स्थिर शून्य ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ असत्य,
        /* .host_buffer           = */ असत्य,
        /* .buffer_from_host_ptr  = */ सत्य,
        /* .events                = */ असत्य,
        /* .mmap_support          = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, अचर वर्ण * params) {
    फर्काऊ ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    फर्काऊ ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

स्थिर ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, शून्य * ptr, माप_प्रकार size, माप_प्रकार max_tensor_size) {
    फर्काऊ ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

स्थिर तार्किक ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, अचर संरचना ggml_tensor * op) {
    अचर संरचना ggml_tensor * src0 = op->src[0];
    अचर संरचना ggml_tensor * src1 = op->src[1];

    विकल्प (op->op) {
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_RESHAPE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
            फर्काऊ सत्य;

        पक्ष GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            अचर संरचना ggml_tensor * src0 = op->src[0];
            अचर संरचना ggml_tensor * src1 = op->src[1];

            अचर int64_t ne10 = src1->ne[0];

            अचर int64_t ne0 = op->ne[0];
            अचर int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            अचर int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            यदि (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                फर्काऊ असत्य;
            }

            फर्काऊ ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != रिक्त);
        }

        पक्ष GGML_OP_OUT_PROD:
            फर्काऊ op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != रिक्त);

        स्वभाव:
            फर्काऊ असत्य;

    }

    GGML_UNUSED(dev);
}

स्थिर तार्किक ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    फर्काऊ ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

स्थिर अचर संरचना ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ रिक्त,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ रिक्त,
    /* .event_new            = */ रिक्त,
    /* .event_free           = */ रिक्त,
    /* .event_synchronize    = */ रिक्त,
};

// backend reg interface

स्थिर अचर वर्ण * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    फर्काऊ "BLAS";

    GGML_UNUSED(reg);
}

स्थिर माप_प्रकार ggml_backend_blas_reg_get_device_गन्ती(ggml_backend_reg_t reg) {
    फर्काऊ 1;

    GGML_UNUSED(reg);
}

स्थिर ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार index) {
    GGML_ASSERT(index == 0);

    स्थिर ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    फर्काऊ &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

स्थिर शून्य * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, अचर वर्ण * name) {
    यदि (std::म_भेद(name, "ggml_backend_set_n_threads") == 0) {
        फर्काऊ (शून्य *)ggml_backend_blas_set_n_threads;
    }
    फर्काऊ रिक्त;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

स्थिर अचर संरचना ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_गन्ती,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(शून्य) {
    स्थिर संरचना ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ रिक्त,
    };

    फर्काऊ &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
