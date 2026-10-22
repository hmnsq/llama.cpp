#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-blas.h"
#अन्तर्भाव "ggml-backend-impl.h"

#अन्तर्भाव <future>
#अन्तर्भाव <vector>
#अन्तर्भाव <cstring>

#जर defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#किंवाजर defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#किंवाजर defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#किंवाजर defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#अन्यथा
#   include <cblas.h>
#endif

संरचना ggml_backend_blas_context {
    पूर्णांक n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<अक्षर[]> work_data;
    माप_प्रकार work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<शून्य>> tasks;
#endif
};

स्थिर शून्य ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, संरचना ggml_tensor * dst) {
    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक गणना ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == आकार(दशांश));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    स्थिरांक int64_t r2 = ne12/ne02;
    स्थिरांक int64_t r3 = ne13/ne03;

    स्थिरांक int64_t ne_plane      = ne01*ne00;
    स्थिरांक माप_प्रकार  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*आकार(दशांश);

    जर (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(नवीन अक्षर[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    शून्य * wdata = ctx->work_data.get();

    // convert src0 to float
    जर (type != GGML_TYPE_F32) {
        स्थिरांक स्वतः * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t स्थिरांक to_float = type_traits->to_float;

        साठी (int64_t i03 = 0; i03 < ne03; i03++) {
            साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                स्थिरांक शून्य  *       x      = (अक्षर *)  src0->data + i02*nb02          + i03*nb03;
                      दशांश * स्थिरांक wplane = (दशांश *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                स्थिरांक पूर्णांक min_cols_per_thread = 4096;
                स्थिरांक पूर्णांक min_rows_per_thread = std::max((पूर्णांक)(min_cols_per_thread/ne00), 1);
                स्थिरांक पूर्णांक n_threads = std::max(std::min(ctx->n_threads, (पूर्णांक)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel साठी अंक_threads(n_threads)
                साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((स्थिरांक अक्षर *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#अन्यथा
                साठी (पूर्णांक i = 1; i < n_threads; i++) {
                    स्थिरांक int64_t start =       i*ne01/n_threads;
                    स्थिरांक int64_t end   = (i + 1)*ne01/n_threads;
                    जर (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            साठी (int64_t i01 = start; i01 < end; i01++) {
                                to_float((स्थिरांक अक्षर *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    स्थिरांक int64_t start = 0;
                    स्थिरांक int64_t end   = ne01/n_threads;
                    साठी (int64_t i01 = start; i01 < end; i01++) {
                        to_float((स्थिरांक अक्षर *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        साठी (स्वतः & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#जर defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_अंक_threads(ctx->n_threads);
#किंवाजर defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_अंक_threads(ctx->n_threads);
#किंवाजर defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_अंक_threads(ctx->n_threads);
#किंवाजर defined(GGML_BLAS_USE_MKL)
    mkl_set_अंक_threads(ctx->n_threads);
#endif

    साठी (int64_t i13 = 0; i13 < ne13; i13++) {
        साठी (int64_t i12 = 0; i12 < ne12; i12++) {
            स्थिरांक int64_t i03 = i13/r3;
            स्थिरांक int64_t i02 = i12/r2;

            स्थिरांक दशांश * x = (दशांश *) ((अक्षर *) src0->data + i02*nb02 + i03*nb03);
            स्थिरांक दशांश * y = (दशांश *) ((अक्षर *) src1->data + i12*nb12 + i13*nb13);
                  दशांश * d = (दशांश *) ((अक्षर *)  dst->data + i12*nb2  + i13*nb3);

            जर (type != GGML_TYPE_F32) {
                x = (दशांश *) wdata + i02*ne_plane + i03*ne02*ne_plane;
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
    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == आकार(दशांश));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == आकार(दशांश));
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

    पूर्णांक n = src0->ne[0];
    पूर्णांक k = src0->ne[1];
    पूर्णांक m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    पूर्णांक lda;

    जर (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } अन्यथा {
        transposeA = CblasNoTrans;
        lda = k;
    }

    दशांश * a = (दशांश *) ((अक्षर *) src1->data);
    दशांश * b = (दशांश *) ((अक्षर *) src0->data);
    दशांश * c = (दशांश *) ((अक्षर *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

स्थिर स्थिरांक अक्षर * ggml_backend_blas_get_name(ggml_backend_t backend) {
    परत "BLAS";

    GGML_UNUSED(backend);
}

स्थिर शून्य ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

स्थिर गणना ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, संरचना ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = cgraph->nodes[i];

        जर ((गाठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            सुरूठेवा;
        }

        निवड (गाठ->op) {
            स्थिती GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, गाठ);
                थांबा;

            स्थिती GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, गाठ);
                थांबा;

            स्थिती GGML_OP_NONE:
            स्थिती GGML_OP_RESHAPE:
            स्थिती GGML_OP_VIEW:
            स्थिती GGML_OP_PERMUTE:
            स्थिती GGML_OP_TRANSPOSE:
                थांबा;

            पूर्वनिर्धारित:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(गाठ));
        }
    }

    परत GGML_STATUS_SUCCESS;

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
    परत &guid;
}

ggml_backend_t ggml_backend_blas_init(शून्य) {
    ggml_backend_blas_context * ctx = नवीन ggml_backend_blas_context;

    ggml_backend_t backend = नवीन ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#जर defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    जर (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#जर defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    परत backend;
}

तार्किक ggml_backend_is_blas(ggml_backend_t backend) {
    परत backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

शून्य ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, पूर्णांक n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

स्थिर स्थिरांक अक्षर * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    परत "BLAS";

    GGML_UNUSED(dev);
}

स्थिर स्थिरांक अक्षर * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #जर defined(GGML_BLAS_USE_ACCELERATE)
        परत "Accelerate";
    #किंवाजर defined(GGML_BLAS_USE_MKL)
        परत "MKL";
    #किंवाजर defined(GGML_BLAS_USE_AOCL)
        परत "AOCL-BLAS";
    #किंवाजर defined(GGML_BLAS_USE_BLIS)
        परत "BLIS";
    #किंवाजर defined(GGML_BLAS_USE_NVPL)
        परत "NVPL";
    #किंवाजर defined(GGML_BLAS_USE_OPENBLAS)
        परत "OpenBLAS";
    #अन्यथा
        परत "BLAS";
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
    परत GGML_BACKEND_DEVICE_TYPE_ACCEL;

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

स्थिर ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, स्थिरांक अक्षर * params) {
    परत ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    परत ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

स्थिर ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, शून्य * ptr, माप_प्रकार size, माप_प्रकार max_tensor_size) {
    परत ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

स्थिर तार्किक ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];

    निवड (op->op) {
        स्थिती GGML_OP_NONE:
        स्थिती GGML_OP_RESHAPE:
        स्थिती GGML_OP_VIEW:
        स्थिती GGML_OP_PERMUTE:
        स्थिती GGML_OP_TRANSPOSE:
            परत सत्य;

        स्थिती GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
            स्थिरांक संरचना ggml_tensor * src1 = op->src[1];

            स्थिरांक int64_t ne10 = src1->ne[0];

            स्थिरांक int64_t ne0 = op->ne[0];
            स्थिरांक int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            स्थिरांक int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            जर (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                परत असत्य;
            }

            परत ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != रिक्त);
        }

        स्थिती GGML_OP_OUT_PROD:
            परत op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != रिक्त);

        पूर्वनिर्धारित:
            परत असत्य;

    }

    GGML_UNUSED(dev);
}

स्थिर तार्किक ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    परत ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

स्थिर स्थिरांक संरचना ggml_backend_device_i ggml_backend_blas_device_i = {
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

स्थिर स्थिरांक अक्षर * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    परत "BLAS";

    GGML_UNUSED(reg);
}

स्थिर माप_प्रकार ggml_backend_blas_reg_get_device_मोजणी(ggml_backend_reg_t reg) {
    परत 1;

    GGML_UNUSED(reg);
}

स्थिर ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार index) {
    GGML_ASSERT(index == 0);

    स्थिर ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    परत &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

स्थिर शून्य * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, स्थिरांक अक्षर * name) {
    जर (std::म_भेद(name, "ggml_backend_set_n_threads") == 0) {
        परत (शून्य *)ggml_backend_blas_set_n_threads;
    }
    परत रिक्त;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

स्थिर स्थिरांक संरचना ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_मोजणी,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(शून्य) {
    स्थिर संरचना ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ रिक्त,
    };

    परत &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
