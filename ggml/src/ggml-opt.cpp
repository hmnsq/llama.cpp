#ضمن "ggml-opt.h"

#ضمن "ggml.h"
#ضمن "ggml-alloc.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"

#ضمن <algorithm>
#ضمن <cmath>
#ضمن <cstdint>
#ضمن <cinttypes>
#ضمن <map>
#ضمن <random>
#ضمن <vector>

بنية ggml_opt_dataset {
    بنية ggml_context   * ctx    = تش_اشاره;
    ggml_backend_بفر_t   buf    = تش_اشاره;
    بنية ggml_tensor    * معلومات   = تش_اشاره;
    بنية ggml_tensor    * labels = تش_اشاره;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_معلومات    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

بنية ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = تش_اشاره;
    ggml_cgraph              * allocated_ګراف      = تش_اشاره;
    ggml_cgraph              * allocated_ګراف_copy = تش_اشاره;
    بنية ggml_context      * ctx_static           = تش_اشاره;
    بنية ggml_context      * ctx_cpu              = تش_اشاره;
    بنية ggml_context      * ctx_compute          = تش_اشاره;
    بنية ggml_context      * ctx_copy             = تش_اشاره;
    ggml_backend_بفر_t      buf_static           = تش_اشاره;
    ggml_backend_بفر_t      buf_cpu              = تش_اشاره;
    std::mt19937               rng;
    تعداد ggml_opt_loss_type    loss_type;
    تعداد ggml_opt_build_type   build_type;
    تعداد ggml_opt_build_type   build_type_alloc;

    بنية ggml_tensor * inputs  = تش_اشاره;
    بنية ggml_tensor * outputs = تش_اشاره;
    بنية ggml_tensor * labels  = تش_اشاره;

    بنية ggml_tensor * loss     = تش_اشاره;
    بنية ggml_tensor * pred     = تش_اشاره;
    بنية ggml_tensor * ncorrect = تش_اشاره;

    بنية ggml_cgraph * gf      = تش_اشاره;
    بنية ggml_cgraph * gb_grad = تش_اشاره;
    بنية ggml_cgraph * gb_opt  = تش_اشاره;
    منطقي static_graphs           = خطأ;
    منطقي eval_ready              = خطأ;
    std::vector<بنية ggml_tensor *> grad_accs;
    std::vector<بنية ggml_tensor *> grad_m;
    std::vector<بنية ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    منطقي    loss_فی_datapoint = خطأ;

    ggml_opt_get_optimizer_params get_opt_pars    = تش_اشاره;
    فراغ *                        get_opt_pars_ud = تش_اشاره;
    بنية ggml_tensor *          opt_ګام_params = تش_اشاره; // Stores output of get_opt_pars.

    تعداد ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

بنية ggml_opt_پایله {
    int64_t              ndata    = 0;
    std::vector<عائم>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    منطقي    loss_فی_datapoint = خطأ;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        تعداد ggml_type type_معلومات,
        تعداد ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t پایله = جديد ggml_opt_dataset;
    پایله->ndata       = ndata;
    پایله->ndata_shard = ndata_shard;

    {
        بنية ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ تش_اشاره,
            /*.no_alloc   =*/ ریښتیا,
        };
        پایله->ctx = ggml_init(params);
    }

    پایله->معلومات = ggml_new_tensor_2d(پایله->ctx, type_معلومات, ne_datapoint, ndata);
    پایله->nbs_معلومات = ggml_nbytes(پایله->معلومات) * ndata_shard/ndata;

    إذا (ne_label > 0) {
        پایله->labels = ggml_new_tensor_2d(پایله->ctx, type_label, ne_label, ndata);
        پایله->nbs_labels = ggml_nbytes(پایله->labels) * ndata_shard/ndata;
    } وإلا {
        پایله->labels = تش_اشاره;
        پایله->nbs_labels = 0;
    }

    پایله->buf = ggml_backend_alloc_ctx_tensors_from_buft(پایله->ctx, ggml_backend_cpu_بفر_type());

    ثابت int64_t nshards = ndata/ndata_shard;
    پایله->permutation.resize(nshards);
    لكل (int64_t i = 0; i < nshards; ++i) {
        پایله->permutation[i] = i;
    }
    أرجع پایله;
}

فراغ ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_بفر_free(dataset->buf);
    ggml_free(dataset->ctx);
    ړنګول dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    أرجع dataset->ndata;
}

بنية ggml_tensor * ggml_opt_dataset_معلومات(ggml_opt_dataset_t dataset) {
    أرجع dataset->معلومات;
}

بنية ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    أرجع dataset->labels;
}

فراغ ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    إذا (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        أرجع;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    ثابت int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

فراغ ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, بنية ggml_tensor * معلومات_batch, بنية ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   معلومات_batch && ggml_is_contiguous(معلومات_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == تش_اشاره) == (dataset->labels == تش_اشاره));
    GGML_ASSERT(                   معلومات_batch->type == dataset->معلومات->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    ثابت size_t nb_معلومات_batch = ggml_nbytes(معلومات_batch);
    GGML_ASSERT(nb_معلومات_batch % dataset->nbs_معلومات == 0);
    ثابت int64_t shards_فی_batch = nb_معلومات_batch / dataset->nbs_معلومات;

    إذا (labels_batch) {
        ثابت size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_فی_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_فی_batch <= int64_t(dataset->permutation.size()));

    لكل (int64_t ishard_batch = 0; ishard_batch < shards_فی_batch; ++ishard_batch) {
        ثابت int64_t ishard = dataset->permutation[ibatch*shards_فی_batch + ishard_batch];

        ثابت حرف * ptr_معلومات = (ثابت حرف *) dataset->معلومات->معلومات + ishard*dataset->nbs_معلومات;
        ggml_backend_tensor_سېټ(معلومات_batch, ptr_معلومات, ishard_batch*dataset->nbs_معلومات, dataset->nbs_معلومات);

        إذا (!labels_batch) {
            استمر;
        }

        ثابت حرف * ptr_labels = (ثابت حرف *) dataset->labels->معلومات + ishard*dataset->nbs_labels;
        ggml_backend_tensor_سېټ(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

فراغ ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, فراغ * معلومات_batch, size_t nb_معلومات_batch, فراغ * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == تش_اشاره) == (dataset->labels == تش_اشاره));
    GGML_ASSERT(nb_معلومات_batch % dataset->nbs_معلومات == 0);

    ثابت int64_t shards_فی_batch = nb_معلومات_batch / dataset->nbs_معلومات;

    GGML_ASSERT((ibatch + 1)*shards_فی_batch <= int64_t(dataset->permutation.size()));

    لكل (int64_t ishard_batch = 0; ishard_batch < shards_فی_batch; ++ishard_batch) {
        ثابت int64_t ishard = dataset->permutation[ibatch*shards_فی_batch + ishard_batch];

        ثابت حرف * ptr_معلومات       = (ثابت حرف *) dataset->معلومات->معلومات + ishard      *dataset->nbs_معلومات;
        حرف       * ptr_معلومات_batch = (حرف       *) معلومات_batch          + ishard_batch*dataset->nbs_معلومات;
        حافظه_نقل(ptr_معلومات_batch, ptr_معلومات, dataset->nbs_معلومات);

        إذا (!labels_batch) {
            استمر;
        }

        ثابت حرف * ptr_labels       = (ثابت حرف *) dataset->labels->معلومات + ishard      *dataset->nbs_labels;
        حرف       * ptr_labels_batch = (حرف       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        حافظه_نقل(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

بنية ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(فراغ * userdata) {
    GGML_ناکارول_شوی(userdata);

    ggml_opt_optimizer_params پایله;

    پایله.adamw.alpha = 0.001f;
    پایله.adamw.beta1 = 0.9f;
    پایله.adamw.beta2 = 0.999f;
    پایله.adamw.eps   = 1e-8f;
    پایله.adamw.wd    = 0.0f;

    پایله.sgd.alpha   = 1e-3f;
    پایله.sgd.wd      = 0.0f;

    أرجع پایله;
}


بنية ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(فراغ * userdata) {
    أرجع *((بنية ggml_opt_optimizer_params *) userdata);
}

بنية ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        تعداد ggml_opt_loss_type   loss_type) {
    أرجع {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ تش_اشاره,
        /*inputs          =*/ تش_اشاره,
        /*logits          =*/ تش_اشاره,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ تش_اشاره,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

ساكن ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    إذا (!tensor) {
        أرجع تش_اشاره;
    }

    إذا (tensor_map.find(tensor) != tensor_map.end()) {
        أرجع tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    حافظه_نقل(new_tensor->op_params, tensor->op_params, حجم(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->معلومات = tensor->معلومات;
    new_tensor->بفر = tensor->بفر;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    أرجع new_tensor;
}

ساكن ggml_cgraph * dup_ګراف(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_ګراف_custom(ctx, src->size, /*grads =*/ ریښتیا);

    لكل (صحيح i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    لكل (صحيح i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    لكل (صحيح i = 0; i < src->n_nodes; ++i) {
        ثابت size_t igrad_src = ggml_hash_پیدا_کړه(&src->لیدل_شوی_hash_سېټ, src->nodes[i]);
        ثابت size_t igrad_dst = ggml_hash_پیدا_کړه(&dst->لیدل_شوی_hash_سېټ, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_ډک);
        GGML_ASSERT(ggml_bitset_get(src->لیدل_شوی_hash_سېټ.کارول_شوی, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_ډک);
        GGML_ASSERT(ggml_bitset_get(dst->لیدل_شوی_hash_سېټ.کارول_شوی, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    أرجع dst;
}

ساكن فراغ ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->معلومات) && "when using static graphs the inputs must be allocated statically");

    ثابت تعداد ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    ثابت منطقي accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    ثابت منطقي need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_سېټ_ننوت(opt_ctx->inputs);
    ggml_سېټ_output(opt_ctx->outputs);

    صحيح n_param = 0;
    لكل (صحيح i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        ثابت بنية ggml_tensor * غوټه = opt_ctx->gf->nodes[i];
        إذا (غوټه->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(غوټه->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    إذا (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        ثابت size_t tensors_فی_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        ثابت size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        ثابت size_t اندازه_meta = (n_loss + tensors_فی_param*n_param + tensors_const) * ggml_tensor_overhead();
        بنية ggml_init_params params = {
            /*.mem_size   =*/ اندازه_meta,
            /*.mem_buffer =*/ تش_اشاره,
            /*.no_alloc   =*/ ریښتیا,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        ثابت size_t اندازه_meta = 1 * ggml_tensor_overhead();
        بنية ggml_init_params params = {
            /*.mem_size   =*/ اندازه_meta,
            /*.mem_buffer =*/ تش_اشاره,
            /*.no_alloc   =*/ ریښتیا,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_بفر_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = تش_اشاره;
    }

    بنية ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    بدل (opt_ctx->loss_type) {
        حالة GGML_OPT_LOSS_TYPE_منځنۍ: {
            opt_ctx->loss = ggml_مجموعه(ctx_results, opt_ctx->outputs);
            ggml_سېټ_name(opt_ctx->loss, "loss_sum");
            ثابت عائم scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_سېټ_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_فی_datapoint = ریښتیا;
            اكسر;
        }
        حالة GGML_OPT_LOSS_TYPE_مجموعه: {
            opt_ctx->loss = ggml_مجموعه(ctx_results, opt_ctx->outputs);
            ggml_سېټ_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_فی_datapoint = خطأ;
            اكسر;
        }
        حالة GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_سېټ_ننوت(opt_ctx->labels);
            ggml_سېټ_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_سېټ_name(opt_ctx->loss, "loss_cross_entropy");
            إذا (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_سېټ_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_فی_datapoint = ریښتیا;
            اكسر;
        }
        حالة GGML_OPT_LOSS_TYPE_منځنۍ_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_سېټ_ننوت(opt_ctx->labels);
            ggml_سېټ_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_سېټ_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_سېټ_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_مجموعه(ctx_results, opt_ctx->loss);
            ggml_سېټ_name(opt_ctx->loss, "loss_sum_squared_error");
            ثابت عائم scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_سېټ_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_فی_datapoint = ریښتیا;
            اكسر;
        }
    }
    ggml_سېټ_output(opt_ctx->loss);
    ggml_سېټ_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    إذا (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_سېټ_name(opt_ctx->pred, "pred");
        ggml_سېټ_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_شمېر_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_سېټ_name(opt_ctx->ncorrect, "ncorrect");
        ggml_سېټ_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    إذا (opt_ctx->buf_static) {
        إذا (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            أرجع;
        }
    } وإلا إذا (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        أرجع;
    }

    إذا (opt_ctx->grad_accs.تشه()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        ثابت صحيح n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        لكل (صحيح i = 0; i < n_nodes; ++i) {
            ggml_tensor * غوټه = opt_ctx->gf->nodes[i];
            إذا ((accumulate && (غوټه->flags & GGML_TENSOR_FLAG_PARAM)) || (غوټه->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, غوټه->ne);
            } وإلا {
                opt_ctx->grad_accs[i] = تش_اشاره;
            }
        }

        إذا (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            لكل (صحيح i = 0; i < n_nodes; ++i) {
                ggml_tensor * غوټه = opt_ctx->gf->nodes[i];
                إذا (غوټه->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, غوټه->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, غوټه->ne);
                } وإلا {
                    opt_ctx->grad_m[i] = تش_اشاره;
                    opt_ctx->grad_v[i] = تش_اشاره;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_ګراف_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ ریښتیا);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.معلومات());

    إذا (opt_ctx->buf_static) {
        إذا (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            أرجع;
        }
    } وإلا إذا (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_ګراف_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_ګراف_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ ریښتیا);

    opt_ctx->opt_ګام_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_ګام_params;
    ggml_سېټ_ننوت(adamw_params);
    ثابت حرف * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    لكل (صحيح i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        بنية ggml_tensor * غوټه = opt_ctx->gb_opt->nodes[i];
        بنية ggml_tensor * grad = ggml_ګراف_get_grad(opt_ctx->gb_opt, غوټه);

        إذا (grad && (غوټه->flags & GGML_TENSOR_FLAG_PARAM)) {
            بنية ggml_tensor * m = تش_اشاره;
            بنية ggml_tensor * v = تش_اشاره;
            إذا (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", غوټه->name);
                ggml_format_name(v, "AdamW v for %s", غوټه->name);
            }
            بنية ggml_tensor * opt_ګام;
            بدل (optimizer) {
                حالة GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_ګام = ggml_opt_ګام_adamw(opt_ctx->ctx_compute, غوټه, grad, m, v, adamw_params);
                    اكسر;
                حالة GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_ګام = ggml_opt_ګام_sgd(opt_ctx->ctx_compute, غوټه, grad, adamw_params);
                    اكسر;
                افتراضي:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_ګام, "%s step for %s", optimizer_name, غوټه->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_ګام);
        }
    }

    إذا (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_ګراف_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_بفر_type());
}

ggml_opt_context_t ggml_opt_init(بنية ggml_opt_params params) {
    ggml_opt_context_t پایله = جديد بنية ggml_opt_context;
    پایله->backend_sched    = params.backend_sched;
    پایله->ctx_compute      = params.ctx_compute;
    پایله->loss_type        = params.loss_type;
    پایله->build_type       = params.build_type;
    پایله->build_type_alloc = params.build_type;
    پایله->inputs           = params.inputs;
    پایله->outputs          = params.outputs;
    پایله->opt_period       = params.opt_period;
    پایله->get_opt_pars     = params.get_opt_pars;
    پایله->get_opt_pars_ud  = params.get_opt_pars_ud;
    پایله->optimizer        = params.optimizer;

    GGML_ASSERT(پایله->opt_period >= 1);

    پایله->static_graphs = پایله->ctx_compute;

    إذا (!پایله->static_graphs) {
        GGML_ASSERT(!پایله->inputs);
        GGML_ASSERT(!پایله->outputs);
        أرجع پایله;
    }

    GGML_ASSERT(پایله->inputs);
    GGML_ASSERT(پایله->outputs);

    پایله->gf = ggml_new_ګراف_custom(پایله->ctx_compute, GGML_DEFAULT_ګراف_اندازه, /*grads =*/ ریښتیا); // Forward pass.
    ggml_build_forward_expand(پایله->gf, پایله->outputs);

    ggml_opt_build(پایله);

    أرجع پایله;
}

فراغ ggml_opt_free(ggml_opt_context_t opt_ctx) {
    إذا (opt_ctx == تش_اشاره) {
        أرجع;
    }
    ggml_backend_بفر_free(opt_ctx->buf_static);
    ggml_backend_بفر_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    ړنګول opt_ctx;
}

فراغ ggml_opt_reset(ggml_opt_context_t opt_ctx, منطقي optimizer) {
    إذا (optimizer) {
        ggml_ګراف_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } وإلا {
        ggml_ګراف_reset(opt_ctx->gb_grad);
    }
}

منطقي ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    أرجع opt_ctx->static_graphs;
}

بنية ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    أرجع opt_ctx->inputs;
}

بنية ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    أرجع opt_ctx->outputs;
}

بنية ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    أرجع opt_ctx->labels;
}

بنية ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    أرجع opt_ctx->loss;
}

بنية ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    أرجع opt_ctx->pred;
}

بنية ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    أرجع opt_ctx->ncorrect;
}

بنية ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, بنية ggml_tensor * غوټه) {
    أرجع ggml_ګراف_get_grad_acc(opt_ctx->gb_opt, غوټه);
}

// ====== Optimization Result ======

ggml_opt_پایله_t ggml_opt_پایله_init() {
    أرجع جديد ggml_opt_پایله;
}

فراغ ggml_opt_پایله_free(ggml_opt_پایله_t پایله) {
    ړنګول پایله;
}

فراغ ggml_opt_پایله_reset(ggml_opt_پایله_t پایله) {
    پایله->ndata = 0;
    پایله->loss.clear();
    پایله->pred.clear();
    پایله->ncorrect = 0;
}

فراغ ggml_opt_پایله_ndata(ggml_opt_پایله_t پایله, int64_t * ndata) {
    *ndata = پایله->ndata;
}

فراغ ggml_opt_پایله_loss(ggml_opt_پایله_t پایله, مزدوج * loss, مزدوج * unc) {
    ثابت int64_t nbatches = پایله->loss.size(); // Number of physical batches.

    إذا (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        أرجع;
    }

    مزدوج مجموعه         = 0.0;
    مزدوج مجموعه_squared = 0.0;

    لكل (ثابت عائم & loss : پایله->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        ثابت عائم loss_scaled = پایله->loss_فی_datapoint ? loss*پایله->opt_period : loss;
        مجموعه         += loss_scaled;
        مجموعه_squared += loss_scaled*loss_scaled;
    }

    ثابت مزدوج منځنۍ_2 = مجموعه/nbatches;
    *loss = پایله->loss_فی_datapoint ? منځنۍ_2 : مجموعه;

    إذا (!unc) {
        أرجع;
    }

    إذا (nbatches < 2) {
        *unc = NAN;
        أرجع;
    }

    ثابت مزدوج var_مجموعه = مجموعه_squared/nbatches - منځنۍ_2*منځنۍ_2; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = پایله->loss_فی_datapoint ? جذر(var_مجموعه / (nbatches - 1)) : جذر(var_مجموعه * nbatches/(nbatches - 1));
}

فراغ ggml_opt_پایله_pred(ggml_opt_پایله_t پایله, int32_t * pred) {
    لكل (size_t i = 0; i < پایله->pred.size(); ++i) {
        pred[i] = پایله->pred[i];
    }
}

فراغ ggml_opt_پایله_accuracy(ggml_opt_پایله_t پایله, مزدوج * accuracy, مزدوج * unc) {
    *accuracy = پایله->ncorrect >= 0 ? مزدوج(پایله->ncorrect) / مزدوج(پایله->ndata) : NAN;

    إذا (!unc) {
        أرجع;
    }

    *unc = پایله->ncorrect >= 0 && پایله->ndata >= 2 ?
        جذر((*accuracy) * (1.0 - (*accuracy)) / مزدوج(پایله->ndata - 1)) : NAN;
}

// ====== Computation ======

فراغ ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        بنية ggml_context * ctx_compute,
        بنية ggml_cgraph  * gf,
        بنية ggml_tensor  * inputs,
        بنية ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

فراغ ggml_opt_alloc(ggml_opt_context_t opt_ctx, منطقي backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    إذا (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_ګراف_reset(opt_ctx->gb_grad);
    }
    إذا (backward) {
        ثابت int32_t opt_i_بل = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_بل == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } وإلا {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    إذا (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    بنية ggml_cgraph * graph = تش_اشاره;
    بدل (opt_ctx->build_type) {
        حالة GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } اكسر;
        حالة GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } اكسر;
        حالة GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } اكسر;
    }
    GGML_ASSERT(graph);

    إذا (opt_ctx->allocated_ګراف == graph) {
        opt_ctx->eval_ready = ریښتیا;
        أرجع;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    إذا (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_ګراف_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ تش_اشاره,
            /*.no_alloc   =*/ ریښتیا,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_ګراف_copy = dup_ګراف(opt_ctx->ctx_copy, graph);
    } وإلا {
        opt_ctx->allocated_ګراف_copy = graph;
    }

    ggml_backend_sched_alloc_ګراف(opt_ctx->backend_sched, opt_ctx->allocated_ګراف_copy);
    opt_ctx->allocated_ګراف = graph;

    opt_ctx->eval_ready = ریښتیا;
}

فراغ ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_پایله_t پایله) {
    GGML_ASSERT(opt_ctx->eval_ready);
    إذا (opt_ctx->allocated_ګراف == opt_ctx->gb_opt) {
        ثابت ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        بدل (opt_ctx->optimizer) {
            حالة GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                ثابت عائم beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                ثابت عائم beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                عائم * adamw_par_معلومات = ggml_get_معلومات_f32(opt_ctx->opt_ګام_params);
                adamw_par_معلومات[0] = opt_pars.adamw.alpha;
                adamw_par_معلومات[1] = opt_pars.adamw.beta1;
                adamw_par_معلومات[2] = opt_pars.adamw.beta2;
                adamw_par_معلومات[3] = opt_pars.adamw.eps;
                adamw_par_معلومات[4] = opt_pars.adamw.wd;
                adamw_par_معلومات[5] = beta1h;
                adamw_par_معلومات[6] = beta2h;
            } اكسر;
            حالة GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                عائم * sgd = ggml_get_معلومات_f32(opt_ctx->opt_ګام_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } اكسر;
            افتراضي:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_ګراف_compute(opt_ctx->backend_sched, opt_ctx->allocated_ګراف_copy);
    opt_ctx->iter += opt_ctx->allocated_ګراف == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    إذا (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = تش_اشاره;
        opt_ctx->gb_grad              = تش_اشاره;
        opt_ctx->gb_opt               = تش_اشاره;
        opt_ctx->allocated_ګراف      = تش_اشاره;
        opt_ctx->allocated_ګراف_copy = تش_اشاره;
    }

    opt_ctx->eval_ready = خطأ;

    إذا (!پایله) {
        أرجع;
    }

    إذا (پایله->ndata == 0) {
        پایله->loss_فی_datapoint = opt_ctx->loss_فی_datapoint;
        پایله->opt_period         = opt_ctx->opt_period;
    } وإلا {
        GGML_ASSERT(پایله->loss_فی_datapoint == opt_ctx->loss_فی_datapoint);
        GGML_ASSERT(پایله->opt_period         == opt_ctx->opt_period);
    }

    ثابت int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(پایله->ndata == ndata*int64_t(پایله->loss.size()) && "varying batch size not supported");
    پایله->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    عائم loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    پایله->loss.push_back(loss);

    إذا (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.معلومات(), 0, ggml_nbytes(opt_ctx->pred));
        پایله->pred.insert(پایله->pred.end(), pred.begin(), pred.end());
    }

    إذا (!opt_ctx->ncorrect || پایله->ncorrect < 0) {
        پایله->ncorrect = -1;
        أرجع;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    پایله->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

فراغ ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_پایله_t       پایله_train,
        ggml_opt_پایله_t       پایله_eval,
        int64_t                 idata_وېشل,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    بنية ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    بنية ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    بنية ggml_tensor * معلومات   = ggml_opt_dataset_معلومات(dataset);
    GGML_ASSERT(معلومات->ne[0] == inputs->ne[0]);

    ثابت int64_t ndata       =   معلومات->ne[1];
    ثابت int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(معلومات->ne[1] % inputs->ne[1] == 0);
    ثابت int64_t nbatches = ndata/ndata_batch;

    idata_وېشل = idata_وېشل < 0 ? ndata : idata_وېشل;
    GGML_ASSERT(idata_وېشل % ndata_batch == 0);
    ثابت int64_t ibatch_وېشل = idata_وېشل / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_پیل = ggml_وخت_us();
    لكل (; ibatch < ibatch_وېشل; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ ریښتیا);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, پایله_train);
        إذا (callback_train) {
            callback_train(ریښتیا, opt_ctx, dataset, پایله_train, ibatch+1, ibatch_وېشل, t_loop_پیل);
        }
    }
    t_loop_پیل = ggml_وخت_us();
    لكل (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ خطأ);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, پایله_eval);
        إذا (callback_eval) {
            callback_eval(خطأ, opt_ctx, dataset, پایله_eval, ibatch+1-ibatch_وېشل, nbatches-ibatch_وېشل, t_loop_پیل);
        }
    }
}

فراغ ggml_opt_epoch_callback_پرمختګ_bar(
        منطقي               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_پایله_t  پایله,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_پیل_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_اوږدتیا = 8;
    ثابت int64_t ibatch8 = 8 * ibatch;
    لكل (int64_t j = 0; j < bar_اوږدتیا; ++j) {
        إذا        (ibatch_max * (8*j + 8) / bar_اوږدتیا < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } وإلا إذا (ibatch_max * (8*j + 7) / bar_اوږدتیا < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } وإلا إذا (ibatch_max * (8*j + 6) / bar_اوږدتیا < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } وإلا إذا (ibatch_max * (8*j + 5) / bar_اوږدتیا < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } وإلا إذا (ibatch_max * (8*j + 4) / bar_اوږدتیا < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } وإلا إذا (ibatch_max * (8*j + 3) / bar_اوږدتیا < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } وإلا إذا (ibatch_max * (8*j + 2) / bar_اوږدتیا < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } وإلا إذا (ibatch_max * (8*j + 1) / bar_اوږدتیا < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } وإلا {
            fprintf(stderr, " ");
        }
    }

    ثابت int64_t batch_اندازه = ggml_opt_inputs(opt_ctx)->ne[1];
    ثابت int64_t idata      = ibatch*batch_اندازه;
    ثابت int64_t idata_max  = ibatch_max*batch_اندازه;

    مزدوج loss;
    مزدوج loss_unc;
    ggml_opt_پایله_loss(پایله, &loss, &loss_unc);

    مزدوج accuracy;
    مزدوج accuracy_unc;
    ggml_opt_پایله_accuracy(پایله, &accuracy, &accuracy_unc);

    ثابت int64_t t_ibatch_us = ggml_وخت_us() - t_پیل_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    ثابت int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    ثابت int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    ثابت int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    ثابت int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    ثابت int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    إذا (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_ناکارول_شوی(dataset);
}

فراغ ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        تعداد ggml_opt_loss_type         loss_type,
        تعداد ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        عائم                           val_وېشل,
        منطقي                            silent) {
    ggml_وخت_init();
    ثابت int64_t t_پیل_us = ggml_وخت_us();

    ثابت int64_t ndata           = ggml_opt_dataset_معلومات(dataset)->ne[1];
    ثابت int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    ثابت int64_t opt_period       = nbatch_logical / nbatch_physical;
    ثابت int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_وېشل >= 0.0f);
    GGML_ASSERT(val_وېشل <  1.0f);
    ثابت int64_t ibatch_وېشل = int64_t(((1.0f - val_وېشل) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    ثابت int64_t idata_وېشل  = ibatch_وېشل * nbatch_physical;

    int64_t epoch = 1;

    ggml_opt_params params = ggml_opt_default_params(backend_sched, loss_type);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs         = outputs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    params.optimizer       = optimizer;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    إذا (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_پایله_t پایله_train = ggml_opt_پایله_init();
    ggml_opt_پایله_t پایله_val   = ggml_opt_پایله_init();

    ggml_opt_epoch_callback epoch_callback = silent ? تش_اشاره : ggml_opt_epoch_callback_پرمختګ_bar;

    لكل (; epoch <= nepoch; ++epoch) {
        إذا (nbatch_logical < idata_وېشل) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_وېشل);
        }

        ggml_opt_پایله_reset(پایله_train);
        ggml_opt_پایله_reset(پایله_val);

        إذا (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, پایله_train, پایله_val, idata_وېشل, epoch_callback, epoch_callback);
        إذا (!silent) {
            fprintf(stderr, "\n");
        }
    }

    إذا (!silent) {
        int64_t t_ټول_s = (ggml_وخت_us() - t_پیل_us) / 1000000;
        ثابت int64_t t_ټول_h = t_ټول_s / 3600;
        t_ټول_s -= t_ټول_h * 3600;
        ثابت int64_t t_ټول_m = t_ټول_s / 60;
        t_ټول_s -= t_ټول_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_ټول_h, t_ټول_m, t_ټول_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_پایله_free(پایله_train);
    ggml_opt_پایله_free(پایله_val);
}

تعداد ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    أرجع c->optimizer;
}

GGML_API ثابت حرف * ggml_opt_optimizer_name(تعداد ggml_opt_optimizer_type o) {
    بدل (o) {
        حالة GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            أرجع "adamw";
        حالة GGML_OPT_OPTIMIZER_TYPE_SGD:
            أرجع "sgd";
        افتراضي:
            أرجع "undefined";
    };
}
