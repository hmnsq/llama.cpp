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
    بنية ggml_context   * ctx    = خالی_سوچک;
    ggml_backend_بفر_t   buf    = خالی_سوچک;
    بنية ggml_tensor    * انکڑے   = خالی_سوچک;
    بنية ggml_tensor    * labels = خالی_سوچک;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_انکڑے    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

بنية ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = خالی_سوچک;
    ggml_cgraph              * allocated_گراف      = خالی_سوچک;
    ggml_cgraph              * allocated_گراف_copy = خالی_سوچک;
    بنية ggml_context      * ctx_static           = خالی_سوچک;
    بنية ggml_context      * ctx_cpu              = خالی_سوچک;
    بنية ggml_context      * ctx_compute          = خالی_سوچک;
    بنية ggml_context      * ctx_copy             = خالی_سوچک;
    ggml_backend_بفر_t      buf_static           = خالی_سوچک;
    ggml_backend_بفر_t      buf_cpu              = خالی_سوچک;
    std::mt19937               rng;
    تعداد ggml_opt_loss_type    loss_type;
    تعداد ggml_opt_build_type   build_type;
    تعداد ggml_opt_build_type   build_type_alloc;

    بنية ggml_tensor * inputs  = خالی_سوچک;
    بنية ggml_tensor * outputs = خالی_سوچک;
    بنية ggml_tensor * labels  = خالی_سوچک;

    بنية ggml_tensor * loss     = خالی_سوچک;
    بنية ggml_tensor * pred     = خالی_سوچک;
    بنية ggml_tensor * ncorrect = خالی_سوچک;

    بنية ggml_cgraph * gf      = خالی_سوچک;
    بنية ggml_cgraph * gb_grad = خالی_سوچک;
    بنية ggml_cgraph * gb_opt  = خالی_سوچک;
    منطقي static_graphs           = خطأ;
    منطقي eval_ready              = خطأ;
    std::vector<بنية ggml_tensor *> grad_accs;
    std::vector<بنية ggml_tensor *> grad_m;
    std::vector<بنية ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    منطقي    loss_فی_datapoint = خطأ;

    ggml_opt_get_optimizer_params get_opt_pars    = خالی_سوچک;
    فراغ *                        get_opt_pars_ud = خالی_سوچک;
    بنية ggml_tensor *          opt_قدم_params = خالی_سوچک; // Stores output of get_opt_pars.

    تعداد ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

بنية ggml_opt_نتیجہ {
    int64_t              ndata    = 0;
    std::vector<عائم>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    منطقي    loss_فی_datapoint = خطأ;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        تعداد ggml_type type_انکڑے,
        تعداد ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t نتیجہ = جديد ggml_opt_dataset;
    نتیجہ->ndata       = ndata;
    نتیجہ->ndata_shard = ndata_shard;

    {
        بنية ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ خالی_سوچک,
            /*.no_alloc   =*/ سچ,
        };
        نتیجہ->ctx = ggml_init(params);
    }

    نتیجہ->انکڑے = ggml_new_tensor_2d(نتیجہ->ctx, type_انکڑے, ne_datapoint, ndata);
    نتیجہ->nbs_انکڑے = ggml_nbytes(نتیجہ->انکڑے) * ndata_shard/ndata;

    إذا (ne_label > 0) {
        نتیجہ->labels = ggml_new_tensor_2d(نتیجہ->ctx, type_label, ne_label, ndata);
        نتیجہ->nbs_labels = ggml_nbytes(نتیجہ->labels) * ndata_shard/ndata;
    } وإلا {
        نتیجہ->labels = خالی_سوچک;
        نتیجہ->nbs_labels = 0;
    }

    نتیجہ->buf = ggml_backend_alloc_ctx_tensors_from_buft(نتیجہ->ctx, ggml_backend_cpu_بفر_type());

    ثابت int64_t nshards = ndata/ndata_shard;
    نتیجہ->permutation.resize(nshards);
    لكل (int64_t i = 0; i < nshards; ++i) {
        نتیجہ->permutation[i] = i;
    }
    أرجع نتیجہ;
}

فراغ ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_بفر_free(dataset->buf);
    ggml_free(dataset->ctx);
    مٹاؤ dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    أرجع dataset->ndata;
}

بنية ggml_tensor * ggml_opt_dataset_انکڑے(ggml_opt_dataset_t dataset) {
    أرجع dataset->انکڑے;
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

فراغ ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, بنية ggml_tensor * انکڑے_batch, بنية ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   انکڑے_batch && ggml_is_contiguous(انکڑے_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == خالی_سوچک) == (dataset->labels == خالی_سوچک));
    GGML_ASSERT(                   انکڑے_batch->type == dataset->انکڑے->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    ثابت size_t nb_انکڑے_batch = ggml_nbytes(انکڑے_batch);
    GGML_ASSERT(nb_انکڑے_batch % dataset->nbs_انکڑے == 0);
    ثابت int64_t shards_فی_batch = nb_انکڑے_batch / dataset->nbs_انکڑے;

    إذا (labels_batch) {
        ثابت size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_فی_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_فی_batch <= int64_t(dataset->permutation.size()));

    لكل (int64_t ishard_batch = 0; ishard_batch < shards_فی_batch; ++ishard_batch) {
        ثابت int64_t ishard = dataset->permutation[ibatch*shards_فی_batch + ishard_batch];

        ثابت حرف * ptr_انکڑے = (ثابت حرف *) dataset->انکڑے->انکڑے + ishard*dataset->nbs_انکڑے;
        ggml_backend_tensor_سیٹ(انکڑے_batch, ptr_انکڑے, ishard_batch*dataset->nbs_انکڑے, dataset->nbs_انکڑے);

        إذا (!labels_batch) {
            استمر;
        }

        ثابت حرف * ptr_labels = (ثابت حرف *) dataset->labels->انکڑے + ishard*dataset->nbs_labels;
        ggml_backend_tensor_سیٹ(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

فراغ ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, فراغ * انکڑے_batch, size_t nb_انکڑے_batch, فراغ * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == خالی_سوچک) == (dataset->labels == خالی_سوچک));
    GGML_ASSERT(nb_انکڑے_batch % dataset->nbs_انکڑے == 0);

    ثابت int64_t shards_فی_batch = nb_انکڑے_batch / dataset->nbs_انکڑے;

    GGML_ASSERT((ibatch + 1)*shards_فی_batch <= int64_t(dataset->permutation.size()));

    لكل (int64_t ishard_batch = 0; ishard_batch < shards_فی_batch; ++ishard_batch) {
        ثابت int64_t ishard = dataset->permutation[ibatch*shards_فی_batch + ishard_batch];

        ثابت حرف * ptr_انکڑے       = (ثابت حرف *) dataset->انکڑے->انکڑے + ishard      *dataset->nbs_انکڑے;
        حرف       * ptr_انکڑے_batch = (حرف       *) انکڑے_batch          + ishard_batch*dataset->nbs_انکڑے;
        حافظہ_نقل(ptr_انکڑے_batch, ptr_انکڑے, dataset->nbs_انکڑے);

        إذا (!labels_batch) {
            استمر;
        }

        ثابت حرف * ptr_labels       = (ثابت حرف *) dataset->labels->انکڑے + ishard      *dataset->nbs_labels;
        حرف       * ptr_labels_batch = (حرف       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        حافظہ_نقل(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

بنية ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(فراغ * userdata) {
    GGML_بن_ورتیا(userdata);

    ggml_opt_optimizer_params نتیجہ;

    نتیجہ.adamw.alpha = 0.001f;
    نتیجہ.adamw.beta1 = 0.9f;
    نتیجہ.adamw.beta2 = 0.999f;
    نتیجہ.adamw.eps   = 1e-8f;
    نتیجہ.adamw.wd    = 0.0f;

    نتیجہ.sgd.alpha   = 1e-3f;
    نتیجہ.sgd.wd      = 0.0f;

    أرجع نتیجہ;
}


بنية ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(فراغ * userdata) {
    أرجع *((بنية ggml_opt_optimizer_params *) userdata);
}

بنية ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        تعداد ggml_opt_loss_type   loss_type) {
    أرجع {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ خالی_سوچک,
        /*inputs          =*/ خالی_سوچک,
        /*logits          =*/ خالی_سوچک,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ خالی_سوچک,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

ساكن ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    إذا (!tensor) {
        أرجع خالی_سوچک;
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
    حافظہ_نقل(new_tensor->op_params, tensor->op_params, حجم(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->انکڑے = tensor->انکڑے;
    new_tensor->بفر = tensor->بفر;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    أرجع new_tensor;
}

ساكن ggml_cgraph * dup_گراف(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_گراف_custom(ctx, src->size, /*grads =*/ سچ);

    لكل (صحيح i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    لكل (صحيح i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    لكل (صحيح i = 0; i < src->n_nodes; ++i) {
        ثابت size_t igrad_src = ggml_hash_لبھو(&src->ویکھیا_hash_سیٹ, src->nodes[i]);
        ثابت size_t igrad_dst = ggml_hash_لبھو(&dst->ویکھیا_hash_سیٹ, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_بھریا);
        GGML_ASSERT(ggml_bitset_get(src->ویکھیا_hash_سیٹ.ورتیا, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_بھریا);
        GGML_ASSERT(ggml_bitset_get(dst->ویکھیا_hash_سیٹ.ورتیا, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    أرجع dst;
}

ساكن فراغ ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->انکڑے) && "when using static graphs the inputs must be allocated statically");

    ثابت تعداد ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    ثابت منطقي accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    ثابت منطقي need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_سیٹ_داخلہ(opt_ctx->inputs);
    ggml_سیٹ_output(opt_ctx->outputs);

    صحيح n_param = 0;
    لكل (صحيح i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        ثابت بنية ggml_tensor * گرہ = opt_ctx->gf->nodes[i];
        إذا (گرہ->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(گرہ->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
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
        ثابت size_t ناپ_meta = (n_loss + tensors_فی_param*n_param + tensors_const) * ggml_tensor_overhead();
        بنية ggml_init_params params = {
            /*.mem_size   =*/ ناپ_meta,
            /*.mem_buffer =*/ خالی_سوچک,
            /*.no_alloc   =*/ سچ,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        ثابت size_t ناپ_meta = 1 * ggml_tensor_overhead();
        بنية ggml_init_params params = {
            /*.mem_size   =*/ ناپ_meta,
            /*.mem_buffer =*/ خالی_سوچک,
            /*.no_alloc   =*/ سچ,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_بفر_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = خالی_سوچک;
    }

    بنية ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    بدل (opt_ctx->loss_type) {
        حالة GGML_OPT_LOSS_TYPE_اوسط: {
            opt_ctx->loss = ggml_جوڑ(ctx_results, opt_ctx->outputs);
            ggml_سیٹ_name(opt_ctx->loss, "loss_sum");
            ثابت عائم scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_سیٹ_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_فی_datapoint = سچ;
            اكسر;
        }
        حالة GGML_OPT_LOSS_TYPE_جوڑ: {
            opt_ctx->loss = ggml_جوڑ(ctx_results, opt_ctx->outputs);
            ggml_سیٹ_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_فی_datapoint = خطأ;
            اكسر;
        }
        حالة GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_سیٹ_داخلہ(opt_ctx->labels);
            ggml_سیٹ_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_سیٹ_name(opt_ctx->loss, "loss_cross_entropy");
            إذا (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_سیٹ_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_فی_datapoint = سچ;
            اكسر;
        }
        حالة GGML_OPT_LOSS_TYPE_اوسط_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_سیٹ_داخلہ(opt_ctx->labels);
            ggml_سیٹ_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_سیٹ_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_سیٹ_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_جوڑ(ctx_results, opt_ctx->loss);
            ggml_سیٹ_name(opt_ctx->loss, "loss_sum_squared_error");
            ثابت عائم scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_سیٹ_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_فی_datapoint = سچ;
            اكسر;
        }
    }
    ggml_سیٹ_output(opt_ctx->loss);
    ggml_سیٹ_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    إذا (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_سیٹ_name(opt_ctx->pred, "pred");
        ggml_سیٹ_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_گنتی_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_سیٹ_name(opt_ctx->ncorrect, "ncorrect");
        ggml_سیٹ_output(opt_ctx->ncorrect);
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

    إذا (opt_ctx->grad_accs.خالی_تھاواں()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        ثابت صحيح n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        لكل (صحيح i = 0; i < n_nodes; ++i) {
            ggml_tensor * گرہ = opt_ctx->gf->nodes[i];
            إذا ((accumulate && (گرہ->flags & GGML_TENSOR_FLAG_PARAM)) || (گرہ->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, گرہ->ne);
            } وإلا {
                opt_ctx->grad_accs[i] = خالی_سوچک;
            }
        }

        إذا (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            لكل (صحيح i = 0; i < n_nodes; ++i) {
                ggml_tensor * گرہ = opt_ctx->gf->nodes[i];
                إذا (گرہ->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, گرہ->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, گرہ->ne);
                } وإلا {
                    opt_ctx->grad_m[i] = خالی_سوچک;
                    opt_ctx->grad_v[i] = خالی_سوچک;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_گراف_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ سچ);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.انکڑے());

    إذا (opt_ctx->buf_static) {
        إذا (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            أرجع;
        }
    } وإلا إذا (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_گراف_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_گراف_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ سچ);

    opt_ctx->opt_قدم_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_قدم_params;
    ggml_سیٹ_داخلہ(adamw_params);
    ثابت حرف * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    لكل (صحيح i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        بنية ggml_tensor * گرہ = opt_ctx->gb_opt->nodes[i];
        بنية ggml_tensor * grad = ggml_گراف_get_grad(opt_ctx->gb_opt, گرہ);

        إذا (grad && (گرہ->flags & GGML_TENSOR_FLAG_PARAM)) {
            بنية ggml_tensor * m = خالی_سوچک;
            بنية ggml_tensor * v = خالی_سوچک;
            إذا (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", گرہ->name);
                ggml_format_name(v, "AdamW v for %s", گرہ->name);
            }
            بنية ggml_tensor * opt_قدم;
            بدل (optimizer) {
                حالة GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_قدم = ggml_opt_قدم_adamw(opt_ctx->ctx_compute, گرہ, grad, m, v, adamw_params);
                    اكسر;
                حالة GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_قدم = ggml_opt_قدم_sgd(opt_ctx->ctx_compute, گرہ, grad, adamw_params);
                    اكسر;
                افتراضي:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_قدم, "%s step for %s", optimizer_name, گرہ->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_قدم);
        }
    }

    إذا (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_گراف_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_بفر_type());
}

ggml_opt_context_t ggml_opt_init(بنية ggml_opt_params params) {
    ggml_opt_context_t نتیجہ = جديد بنية ggml_opt_context;
    نتیجہ->backend_sched    = params.backend_sched;
    نتیجہ->ctx_compute      = params.ctx_compute;
    نتیجہ->loss_type        = params.loss_type;
    نتیجہ->build_type       = params.build_type;
    نتیجہ->build_type_alloc = params.build_type;
    نتیجہ->inputs           = params.inputs;
    نتیجہ->outputs          = params.outputs;
    نتیجہ->opt_period       = params.opt_period;
    نتیجہ->get_opt_pars     = params.get_opt_pars;
    نتیجہ->get_opt_pars_ud  = params.get_opt_pars_ud;
    نتیجہ->optimizer        = params.optimizer;

    GGML_ASSERT(نتیجہ->opt_period >= 1);

    نتیجہ->static_graphs = نتیجہ->ctx_compute;

    إذا (!نتیجہ->static_graphs) {
        GGML_ASSERT(!نتیجہ->inputs);
        GGML_ASSERT(!نتیجہ->outputs);
        أرجع نتیجہ;
    }

    GGML_ASSERT(نتیجہ->inputs);
    GGML_ASSERT(نتیجہ->outputs);

    نتیجہ->gf = ggml_new_گراف_custom(نتیجہ->ctx_compute, GGML_DEFAULT_گراف_ناپ, /*grads =*/ سچ); // Forward pass.
    ggml_build_forward_expand(نتیجہ->gf, نتیجہ->outputs);

    ggml_opt_build(نتیجہ);

    أرجع نتیجہ;
}

فراغ ggml_opt_free(ggml_opt_context_t opt_ctx) {
    إذا (opt_ctx == خالی_سوچک) {
        أرجع;
    }
    ggml_backend_بفر_free(opt_ctx->buf_static);
    ggml_backend_بفر_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    مٹاؤ opt_ctx;
}

فراغ ggml_opt_reset(ggml_opt_context_t opt_ctx, منطقي optimizer) {
    إذا (optimizer) {
        ggml_گراف_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } وإلا {
        ggml_گراف_reset(opt_ctx->gb_grad);
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

بنية ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, بنية ggml_tensor * گرہ) {
    أرجع ggml_گراف_get_grad_acc(opt_ctx->gb_opt, گرہ);
}

// ====== Optimization Result ======

ggml_opt_نتیجہ_t ggml_opt_نتیجہ_init() {
    أرجع جديد ggml_opt_نتیجہ;
}

فراغ ggml_opt_نتیجہ_free(ggml_opt_نتیجہ_t نتیجہ) {
    مٹاؤ نتیجہ;
}

فراغ ggml_opt_نتیجہ_reset(ggml_opt_نتیجہ_t نتیجہ) {
    نتیجہ->ndata = 0;
    نتیجہ->loss.clear();
    نتیجہ->pred.clear();
    نتیجہ->ncorrect = 0;
}

فراغ ggml_opt_نتیجہ_ndata(ggml_opt_نتیجہ_t نتیجہ, int64_t * ndata) {
    *ndata = نتیجہ->ndata;
}

فراغ ggml_opt_نتیجہ_loss(ggml_opt_نتیجہ_t نتیجہ, مزدوج * loss, مزدوج * unc) {
    ثابت int64_t nbatches = نتیجہ->loss.size(); // Number of physical batches.

    إذا (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        أرجع;
    }

    مزدوج جوڑ         = 0.0;
    مزدوج جوڑ_squared = 0.0;

    لكل (ثابت عائم & loss : نتیجہ->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        ثابت عائم loss_scaled = نتیجہ->loss_فی_datapoint ? loss*نتیجہ->opt_period : loss;
        جوڑ         += loss_scaled;
        جوڑ_squared += loss_scaled*loss_scaled;
    }

    ثابت مزدوج اوسط_2 = جوڑ/nbatches;
    *loss = نتیجہ->loss_فی_datapoint ? اوسط_2 : جوڑ;

    إذا (!unc) {
        أرجع;
    }

    إذا (nbatches < 2) {
        *unc = NAN;
        أرجع;
    }

    ثابت مزدوج var_جوڑ = جوڑ_squared/nbatches - اوسط_2*اوسط_2; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = نتیجہ->loss_فی_datapoint ? جذر(var_جوڑ / (nbatches - 1)) : جذر(var_جوڑ * nbatches/(nbatches - 1));
}

فراغ ggml_opt_نتیجہ_pred(ggml_opt_نتیجہ_t نتیجہ, int32_t * pred) {
    لكل (size_t i = 0; i < نتیجہ->pred.size(); ++i) {
        pred[i] = نتیجہ->pred[i];
    }
}

فراغ ggml_opt_نتیجہ_accuracy(ggml_opt_نتیجہ_t نتیجہ, مزدوج * accuracy, مزدوج * unc) {
    *accuracy = نتیجہ->ncorrect >= 0 ? مزدوج(نتیجہ->ncorrect) / مزدوج(نتیجہ->ndata) : NAN;

    إذا (!unc) {
        أرجع;
    }

    *unc = نتیجہ->ncorrect >= 0 && نتیجہ->ndata >= 2 ?
        جذر((*accuracy) * (1.0 - (*accuracy)) / مزدوج(نتیجہ->ndata - 1)) : NAN;
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
        ggml_گراف_reset(opt_ctx->gb_grad);
    }
    إذا (backward) {
        ثابت int32_t opt_i_اگلا = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_اگلا == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } وإلا {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    إذا (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    بنية ggml_cgraph * graph = خالی_سوچک;
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

    إذا (opt_ctx->allocated_گراف == graph) {
        opt_ctx->eval_ready = سچ;
        أرجع;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    إذا (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_گراف_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ خالی_سوچک,
            /*.no_alloc   =*/ سچ,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_گراف_copy = dup_گراف(opt_ctx->ctx_copy, graph);
    } وإلا {
        opt_ctx->allocated_گراف_copy = graph;
    }

    ggml_backend_sched_alloc_گراف(opt_ctx->backend_sched, opt_ctx->allocated_گراف_copy);
    opt_ctx->allocated_گراف = graph;

    opt_ctx->eval_ready = سچ;
}

فراغ ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_نتیجہ_t نتیجہ) {
    GGML_ASSERT(opt_ctx->eval_ready);
    إذا (opt_ctx->allocated_گراف == opt_ctx->gb_opt) {
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

                عائم * adamw_par_انکڑے = ggml_get_انکڑے_f32(opt_ctx->opt_قدم_params);
                adamw_par_انکڑے[0] = opt_pars.adamw.alpha;
                adamw_par_انکڑے[1] = opt_pars.adamw.beta1;
                adamw_par_انکڑے[2] = opt_pars.adamw.beta2;
                adamw_par_انکڑے[3] = opt_pars.adamw.eps;
                adamw_par_انکڑے[4] = opt_pars.adamw.wd;
                adamw_par_انکڑے[5] = beta1h;
                adamw_par_انکڑے[6] = beta2h;
            } اكسر;
            حالة GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                عائم * sgd = ggml_get_انکڑے_f32(opt_ctx->opt_قدم_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } اكسر;
            افتراضي:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_گراف_compute(opt_ctx->backend_sched, opt_ctx->allocated_گراف_copy);
    opt_ctx->iter += opt_ctx->allocated_گراف == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    إذا (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = خالی_سوچک;
        opt_ctx->gb_grad              = خالی_سوچک;
        opt_ctx->gb_opt               = خالی_سوچک;
        opt_ctx->allocated_گراف      = خالی_سوچک;
        opt_ctx->allocated_گراف_copy = خالی_سوچک;
    }

    opt_ctx->eval_ready = خطأ;

    إذا (!نتیجہ) {
        أرجع;
    }

    إذا (نتیجہ->ndata == 0) {
        نتیجہ->loss_فی_datapoint = opt_ctx->loss_فی_datapoint;
        نتیجہ->opt_period         = opt_ctx->opt_period;
    } وإلا {
        GGML_ASSERT(نتیجہ->loss_فی_datapoint == opt_ctx->loss_فی_datapoint);
        GGML_ASSERT(نتیجہ->opt_period         == opt_ctx->opt_period);
    }

    ثابت int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(نتیجہ->ndata == ndata*int64_t(نتیجہ->loss.size()) && "varying batch size not supported");
    نتیجہ->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    عائم loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    نتیجہ->loss.push_back(loss);

    إذا (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.انکڑے(), 0, ggml_nbytes(opt_ctx->pred));
        نتیجہ->pred.insert(نتیجہ->pred.end(), pred.begin(), pred.end());
    }

    إذا (!opt_ctx->ncorrect || نتیجہ->ncorrect < 0) {
        نتیجہ->ncorrect = -1;
        أرجع;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    نتیجہ->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

فراغ ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_نتیجہ_t       نتیجہ_train,
        ggml_opt_نتیجہ_t       نتیجہ_eval,
        int64_t                 idata_ونڈو,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    بنية ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    بنية ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    بنية ggml_tensor * انکڑے   = ggml_opt_dataset_انکڑے(dataset);
    GGML_ASSERT(انکڑے->ne[0] == inputs->ne[0]);

    ثابت int64_t ndata       =   انکڑے->ne[1];
    ثابت int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(انکڑے->ne[1] % inputs->ne[1] == 0);
    ثابت int64_t nbatches = ndata/ndata_batch;

    idata_ونڈو = idata_ونڈو < 0 ? ndata : idata_ونڈو;
    GGML_ASSERT(idata_ونڈو % ndata_batch == 0);
    ثابت int64_t ibatch_ونڈو = idata_ونڈو / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_شروع = ggml_ویلا_us();
    لكل (; ibatch < ibatch_ونڈو; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ سچ);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, نتیجہ_train);
        إذا (callback_train) {
            callback_train(سچ, opt_ctx, dataset, نتیجہ_train, ibatch+1, ibatch_ونڈو, t_loop_شروع);
        }
    }
    t_loop_شروع = ggml_ویلا_us();
    لكل (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ خطأ);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, نتیجہ_eval);
        إذا (callback_eval) {
            callback_eval(خطأ, opt_ctx, dataset, نتیجہ_eval, ibatch+1-ibatch_ونڈو, nbatches-ibatch_ونڈو, t_loop_شروع);
        }
    }
}

فراغ ggml_opt_epoch_callback_ترقی_bar(
        منطقي               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_نتیجہ_t  نتیجہ,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_شروع_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_لمائی = 8;
    ثابت int64_t ibatch8 = 8 * ibatch;
    لكل (int64_t j = 0; j < bar_لمائی; ++j) {
        إذا        (ibatch_max * (8*j + 8) / bar_لمائی < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } وإلا إذا (ibatch_max * (8*j + 7) / bar_لمائی < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } وإلا إذا (ibatch_max * (8*j + 6) / bar_لمائی < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } وإلا إذا (ibatch_max * (8*j + 5) / bar_لمائی < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } وإلا إذا (ibatch_max * (8*j + 4) / bar_لمائی < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } وإلا إذا (ibatch_max * (8*j + 3) / bar_لمائی < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } وإلا إذا (ibatch_max * (8*j + 2) / bar_لمائی < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } وإلا إذا (ibatch_max * (8*j + 1) / bar_لمائی < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } وإلا {
            fprintf(stderr, " ");
        }
    }

    ثابت int64_t batch_ناپ = ggml_opt_inputs(opt_ctx)->ne[1];
    ثابت int64_t idata      = ibatch*batch_ناپ;
    ثابت int64_t idata_max  = ibatch_max*batch_ناپ;

    مزدوج loss;
    مزدوج loss_unc;
    ggml_opt_نتیجہ_loss(نتیجہ, &loss, &loss_unc);

    مزدوج accuracy;
    مزدوج accuracy_unc;
    ggml_opt_نتیجہ_accuracy(نتیجہ, &accuracy, &accuracy_unc);

    ثابت int64_t t_ibatch_us = ggml_ویلا_us() - t_شروع_us;
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

    GGML_بن_ورتیا(dataset);
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
        عائم                           val_ونڈو,
        منطقي                            silent) {
    ggml_ویلا_init();
    ثابت int64_t t_شروع_us = ggml_ویلا_us();

    ثابت int64_t ndata           = ggml_opt_dataset_انکڑے(dataset)->ne[1];
    ثابت int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    ثابت int64_t opt_period       = nbatch_logical / nbatch_physical;
    ثابت int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_ونڈو >= 0.0f);
    GGML_ASSERT(val_ونڈو <  1.0f);
    ثابت int64_t ibatch_ونڈو = int64_t(((1.0f - val_ونڈو) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    ثابت int64_t idata_ونڈو  = ibatch_ونڈو * nbatch_physical;

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

    ggml_opt_نتیجہ_t نتیجہ_train = ggml_opt_نتیجہ_init();
    ggml_opt_نتیجہ_t نتیجہ_val   = ggml_opt_نتیجہ_init();

    ggml_opt_epoch_callback epoch_callback = silent ? خالی_سوچک : ggml_opt_epoch_callback_ترقی_bar;

    لكل (; epoch <= nepoch; ++epoch) {
        إذا (nbatch_logical < idata_ونڈو) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_ونڈو);
        }

        ggml_opt_نتیجہ_reset(نتیجہ_train);
        ggml_opt_نتیجہ_reset(نتیجہ_val);

        إذا (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, نتیجہ_train, نتیجہ_val, idata_ونڈو, epoch_callback, epoch_callback);
        إذا (!silent) {
            fprintf(stderr, "\n");
        }
    }

    إذا (!silent) {
        int64_t t_کل_s = (ggml_ویلا_us() - t_شروع_us) / 1000000;
        ثابت int64_t t_کل_h = t_کل_s / 3600;
        t_کل_s -= t_کل_h * 3600;
        ثابت int64_t t_کل_m = t_کل_s / 60;
        t_کل_s -= t_کل_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_کل_h, t_کل_m, t_کل_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_نتیجہ_free(نتیجہ_train);
    ggml_opt_نتیجہ_free(نتیجہ_val);
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
