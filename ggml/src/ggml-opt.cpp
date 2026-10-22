#شامل "ggml-opt.h"

#شامل "ggml.h"
#شامل "ggml-alloc.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"

#شامل <algorithm>
#شامل <cmath>
#شامل <cstdint>
#شامل <cinttypes>
#شامل <map>
#شامل <random>
#شامل <vector>

ساخت ggml_opt_dataset {
    ساخت ggml_context   * ctx    = خالی_اشاریہ;
    ggml_backend_بفر_t   buf    = خالی_اشاریہ;
    ساخت ggml_tensor    * ڈیٹا   = خالی_اشاریہ;
    ساخت ggml_tensor    * labels = خالی_اشاریہ;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_ڈیٹا    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

ساخت ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = خالی_اشاریہ;
    ggml_cgraph              * allocated_گراف      = خالی_اشاریہ;
    ggml_cgraph              * allocated_گراف_copy = خالی_اشاریہ;
    ساخت ggml_context      * ctx_static           = خالی_اشاریہ;
    ساخت ggml_context      * ctx_cpu              = خالی_اشاریہ;
    ساخت ggml_context      * ctx_compute          = خالی_اشاریہ;
    ساخت ggml_context      * ctx_copy             = خالی_اشاریہ;
    ggml_backend_بفر_t      buf_static           = خالی_اشاریہ;
    ggml_backend_بفر_t      buf_cpu              = خالی_اشاریہ;
    std::mt19937               rng;
    شمار ggml_opt_loss_type    loss_type;
    شمار ggml_opt_build_type   build_type;
    شمار ggml_opt_build_type   build_type_alloc;

    ساخت ggml_tensor * inputs  = خالی_اشاریہ;
    ساخت ggml_tensor * outputs = خالی_اشاریہ;
    ساخت ggml_tensor * labels  = خالی_اشاریہ;

    ساخت ggml_tensor * loss     = خالی_اشاریہ;
    ساخت ggml_tensor * pred     = خالی_اشاریہ;
    ساخت ggml_tensor * ncorrect = خالی_اشاریہ;

    ساخت ggml_cgraph * gf      = خالی_اشاریہ;
    ساخت ggml_cgraph * gb_grad = خالی_اشاریہ;
    ساخت ggml_cgraph * gb_opt  = خالی_اشاریہ;
    منطقی static_graphs           = جھوٹ;
    منطقی eval_ready              = جھوٹ;
    std::vector<ساخت ggml_tensor *> grad_accs;
    std::vector<ساخت ggml_tensor *> grad_m;
    std::vector<ساخت ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    منطقی    loss_فی_datapoint = جھوٹ;

    ggml_opt_get_optimizer_params get_opt_pars    = خالی_اشاریہ;
    خالی *                        get_opt_pars_ud = خالی_اشاریہ;
    ساخت ggml_tensor *          opt_قدم_params = خالی_اشاریہ; // Stores output of get_opt_pars.

    شمار ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

ساخت ggml_opt_نتیجہ {
    int64_t              ndata    = 0;
    std::vector<اعشاری>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    منطقی    loss_فی_datapoint = جھوٹ;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        شمار ggml_type type_ڈیٹا,
        شمار ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t نتیجہ = نیا ggml_opt_dataset;
    نتیجہ->ndata       = ndata;
    نتیجہ->ndata_shard = ndata_shard;

    {
        ساخت ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ خالی_اشاریہ,
            /*.no_alloc   =*/ سچ,
        };
        نتیجہ->ctx = ggml_init(params);
    }

    نتیجہ->ڈیٹا = ggml_new_tensor_2d(نتیجہ->ctx, type_ڈیٹا, ne_datapoint, ndata);
    نتیجہ->nbs_ڈیٹا = ggml_nbytes(نتیجہ->ڈیٹا) * ndata_shard/ndata;

    اگر (ne_label > 0) {
        نتیجہ->labels = ggml_new_tensor_2d(نتیجہ->ctx, type_label, ne_label, ndata);
        نتیجہ->nbs_labels = ggml_nbytes(نتیجہ->labels) * ndata_shard/ndata;
    } ورنہ {
        نتیجہ->labels = خالی_اشاریہ;
        نتیجہ->nbs_labels = 0;
    }

    نتیجہ->buf = ggml_backend_alloc_ctx_tensors_from_buft(نتیجہ->ctx, ggml_backend_cpu_بفر_type());

    ثابت int64_t nshards = ndata/ndata_shard;
    نتیجہ->permutation.resize(nshards);
    برائے (int64_t i = 0; i < nshards; ++i) {
        نتیجہ->permutation[i] = i;
    }
    واپس نتیجہ;
}

خالی ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_بفر_free(dataset->buf);
    ggml_free(dataset->ctx);
    حذف dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    واپس dataset->ndata;
}

ساخت ggml_tensor * ggml_opt_dataset_ڈیٹا(ggml_opt_dataset_t dataset) {
    واپس dataset->ڈیٹا;
}

ساخت ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    واپس dataset->labels;
}

خالی ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    اگر (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        واپس;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    ثابت int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

خالی ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, ساخت ggml_tensor * ڈیٹا_batch, ساخت ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   ڈیٹا_batch && ggml_is_contiguous(ڈیٹا_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == خالی_اشاریہ) == (dataset->labels == خالی_اشاریہ));
    GGML_ASSERT(                   ڈیٹا_batch->type == dataset->ڈیٹا->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    ثابت size_t nb_ڈیٹا_batch = ggml_nbytes(ڈیٹا_batch);
    GGML_ASSERT(nb_ڈیٹا_batch % dataset->nbs_ڈیٹا == 0);
    ثابت int64_t shards_فی_batch = nb_ڈیٹا_batch / dataset->nbs_ڈیٹا;

    اگر (labels_batch) {
        ثابت size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_فی_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_فی_batch <= int64_t(dataset->permutation.size()));

    برائے (int64_t ishard_batch = 0; ishard_batch < shards_فی_batch; ++ishard_batch) {
        ثابت int64_t ishard = dataset->permutation[ibatch*shards_فی_batch + ishard_batch];

        ثابت حرف * ptr_ڈیٹا = (ثابت حرف *) dataset->ڈیٹا->ڈیٹا + ishard*dataset->nbs_ڈیٹا;
        ggml_backend_tensor_سیٹ(ڈیٹا_batch, ptr_ڈیٹا, ishard_batch*dataset->nbs_ڈیٹا, dataset->nbs_ڈیٹا);

        اگر (!labels_batch) {
            جاری;
        }

        ثابت حرف * ptr_labels = (ثابت حرف *) dataset->labels->ڈیٹا + ishard*dataset->nbs_labels;
        ggml_backend_tensor_سیٹ(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

خالی ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, خالی * ڈیٹا_batch, size_t nb_ڈیٹا_batch, خالی * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == خالی_اشاریہ) == (dataset->labels == خالی_اشاریہ));
    GGML_ASSERT(nb_ڈیٹا_batch % dataset->nbs_ڈیٹا == 0);

    ثابت int64_t shards_فی_batch = nb_ڈیٹا_batch / dataset->nbs_ڈیٹا;

    GGML_ASSERT((ibatch + 1)*shards_فی_batch <= int64_t(dataset->permutation.size()));

    برائے (int64_t ishard_batch = 0; ishard_batch < shards_فی_batch; ++ishard_batch) {
        ثابت int64_t ishard = dataset->permutation[ibatch*shards_فی_batch + ishard_batch];

        ثابت حرف * ptr_ڈیٹا       = (ثابت حرف *) dataset->ڈیٹا->ڈیٹا + ishard      *dataset->nbs_ڈیٹا;
        حرف       * ptr_ڈیٹا_batch = (حرف       *) ڈیٹا_batch          + ishard_batch*dataset->nbs_ڈیٹا;
        حافظہ_نقل(ptr_ڈیٹا_batch, ptr_ڈیٹا, dataset->nbs_ڈیٹا);

        اگر (!labels_batch) {
            جاری;
        }

        ثابت حرف * ptr_labels       = (ثابت حرف *) dataset->labels->ڈیٹا + ishard      *dataset->nbs_labels;
        حرف       * ptr_labels_batch = (حرف       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        حافظہ_نقل(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

ساخت ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(خالی * userdata) {
    GGML_غیر_مستعمل(userdata);

    ggml_opt_optimizer_params نتیجہ;

    نتیجہ.adamw.alpha = 0.001f;
    نتیجہ.adamw.beta1 = 0.9f;
    نتیجہ.adamw.beta2 = 0.999f;
    نتیجہ.adamw.eps   = 1e-8f;
    نتیجہ.adamw.wd    = 0.0f;

    نتیجہ.sgd.alpha   = 1e-3f;
    نتیجہ.sgd.wd      = 0.0f;

    واپس نتیجہ;
}


ساخت ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(خالی * userdata) {
    واپس *((ساخت ggml_opt_optimizer_params *) userdata);
}

ساخت ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        شمار ggml_opt_loss_type   loss_type) {
    واپس {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ خالی_اشاریہ,
        /*inputs          =*/ خالی_اشاریہ,
        /*logits          =*/ خالی_اشاریہ,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ خالی_اشاریہ,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

ساکن ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    اگر (!tensor) {
        واپس خالی_اشاریہ;
    }

    اگر (tensor_map.find(tensor) != tensor_map.end()) {
        واپس tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    حافظہ_نقل(new_tensor->op_params, tensor->op_params, حجم(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->ڈیٹا = tensor->ڈیٹا;
    new_tensor->بفر = tensor->بفر;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    واپس new_tensor;
}

ساکن ggml_cgraph * dup_گراف(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_گراف_custom(ctx, src->size, /*grads =*/ سچ);

    برائے (صحیح i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    برائے (صحیح i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    برائے (صحیح i = 0; i < src->n_nodes; ++i) {
        ثابت size_t igrad_src = ggml_hash_ڈھونڈو(&src->دیکھا_hash_سیٹ, src->nodes[i]);
        ثابت size_t igrad_dst = ggml_hash_ڈھونڈو(&dst->دیکھا_hash_سیٹ, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_بھرا);
        GGML_ASSERT(ggml_bitset_get(src->دیکھا_hash_سیٹ.مستعمل, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_بھرا);
        GGML_ASSERT(ggml_bitset_get(dst->دیکھا_hash_سیٹ.مستعمل, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    واپس dst;
}

ساکن خالی ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->ڈیٹا) && "when using static graphs the inputs must be allocated statically");

    ثابت شمار ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    ثابت منطقی accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    ثابت منطقی need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_سیٹ_داخلہ(opt_ctx->inputs);
    ggml_سیٹ_output(opt_ctx->outputs);

    صحیح n_param = 0;
    برائے (صحیح i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        ثابت ساخت ggml_tensor * گرہ = opt_ctx->gf->nodes[i];
        اگر (گرہ->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(گرہ->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    اگر (!opt_ctx->ctx_static) {
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
        ثابت size_t سائز_meta = (n_loss + tensors_فی_param*n_param + tensors_const) * ggml_tensor_overhead();
        ساخت ggml_init_params params = {
            /*.mem_size   =*/ سائز_meta,
            /*.mem_buffer =*/ خالی_اشاریہ,
            /*.no_alloc   =*/ سچ,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        ثابت size_t سائز_meta = 1 * ggml_tensor_overhead();
        ساخت ggml_init_params params = {
            /*.mem_size   =*/ سائز_meta,
            /*.mem_buffer =*/ خالی_اشاریہ,
            /*.no_alloc   =*/ سچ,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_بفر_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = خالی_اشاریہ;
    }

    ساخت ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    انتخاب (opt_ctx->loss_type) {
        حالت GGML_OPT_LOSS_TYPE_اوسط: {
            opt_ctx->loss = ggml_مجموع(ctx_results, opt_ctx->outputs);
            ggml_سیٹ_name(opt_ctx->loss, "loss_sum");
            ثابت اعشاری scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_سیٹ_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_فی_datapoint = سچ;
            توڑو;
        }
        حالت GGML_OPT_LOSS_TYPE_مجموع: {
            opt_ctx->loss = ggml_مجموع(ctx_results, opt_ctx->outputs);
            ggml_سیٹ_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_فی_datapoint = جھوٹ;
            توڑو;
        }
        حالت GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_سیٹ_داخلہ(opt_ctx->labels);
            ggml_سیٹ_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_سیٹ_name(opt_ctx->loss, "loss_cross_entropy");
            اگر (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_سیٹ_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_فی_datapoint = سچ;
            توڑو;
        }
        حالت GGML_OPT_LOSS_TYPE_اوسط_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_سیٹ_داخلہ(opt_ctx->labels);
            ggml_سیٹ_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_سیٹ_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_سیٹ_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_مجموع(ctx_results, opt_ctx->loss);
            ggml_سیٹ_name(opt_ctx->loss, "loss_sum_squared_error");
            ثابت اعشاری scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_سیٹ_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_فی_datapoint = سچ;
            توڑو;
        }
    }
    ggml_سیٹ_output(opt_ctx->loss);
    ggml_سیٹ_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    اگر (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_سیٹ_name(opt_ctx->pred, "pred");
        ggml_سیٹ_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_گنتی_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_سیٹ_name(opt_ctx->ncorrect, "ncorrect");
        ggml_سیٹ_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    اگر (opt_ctx->buf_static) {
        اگر (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            واپس;
        }
    } ورنہ اگر (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        واپس;
    }

    اگر (opt_ctx->grad_accs.خالی_خانے()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        ثابت صحیح n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        برائے (صحیح i = 0; i < n_nodes; ++i) {
            ggml_tensor * گرہ = opt_ctx->gf->nodes[i];
            اگر ((accumulate && (گرہ->flags & GGML_TENSOR_FLAG_PARAM)) || (گرہ->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, گرہ->ne);
            } ورنہ {
                opt_ctx->grad_accs[i] = خالی_اشاریہ;
            }
        }

        اگر (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            برائے (صحیح i = 0; i < n_nodes; ++i) {
                ggml_tensor * گرہ = opt_ctx->gf->nodes[i];
                اگر (گرہ->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, گرہ->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, گرہ->ne);
                } ورنہ {
                    opt_ctx->grad_m[i] = خالی_اشاریہ;
                    opt_ctx->grad_v[i] = خالی_اشاریہ;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_گراف_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ سچ);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.ڈیٹا());

    اگر (opt_ctx->buf_static) {
        اگر (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            واپس;
        }
    } ورنہ اگر (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
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
    برائے (صحیح i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        ساخت ggml_tensor * گرہ = opt_ctx->gb_opt->nodes[i];
        ساخت ggml_tensor * grad = ggml_گراف_get_grad(opt_ctx->gb_opt, گرہ);

        اگر (grad && (گرہ->flags & GGML_TENSOR_FLAG_PARAM)) {
            ساخت ggml_tensor * m = خالی_اشاریہ;
            ساخت ggml_tensor * v = خالی_اشاریہ;
            اگر (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", گرہ->name);
                ggml_format_name(v, "AdamW v for %s", گرہ->name);
            }
            ساخت ggml_tensor * opt_قدم;
            انتخاب (optimizer) {
                حالت GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_قدم = ggml_opt_قدم_adamw(opt_ctx->ctx_compute, گرہ, grad, m, v, adamw_params);
                    توڑو;
                حالت GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_قدم = ggml_opt_قدم_sgd(opt_ctx->ctx_compute, گرہ, grad, adamw_params);
                    توڑو;
                طےشدہ:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_قدم, "%s step for %s", optimizer_name, گرہ->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_قدم);
        }
    }

    اگر (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_گراف_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_بفر_type());
}

ggml_opt_context_t ggml_opt_init(ساخت ggml_opt_params params) {
    ggml_opt_context_t نتیجہ = نیا ساخت ggml_opt_context;
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

    اگر (!نتیجہ->static_graphs) {
        GGML_ASSERT(!نتیجہ->inputs);
        GGML_ASSERT(!نتیجہ->outputs);
        واپس نتیجہ;
    }

    GGML_ASSERT(نتیجہ->inputs);
    GGML_ASSERT(نتیجہ->outputs);

    نتیجہ->gf = ggml_new_گراف_custom(نتیجہ->ctx_compute, GGML_DEFAULT_گراف_سائز, /*grads =*/ سچ); // Forward pass.
    ggml_build_forward_expand(نتیجہ->gf, نتیجہ->outputs);

    ggml_opt_build(نتیجہ);

    واپس نتیجہ;
}

خالی ggml_opt_free(ggml_opt_context_t opt_ctx) {
    اگر (opt_ctx == خالی_اشاریہ) {
        واپس;
    }
    ggml_backend_بفر_free(opt_ctx->buf_static);
    ggml_backend_بفر_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    حذف opt_ctx;
}

خالی ggml_opt_reset(ggml_opt_context_t opt_ctx, منطقی optimizer) {
    اگر (optimizer) {
        ggml_گراف_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } ورنہ {
        ggml_گراف_reset(opt_ctx->gb_grad);
    }
}

منطقی ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    واپس opt_ctx->static_graphs;
}

ساخت ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    واپس opt_ctx->inputs;
}

ساخت ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    واپس opt_ctx->outputs;
}

ساخت ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    واپس opt_ctx->labels;
}

ساخت ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    واپس opt_ctx->loss;
}

ساخت ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    واپس opt_ctx->pred;
}

ساخت ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    واپس opt_ctx->ncorrect;
}

ساخت ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, ساخت ggml_tensor * گرہ) {
    واپس ggml_گراف_get_grad_acc(opt_ctx->gb_opt, گرہ);
}

// ====== Optimization Result ======

ggml_opt_نتیجہ_t ggml_opt_نتیجہ_init() {
    واپس نیا ggml_opt_نتیجہ;
}

خالی ggml_opt_نتیجہ_free(ggml_opt_نتیجہ_t نتیجہ) {
    حذف نتیجہ;
}

خالی ggml_opt_نتیجہ_reset(ggml_opt_نتیجہ_t نتیجہ) {
    نتیجہ->ndata = 0;
    نتیجہ->loss.clear();
    نتیجہ->pred.clear();
    نتیجہ->ncorrect = 0;
}

خالی ggml_opt_نتیجہ_ndata(ggml_opt_نتیجہ_t نتیجہ, int64_t * ndata) {
    *ndata = نتیجہ->ndata;
}

خالی ggml_opt_نتیجہ_loss(ggml_opt_نتیجہ_t نتیجہ, دوہرا * loss, دوہرا * unc) {
    ثابت int64_t nbatches = نتیجہ->loss.size(); // Number of physical batches.

    اگر (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        واپس;
    }

    دوہرا مجموع         = 0.0;
    دوہرا مجموع_squared = 0.0;

    برائے (ثابت اعشاری & loss : نتیجہ->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        ثابت اعشاری loss_scaled = نتیجہ->loss_فی_datapoint ? loss*نتیجہ->opt_period : loss;
        مجموع         += loss_scaled;
        مجموع_squared += loss_scaled*loss_scaled;
    }

    ثابت دوہرا اوسط_2 = مجموع/nbatches;
    *loss = نتیجہ->loss_فی_datapoint ? اوسط_2 : مجموع;

    اگر (!unc) {
        واپس;
    }

    اگر (nbatches < 2) {
        *unc = NAN;
        واپس;
    }

    ثابت دوہرا var_مجموع = مجموع_squared/nbatches - اوسط_2*اوسط_2; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = نتیجہ->loss_فی_datapoint ? جذر(var_مجموع / (nbatches - 1)) : جذر(var_مجموع * nbatches/(nbatches - 1));
}

خالی ggml_opt_نتیجہ_pred(ggml_opt_نتیجہ_t نتیجہ, int32_t * pred) {
    برائے (size_t i = 0; i < نتیجہ->pred.size(); ++i) {
        pred[i] = نتیجہ->pred[i];
    }
}

خالی ggml_opt_نتیجہ_accuracy(ggml_opt_نتیجہ_t نتیجہ, دوہرا * accuracy, دوہرا * unc) {
    *accuracy = نتیجہ->ncorrect >= 0 ? دوہرا(نتیجہ->ncorrect) / دوہرا(نتیجہ->ndata) : NAN;

    اگر (!unc) {
        واپس;
    }

    *unc = نتیجہ->ncorrect >= 0 && نتیجہ->ndata >= 2 ?
        جذر((*accuracy) * (1.0 - (*accuracy)) / دوہرا(نتیجہ->ndata - 1)) : NAN;
}

// ====== Computation ======

خالی ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        ساخت ggml_context * ctx_compute,
        ساخت ggml_cgraph  * gf,
        ساخت ggml_tensor  * inputs,
        ساخت ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

خالی ggml_opt_alloc(ggml_opt_context_t opt_ctx, منطقی backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    اگر (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_گراف_reset(opt_ctx->gb_grad);
    }
    اگر (backward) {
        ثابت int32_t opt_i_اگلا = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_اگلا == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } ورنہ {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    اگر (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    ساخت ggml_cgraph * graph = خالی_اشاریہ;
    انتخاب (opt_ctx->build_type) {
        حالت GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } توڑو;
        حالت GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } توڑو;
        حالت GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } توڑو;
    }
    GGML_ASSERT(graph);

    اگر (opt_ctx->allocated_گراف == graph) {
        opt_ctx->eval_ready = سچ;
        واپس;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    اگر (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_گراف_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ خالی_اشاریہ,
            /*.no_alloc   =*/ سچ,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_گراف_copy = dup_گراف(opt_ctx->ctx_copy, graph);
    } ورنہ {
        opt_ctx->allocated_گراف_copy = graph;
    }

    ggml_backend_sched_alloc_گراف(opt_ctx->backend_sched, opt_ctx->allocated_گراف_copy);
    opt_ctx->allocated_گراف = graph;

    opt_ctx->eval_ready = سچ;
}

خالی ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_نتیجہ_t نتیجہ) {
    GGML_ASSERT(opt_ctx->eval_ready);
    اگر (opt_ctx->allocated_گراف == opt_ctx->gb_opt) {
        ثابت ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        انتخاب (opt_ctx->optimizer) {
            حالت GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                ثابت اعشاری beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                ثابت اعشاری beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                اعشاری * adamw_par_ڈیٹا = ggml_get_ڈیٹا_f32(opt_ctx->opt_قدم_params);
                adamw_par_ڈیٹا[0] = opt_pars.adamw.alpha;
                adamw_par_ڈیٹا[1] = opt_pars.adamw.beta1;
                adamw_par_ڈیٹا[2] = opt_pars.adamw.beta2;
                adamw_par_ڈیٹا[3] = opt_pars.adamw.eps;
                adamw_par_ڈیٹا[4] = opt_pars.adamw.wd;
                adamw_par_ڈیٹا[5] = beta1h;
                adamw_par_ڈیٹا[6] = beta2h;
            } توڑو;
            حالت GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                اعشاری * sgd = ggml_get_ڈیٹا_f32(opt_ctx->opt_قدم_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } توڑو;
            طےشدہ:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_گراف_compute(opt_ctx->backend_sched, opt_ctx->allocated_گراف_copy);
    opt_ctx->iter += opt_ctx->allocated_گراف == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    اگر (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = خالی_اشاریہ;
        opt_ctx->gb_grad              = خالی_اشاریہ;
        opt_ctx->gb_opt               = خالی_اشاریہ;
        opt_ctx->allocated_گراف      = خالی_اشاریہ;
        opt_ctx->allocated_گراف_copy = خالی_اشاریہ;
    }

    opt_ctx->eval_ready = جھوٹ;

    اگر (!نتیجہ) {
        واپس;
    }

    اگر (نتیجہ->ndata == 0) {
        نتیجہ->loss_فی_datapoint = opt_ctx->loss_فی_datapoint;
        نتیجہ->opt_period         = opt_ctx->opt_period;
    } ورنہ {
        GGML_ASSERT(نتیجہ->loss_فی_datapoint == opt_ctx->loss_فی_datapoint);
        GGML_ASSERT(نتیجہ->opt_period         == opt_ctx->opt_period);
    }

    ثابت int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(نتیجہ->ndata == ndata*int64_t(نتیجہ->loss.size()) && "varying batch size not supported");
    نتیجہ->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    اعشاری loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    نتیجہ->loss.push_back(loss);

    اگر (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.ڈیٹا(), 0, ggml_nbytes(opt_ctx->pred));
        نتیجہ->pred.insert(نتیجہ->pred.end(), pred.begin(), pred.end());
    }

    اگر (!opt_ctx->ncorrect || نتیجہ->ncorrect < 0) {
        نتیجہ->ncorrect = -1;
        واپس;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    نتیجہ->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

خالی ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_نتیجہ_t       نتیجہ_train,
        ggml_opt_نتیجہ_t       نتیجہ_eval,
        int64_t                 idata_بانٹو,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    ساخت ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    ساخت ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    ساخت ggml_tensor * ڈیٹا   = ggml_opt_dataset_ڈیٹا(dataset);
    GGML_ASSERT(ڈیٹا->ne[0] == inputs->ne[0]);

    ثابت int64_t ndata       =   ڈیٹا->ne[1];
    ثابت int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(ڈیٹا->ne[1] % inputs->ne[1] == 0);
    ثابت int64_t nbatches = ndata/ndata_batch;

    idata_بانٹو = idata_بانٹو < 0 ? ndata : idata_بانٹو;
    GGML_ASSERT(idata_بانٹو % ndata_batch == 0);
    ثابت int64_t ibatch_بانٹو = idata_بانٹو / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_آغاز = ggml_وقت_us();
    برائے (; ibatch < ibatch_بانٹو; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ سچ);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, نتیجہ_train);
        اگر (callback_train) {
            callback_train(سچ, opt_ctx, dataset, نتیجہ_train, ibatch+1, ibatch_بانٹو, t_loop_آغاز);
        }
    }
    t_loop_آغاز = ggml_وقت_us();
    برائے (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ جھوٹ);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, نتیجہ_eval);
        اگر (callback_eval) {
            callback_eval(جھوٹ, opt_ctx, dataset, نتیجہ_eval, ibatch+1-ibatch_بانٹو, nbatches-ibatch_بانٹو, t_loop_آغاز);
        }
    }
}

خالی ggml_opt_epoch_callback_پیش_رفت_bar(
        منطقی               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_نتیجہ_t  نتیجہ,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_آغاز_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_لمبائی = 8;
    ثابت int64_t ibatch8 = 8 * ibatch;
    برائے (int64_t j = 0; j < bar_لمبائی; ++j) {
        اگر        (ibatch_max * (8*j + 8) / bar_لمبائی < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } ورنہ اگر (ibatch_max * (8*j + 7) / bar_لمبائی < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } ورنہ اگر (ibatch_max * (8*j + 6) / bar_لمبائی < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } ورنہ اگر (ibatch_max * (8*j + 5) / bar_لمبائی < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } ورنہ اگر (ibatch_max * (8*j + 4) / bar_لمبائی < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } ورنہ اگر (ibatch_max * (8*j + 3) / bar_لمبائی < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } ورنہ اگر (ibatch_max * (8*j + 2) / bar_لمبائی < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } ورنہ اگر (ibatch_max * (8*j + 1) / bar_لمبائی < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } ورنہ {
            fprintf(stderr, " ");
        }
    }

    ثابت int64_t batch_سائز = ggml_opt_inputs(opt_ctx)->ne[1];
    ثابت int64_t idata      = ibatch*batch_سائز;
    ثابت int64_t idata_max  = ibatch_max*batch_سائز;

    دوہرا loss;
    دوہرا loss_unc;
    ggml_opt_نتیجہ_loss(نتیجہ, &loss, &loss_unc);

    دوہرا accuracy;
    دوہرا accuracy_unc;
    ggml_opt_نتیجہ_accuracy(نتیجہ, &accuracy, &accuracy_unc);

    ثابت int64_t t_ibatch_us = ggml_وقت_us() - t_آغاز_us;
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
    اگر (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_غیر_مستعمل(dataset);
}

خالی ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        شمار ggml_opt_loss_type         loss_type,
        شمار ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        اعشاری                           val_بانٹو,
        منطقی                            silent) {
    ggml_وقت_init();
    ثابت int64_t t_آغاز_us = ggml_وقت_us();

    ثابت int64_t ndata           = ggml_opt_dataset_ڈیٹا(dataset)->ne[1];
    ثابت int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    ثابت int64_t opt_period       = nbatch_logical / nbatch_physical;
    ثابت int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_بانٹو >= 0.0f);
    GGML_ASSERT(val_بانٹو <  1.0f);
    ثابت int64_t ibatch_بانٹو = int64_t(((1.0f - val_بانٹو) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    ثابت int64_t idata_بانٹو  = ibatch_بانٹو * nbatch_physical;

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
    اگر (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_نتیجہ_t نتیجہ_train = ggml_opt_نتیجہ_init();
    ggml_opt_نتیجہ_t نتیجہ_val   = ggml_opt_نتیجہ_init();

    ggml_opt_epoch_callback epoch_callback = silent ? خالی_اشاریہ : ggml_opt_epoch_callback_پیش_رفت_bar;

    برائے (; epoch <= nepoch; ++epoch) {
        اگر (nbatch_logical < idata_بانٹو) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_بانٹو);
        }

        ggml_opt_نتیجہ_reset(نتیجہ_train);
        ggml_opt_نتیجہ_reset(نتیجہ_val);

        اگر (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, نتیجہ_train, نتیجہ_val, idata_بانٹو, epoch_callback, epoch_callback);
        اگر (!silent) {
            fprintf(stderr, "\n");
        }
    }

    اگر (!silent) {
        int64_t t_کل_s = (ggml_وقت_us() - t_آغاز_us) / 1000000;
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

شمار ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    واپس c->optimizer;
}

GGML_API ثابت حرف * ggml_opt_optimizer_name(شمار ggml_opt_optimizer_type o) {
    انتخاب (o) {
        حالت GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            واپس "adamw";
        حالت GGML_OPT_OPTIMIZER_TYPE_SGD:
            واپس "sgd";
        طےشدہ:
            واپس "undefined";
    };
}
