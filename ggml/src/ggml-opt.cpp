#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-opt.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-alloc.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cinttypes>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <random>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਬਣਤਰ ggml_opt_dataset {
    ਬਣਤਰ ggml_context   * ctx    = ਖਾਲੀ_ਸੂਚਕ;
    ggml_backend_ਬਫਰ_t   buf    = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_tensor    * ਅੰਕੜੇ   = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_tensor    * labels = ਖਾਲੀ_ਸੂਚਕ;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_ਅੰਕੜੇ    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

ਬਣਤਰ ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = ਖਾਲੀ_ਸੂਚਕ;
    ggml_cgraph              * allocated_ਗ੍ਰਾਫ      = ਖਾਲੀ_ਸੂਚਕ;
    ggml_cgraph              * allocated_ਗ੍ਰਾਫ_copy = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_context      * ctx_static           = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_context      * ctx_cpu              = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_context      * ctx_compute          = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_context      * ctx_copy             = ਖਾਲੀ_ਸੂਚਕ;
    ggml_backend_ਬਫਰ_t      buf_static           = ਖਾਲੀ_ਸੂਚਕ;
    ggml_backend_ਬਫਰ_t      buf_cpu              = ਖਾਲੀ_ਸੂਚਕ;
    std::mt19937               rng;
    ਗਿਣਤੀ ggml_opt_loss_type    loss_type;
    ਗਿਣਤੀ ggml_opt_build_type   build_type;
    ਗਿਣਤੀ ggml_opt_build_type   build_type_alloc;

    ਬਣਤਰ ggml_tensor * inputs  = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_tensor * outputs = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_tensor * labels  = ਖਾਲੀ_ਸੂਚਕ;

    ਬਣਤਰ ggml_tensor * loss     = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_tensor * pred     = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_tensor * ncorrect = ਖਾਲੀ_ਸੂਚਕ;

    ਬਣਤਰ ggml_cgraph * gf      = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_cgraph * gb_grad = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_cgraph * gb_opt  = ਖਾਲੀ_ਸੂਚਕ;
    ਤਾਰਕਿਕ static_graphs           = ਝੂਠ;
    ਤਾਰਕਿਕ eval_ready              = ਝੂਠ;
    std::vector<ਬਣਤਰ ggml_tensor *> grad_accs;
    std::vector<ਬਣਤਰ ggml_tensor *> grad_m;
    std::vector<ਬਣਤਰ ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    ਤਾਰਕਿਕ    loss_ਪ੍ਰਤੀ_datapoint = ਝੂਠ;

    ggml_opt_get_optimizer_params get_opt_pars    = ਖਾਲੀ_ਸੂਚਕ;
    ਖਾਲੀ *                        get_opt_pars_ud = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ggml_tensor *          opt_ਕਦਮ_params = ਖਾਲੀ_ਸੂਚਕ; // Stores output of get_opt_pars.

    ਗਿਣਤੀ ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

ਬਣਤਰ ggml_opt_ਨਤੀਜਾ {
    int64_t              ndata    = 0;
    std::vector<ਦਸ਼ਮਲਵ>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    ਤਾਰਕਿਕ    loss_ਪ੍ਰਤੀ_datapoint = ਝੂਠ;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        ਗਿਣਤੀ ggml_type type_ਅੰਕੜੇ,
        ਗਿਣਤੀ ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t ਨਤੀਜਾ = ਨਵਾਂ ggml_opt_dataset;
    ਨਤੀਜਾ->ndata       = ndata;
    ਨਤੀਜਾ->ndata_shard = ndata_shard;

    {
        ਬਣਤਰ ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ ਖਾਲੀ_ਸੂਚਕ,
            /*.no_alloc   =*/ ਸੱਚ,
        };
        ਨਤੀਜਾ->ctx = ggml_init(params);
    }

    ਨਤੀਜਾ->ਅੰਕੜੇ = ggml_new_tensor_2d(ਨਤੀਜਾ->ctx, type_ਅੰਕੜੇ, ne_datapoint, ndata);
    ਨਤੀਜਾ->nbs_ਅੰਕੜੇ = ggml_nbytes(ਨਤੀਜਾ->ਅੰਕੜੇ) * ndata_shard/ndata;

    ਜੇ (ne_label > 0) {
        ਨਤੀਜਾ->labels = ggml_new_tensor_2d(ਨਤੀਜਾ->ctx, type_label, ne_label, ndata);
        ਨਤੀਜਾ->nbs_labels = ggml_nbytes(ਨਤੀਜਾ->labels) * ndata_shard/ndata;
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ->labels = ਖਾਲੀ_ਸੂਚਕ;
        ਨਤੀਜਾ->nbs_labels = 0;
    }

    ਨਤੀਜਾ->buf = ggml_backend_alloc_ctx_tensors_from_buft(ਨਤੀਜਾ->ctx, ggml_backend_cpu_ਬਫਰ_type());

    ਅਚਲ int64_t nshards = ndata/ndata_shard;
    ਨਤੀਜਾ->permutation.resize(nshards);
    ਲਈ (int64_t i = 0; i < nshards; ++i) {
        ਨਤੀਜਾ->permutation[i] = i;
    }
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਖਾਲੀ ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_ਬਫਰ_free(dataset->buf);
    ggml_free(dataset->ctx);
    ਮਿਟਾਓ dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    ਵਾਪਸ dataset->ndata;
}

ਬਣਤਰ ggml_tensor * ggml_opt_dataset_ਅੰਕੜੇ(ggml_opt_dataset_t dataset) {
    ਵਾਪਸ dataset->ਅੰਕੜੇ;
}

ਬਣਤਰ ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    ਵਾਪਸ dataset->labels;
}

ਖਾਲੀ ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    ਜੇ (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        ਵਾਪਸ;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    ਅਚਲ int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

ਖਾਲੀ ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, ਬਣਤਰ ggml_tensor * ਅੰਕੜੇ_batch, ਬਣਤਰ ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   ਅੰਕੜੇ_batch && ggml_is_contiguous(ਅੰਕੜੇ_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == ਖਾਲੀ_ਸੂਚਕ) == (dataset->labels == ਖਾਲੀ_ਸੂਚਕ));
    GGML_ASSERT(                   ਅੰਕੜੇ_batch->type == dataset->ਅੰਕੜੇ->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    ਅਚਲ size_t nb_ਅੰਕੜੇ_batch = ggml_nbytes(ਅੰਕੜੇ_batch);
    GGML_ASSERT(nb_ਅੰਕੜੇ_batch % dataset->nbs_ਅੰਕੜੇ == 0);
    ਅਚਲ int64_t shards_ਪ੍ਰਤੀ_batch = nb_ਅੰਕੜੇ_batch / dataset->nbs_ਅੰਕੜੇ;

    ਜੇ (labels_batch) {
        ਅਚਲ size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_ਪ੍ਰਤੀ_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_ਪ੍ਰਤੀ_batch <= int64_t(dataset->permutation.size()));

    ਲਈ (int64_t ishard_batch = 0; ishard_batch < shards_ਪ੍ਰਤੀ_batch; ++ishard_batch) {
        ਅਚਲ int64_t ishard = dataset->permutation[ibatch*shards_ਪ੍ਰਤੀ_batch + ishard_batch];

        ਅਚਲ ਅੱਖਰ * ptr_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *) dataset->ਅੰਕੜੇ->ਅੰਕੜੇ + ishard*dataset->nbs_ਅੰਕੜੇ;
        ggml_backend_tensor_ਸੈੱਟ(ਅੰਕੜੇ_batch, ptr_ਅੰਕੜੇ, ishard_batch*dataset->nbs_ਅੰਕੜੇ, dataset->nbs_ਅੰਕੜੇ);

        ਜੇ (!labels_batch) {
            ਜਾਰੀ;
        }

        ਅਚਲ ਅੱਖਰ * ptr_labels = (ਅਚਲ ਅੱਖਰ *) dataset->labels->ਅੰਕੜੇ + ishard*dataset->nbs_labels;
        ggml_backend_tensor_ਸੈੱਟ(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

ਖਾਲੀ ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, ਖਾਲੀ * ਅੰਕੜੇ_batch, size_t nb_ਅੰਕੜੇ_batch, ਖਾਲੀ * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == ਖਾਲੀ_ਸੂਚਕ) == (dataset->labels == ਖਾਲੀ_ਸੂਚਕ));
    GGML_ASSERT(nb_ਅੰਕੜੇ_batch % dataset->nbs_ਅੰਕੜੇ == 0);

    ਅਚਲ int64_t shards_ਪ੍ਰਤੀ_batch = nb_ਅੰਕੜੇ_batch / dataset->nbs_ਅੰਕੜੇ;

    GGML_ASSERT((ibatch + 1)*shards_ਪ੍ਰਤੀ_batch <= int64_t(dataset->permutation.size()));

    ਲਈ (int64_t ishard_batch = 0; ishard_batch < shards_ਪ੍ਰਤੀ_batch; ++ishard_batch) {
        ਅਚਲ int64_t ishard = dataset->permutation[ibatch*shards_ਪ੍ਰਤੀ_batch + ishard_batch];

        ਅਚਲ ਅੱਖਰ * ptr_ਅੰਕੜੇ       = (ਅਚਲ ਅੱਖਰ *) dataset->ਅੰਕੜੇ->ਅੰਕੜੇ + ishard      *dataset->nbs_ਅੰਕੜੇ;
        ਅੱਖਰ       * ptr_ਅੰਕੜੇ_batch = (ਅੱਖਰ       *) ਅੰਕੜੇ_batch          + ishard_batch*dataset->nbs_ਅੰਕੜੇ;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ptr_ਅੰਕੜੇ_batch, ptr_ਅੰਕੜੇ, dataset->nbs_ਅੰਕੜੇ);

        ਜੇ (!labels_batch) {
            ਜਾਰੀ;
        }

        ਅਚਲ ਅੱਖਰ * ptr_labels       = (ਅਚਲ ਅੱਖਰ *) dataset->labels->ਅੰਕੜੇ + ishard      *dataset->nbs_labels;
        ਅੱਖਰ       * ptr_labels_batch = (ਅੱਖਰ       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

ਬਣਤਰ ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(ਖਾਲੀ * userdata) {
    GGML_ਅਣਵਰਤਿਆ(userdata);

    ggml_opt_optimizer_params ਨਤੀਜਾ;

    ਨਤੀਜਾ.adamw.alpha = 0.001f;
    ਨਤੀਜਾ.adamw.beta1 = 0.9f;
    ਨਤੀਜਾ.adamw.beta2 = 0.999f;
    ਨਤੀਜਾ.adamw.eps   = 1e-8f;
    ਨਤੀਜਾ.adamw.wd    = 0.0f;

    ਨਤੀਜਾ.sgd.alpha   = 1e-3f;
    ਨਤੀਜਾ.sgd.wd      = 0.0f;

    ਵਾਪਸ ਨਤੀਜਾ;
}


ਬਣਤਰ ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(ਖਾਲੀ * userdata) {
    ਵਾਪਸ *((ਬਣਤਰ ggml_opt_optimizer_params *) userdata);
}

ਬਣਤਰ ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        ਗਿਣਤੀ ggml_opt_loss_type   loss_type) {
    ਵਾਪਸ {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ ਖਾਲੀ_ਸੂਚਕ,
        /*inputs          =*/ ਖਾਲੀ_ਸੂਚਕ,
        /*logits          =*/ ਖਾਲੀ_ਸੂਚਕ,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ ਖਾਲੀ_ਸੂਚਕ,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

ਸਥਿਰ ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    ਜੇ (!tensor) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਜੇ (tensor_map.find(tensor) != tensor_map.end()) {
        ਵਾਪਸ tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(new_tensor->op_params, tensor->op_params, ਆਕਾਰ(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->ਅੰਕੜੇ = tensor->ਅੰਕੜੇ;
    new_tensor->ਬਫਰ = tensor->ਬਫਰ;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    ਵਾਪਸ new_tensor;
}

ਸਥਿਰ ggml_cgraph * dup_ਗ੍ਰਾਫ(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_ਗ੍ਰਾਫ_custom(ctx, src->size, /*grads =*/ ਸੱਚ);

    ਲਈ (ਪੂਰਨ i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    ਲਈ (ਪੂਰਨ i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    ਲਈ (ਪੂਰਨ i = 0; i < src->n_nodes; ++i) {
        ਅਚਲ size_t igrad_src = ggml_hash_ਲੱਭੋ(&src->ਦੇਖਿਆ_hash_ਸੈੱਟ, src->nodes[i]);
        ਅਚਲ size_t igrad_dst = ggml_hash_ਲੱਭੋ(&dst->ਦੇਖਿਆ_hash_ਸੈੱਟ, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_ਭਰਿਆ);
        GGML_ASSERT(ggml_bitset_get(src->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_ਭਰਿਆ);
        GGML_ASSERT(ggml_bitset_get(dst->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    ਵਾਪਸ dst;
}

ਸਥਿਰ ਖਾਲੀ ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->ਅੰਕੜੇ) && "when using static graphs the inputs must be allocated statically");

    ਅਚਲ ਗਿਣਤੀ ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    ਅਚਲ ਤਾਰਕਿਕ accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    ਅਚਲ ਤਾਰਕਿਕ need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_ਸੈੱਟ_ਨਿਵੇਸ਼(opt_ctx->inputs);
    ggml_ਸੈੱਟ_output(opt_ctx->outputs);

    ਪੂਰਨ n_param = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        ਅਚਲ ਬਣਤਰ ggml_tensor * ਗੰਢ = opt_ctx->gf->nodes[i];
        ਜੇ (ਗੰਢ->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(ਗੰਢ->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    ਜੇ (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        ਅਚਲ size_t tensors_ਪ੍ਰਤੀ_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        ਅਚਲ size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        ਅਚਲ size_t ਮਾਪ_meta = (n_loss + tensors_ਪ੍ਰਤੀ_param*n_param + tensors_const) * ggml_tensor_overhead();
        ਬਣਤਰ ggml_init_params params = {
            /*.mem_size   =*/ ਮਾਪ_meta,
            /*.mem_buffer =*/ ਖਾਲੀ_ਸੂਚਕ,
            /*.no_alloc   =*/ ਸੱਚ,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        ਅਚਲ size_t ਮਾਪ_meta = 1 * ggml_tensor_overhead();
        ਬਣਤਰ ggml_init_params params = {
            /*.mem_size   =*/ ਮਾਪ_meta,
            /*.mem_buffer =*/ ਖਾਲੀ_ਸੂਚਕ,
            /*.no_alloc   =*/ ਸੱਚ,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_ਬਫਰ_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = ਖਾਲੀ_ਸੂਚਕ;
    }

    ਬਣਤਰ ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    ਚੋਣ (opt_ctx->loss_type) {
        ਹਾਲਤ GGML_OPT_LOSS_TYPE_ਔਸਤ: {
            opt_ctx->loss = ggml_ਜੋੜ(ctx_results, opt_ctx->outputs);
            ggml_ਸੈੱਟ_name(opt_ctx->loss, "loss_sum");
            ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_ਸੈੱਟ_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_ਪ੍ਰਤੀ_datapoint = ਸੱਚ;
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_OPT_LOSS_TYPE_ਜੋੜ: {
            opt_ctx->loss = ggml_ਜੋੜ(ctx_results, opt_ctx->outputs);
            ggml_ਸੈੱਟ_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_ਪ੍ਰਤੀ_datapoint = ਝੂਠ;
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_ਸੈੱਟ_ਨਿਵੇਸ਼(opt_ctx->labels);
            ggml_ਸੈੱਟ_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_ਸੈੱਟ_name(opt_ctx->loss, "loss_cross_entropy");
            ਜੇ (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_ਸੈੱਟ_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_ਪ੍ਰਤੀ_datapoint = ਸੱਚ;
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_OPT_LOSS_TYPE_ਔਸਤ_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_ਸੈੱਟ_ਨਿਵੇਸ਼(opt_ctx->labels);
            ggml_ਸੈੱਟ_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_ਸੈੱਟ_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_ਸੈੱਟ_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_ਜੋੜ(ctx_results, opt_ctx->loss);
            ggml_ਸੈੱਟ_name(opt_ctx->loss, "loss_sum_squared_error");
            ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_ਸੈੱਟ_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_ਪ੍ਰਤੀ_datapoint = ਸੱਚ;
            ਰੋਕੋ;
        }
    }
    ggml_ਸੈੱਟ_output(opt_ctx->loss);
    ggml_ਸੈੱਟ_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    ਜੇ (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_ਸੈੱਟ_name(opt_ctx->pred, "pred");
        ggml_ਸੈੱਟ_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_ਗਿਣਤੀ_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_ਸੈੱਟ_name(opt_ctx->ncorrect, "ncorrect");
        ggml_ਸੈੱਟ_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    ਜੇ (opt_ctx->buf_static) {
        ਜੇ (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            ਵਾਪਸ;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ਵਾਪਸ;
    }

    ਜੇ (opt_ctx->grad_accs.ਖਾਲੀ_ਥਾਵਾਂ()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        ਅਚਲ ਪੂਰਨ n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        ਲਈ (ਪੂਰਨ i = 0; i < n_nodes; ++i) {
            ggml_tensor * ਗੰਢ = opt_ctx->gf->nodes[i];
            ਜੇ ((accumulate && (ਗੰਢ->flags & GGML_TENSOR_FLAG_PARAM)) || (ਗੰਢ->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ਗੰਢ->ne);
            } ਨਹੀਂਤਾਂ {
                opt_ctx->grad_accs[i] = ਖਾਲੀ_ਸੂਚਕ;
            }
        }

        ਜੇ (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            ਲਈ (ਪੂਰਨ i = 0; i < n_nodes; ++i) {
                ggml_tensor * ਗੰਢ = opt_ctx->gf->nodes[i];
                ਜੇ (ਗੰਢ->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ਗੰਢ->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ਗੰਢ->ne);
                } ਨਹੀਂਤਾਂ {
                    opt_ctx->grad_m[i] = ਖਾਲੀ_ਸੂਚਕ;
                    opt_ctx->grad_v[i] = ਖਾਲੀ_ਸੂਚਕ;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_ਗ੍ਰਾਫ_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ ਸੱਚ);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.ਅੰਕੜੇ());

    ਜੇ (opt_ctx->buf_static) {
        ਜੇ (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            ਵਾਪਸ;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_ਗ੍ਰਾਫ_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_ਗ੍ਰਾਫ_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ ਸੱਚ);

    opt_ctx->opt_ਕਦਮ_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_ਕਦਮ_params;
    ggml_ਸੈੱਟ_ਨਿਵੇਸ਼(adamw_params);
    ਅਚਲ ਅੱਖਰ * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    ਲਈ (ਪੂਰਨ i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = opt_ctx->gb_opt->nodes[i];
        ਬਣਤਰ ggml_tensor * grad = ggml_ਗ੍ਰਾਫ_get_grad(opt_ctx->gb_opt, ਗੰਢ);

        ਜੇ (grad && (ਗੰਢ->flags & GGML_TENSOR_FLAG_PARAM)) {
            ਬਣਤਰ ggml_tensor * m = ਖਾਲੀ_ਸੂਚਕ;
            ਬਣਤਰ ggml_tensor * v = ਖਾਲੀ_ਸੂਚਕ;
            ਜੇ (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", ਗੰਢ->name);
                ggml_format_name(v, "AdamW v for %s", ਗੰਢ->name);
            }
            ਬਣਤਰ ggml_tensor * opt_ਕਦਮ;
            ਚੋਣ (optimizer) {
                ਹਾਲਤ GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_ਕਦਮ = ggml_opt_ਕਦਮ_adamw(opt_ctx->ctx_compute, ਗੰਢ, grad, m, v, adamw_params);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_ਕਦਮ = ggml_opt_ਕਦਮ_sgd(opt_ctx->ctx_compute, ਗੰਢ, grad, adamw_params);
                    ਰੋਕੋ;
                ਮੂਲ:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_ਕਦਮ, "%s step for %s", optimizer_name, ਗੰਢ->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_ਕਦਮ);
        }
    }

    ਜੇ (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_ਗ੍ਰਾਫ_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_ਬਫਰ_type());
}

ggml_opt_context_t ggml_opt_init(ਬਣਤਰ ggml_opt_params params) {
    ggml_opt_context_t ਨਤੀਜਾ = ਨਵਾਂ ਬਣਤਰ ggml_opt_context;
    ਨਤੀਜਾ->backend_sched    = params.backend_sched;
    ਨਤੀਜਾ->ctx_compute      = params.ctx_compute;
    ਨਤੀਜਾ->loss_type        = params.loss_type;
    ਨਤੀਜਾ->build_type       = params.build_type;
    ਨਤੀਜਾ->build_type_alloc = params.build_type;
    ਨਤੀਜਾ->inputs           = params.inputs;
    ਨਤੀਜਾ->outputs          = params.outputs;
    ਨਤੀਜਾ->opt_period       = params.opt_period;
    ਨਤੀਜਾ->get_opt_pars     = params.get_opt_pars;
    ਨਤੀਜਾ->get_opt_pars_ud  = params.get_opt_pars_ud;
    ਨਤੀਜਾ->optimizer        = params.optimizer;

    GGML_ASSERT(ਨਤੀਜਾ->opt_period >= 1);

    ਨਤੀਜਾ->static_graphs = ਨਤੀਜਾ->ctx_compute;

    ਜੇ (!ਨਤੀਜਾ->static_graphs) {
        GGML_ASSERT(!ਨਤੀਜਾ->inputs);
        GGML_ASSERT(!ਨਤੀਜਾ->outputs);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    GGML_ASSERT(ਨਤੀਜਾ->inputs);
    GGML_ASSERT(ਨਤੀਜਾ->outputs);

    ਨਤੀਜਾ->gf = ggml_new_ਗ੍ਰਾਫ_custom(ਨਤੀਜਾ->ctx_compute, GGML_DEFAULT_ਗ੍ਰਾਫ_ਮਾਪ, /*grads =*/ ਸੱਚ); // Forward pass.
    ggml_build_forward_expand(ਨਤੀਜਾ->gf, ਨਤੀਜਾ->outputs);

    ggml_opt_build(ਨਤੀਜਾ);

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਖਾਲੀ ggml_opt_free(ggml_opt_context_t opt_ctx) {
    ਜੇ (opt_ctx == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    ggml_backend_ਬਫਰ_free(opt_ctx->buf_static);
    ggml_backend_ਬਫਰ_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    ਮਿਟਾਓ opt_ctx;
}

ਖਾਲੀ ggml_opt_reset(ggml_opt_context_t opt_ctx, ਤਾਰਕਿਕ optimizer) {
    ਜੇ (optimizer) {
        ggml_ਗ੍ਰਾਫ_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } ਨਹੀਂਤਾਂ {
        ggml_ਗ੍ਰਾਫ_reset(opt_ctx->gb_grad);
    }
}

ਤਾਰਕਿਕ ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    ਵਾਪਸ opt_ctx->static_graphs;
}

ਬਣਤਰ ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    ਵਾਪਸ opt_ctx->inputs;
}

ਬਣਤਰ ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    ਵਾਪਸ opt_ctx->outputs;
}

ਬਣਤਰ ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    ਵਾਪਸ opt_ctx->labels;
}

ਬਣਤਰ ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    ਵਾਪਸ opt_ctx->loss;
}

ਬਣਤਰ ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    ਵਾਪਸ opt_ctx->pred;
}

ਬਣਤਰ ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    ਵਾਪਸ opt_ctx->ncorrect;
}

ਬਣਤਰ ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, ਬਣਤਰ ggml_tensor * ਗੰਢ) {
    ਵਾਪਸ ggml_ਗ੍ਰਾਫ_get_grad_acc(opt_ctx->gb_opt, ਗੰਢ);
}

// ====== Optimization Result ======

ggml_opt_ਨਤੀਜਾ_t ggml_opt_ਨਤੀਜਾ_init() {
    ਵਾਪਸ ਨਵਾਂ ggml_opt_ਨਤੀਜਾ;
}

ਖਾਲੀ ggml_opt_ਨਤੀਜਾ_free(ggml_opt_ਨਤੀਜਾ_t ਨਤੀਜਾ) {
    ਮਿਟਾਓ ਨਤੀਜਾ;
}

ਖਾਲੀ ggml_opt_ਨਤੀਜਾ_reset(ggml_opt_ਨਤੀਜਾ_t ਨਤੀਜਾ) {
    ਨਤੀਜਾ->ndata = 0;
    ਨਤੀਜਾ->loss.clear();
    ਨਤੀਜਾ->pred.clear();
    ਨਤੀਜਾ->ncorrect = 0;
}

ਖਾਲੀ ggml_opt_ਨਤੀਜਾ_ndata(ggml_opt_ਨਤੀਜਾ_t ਨਤੀਜਾ, int64_t * ndata) {
    *ndata = ਨਤੀਜਾ->ndata;
}

ਖਾਲੀ ggml_opt_ਨਤੀਜਾ_loss(ggml_opt_ਨਤੀਜਾ_t ਨਤੀਜਾ, ਦੁੱਗਣਾ * loss, ਦੁੱਗਣਾ * unc) {
    ਅਚਲ int64_t nbatches = ਨਤੀਜਾ->loss.size(); // Number of physical batches.

    ਜੇ (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        ਵਾਪਸ;
    }

    ਦੁੱਗਣਾ ਜੋੜ         = 0.0;
    ਦੁੱਗਣਾ ਜੋੜ_squared = 0.0;

    ਲਈ (ਅਚਲ ਦਸ਼ਮਲਵ & loss : ਨਤੀਜਾ->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        ਅਚਲ ਦਸ਼ਮਲਵ loss_scaled = ਨਤੀਜਾ->loss_ਪ੍ਰਤੀ_datapoint ? loss*ਨਤੀਜਾ->opt_period : loss;
        ਜੋੜ         += loss_scaled;
        ਜੋੜ_squared += loss_scaled*loss_scaled;
    }

    ਅਚਲ ਦੁੱਗਣਾ ਔਸਤ_2 = ਜੋੜ/nbatches;
    *loss = ਨਤੀਜਾ->loss_ਪ੍ਰਤੀ_datapoint ? ਔਸਤ_2 : ਜੋੜ;

    ਜੇ (!unc) {
        ਵਾਪਸ;
    }

    ਜੇ (nbatches < 2) {
        *unc = NAN;
        ਵਾਪਸ;
    }

    ਅਚਲ ਦੁੱਗਣਾ var_ਜੋੜ = ਜੋੜ_squared/nbatches - ਔਸਤ_2*ਔਸਤ_2; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = ਨਤੀਜਾ->loss_ਪ੍ਰਤੀ_datapoint ? ਵਰਗਮੂਲ(var_ਜੋੜ / (nbatches - 1)) : ਵਰਗਮੂਲ(var_ਜੋੜ * nbatches/(nbatches - 1));
}

ਖਾਲੀ ggml_opt_ਨਤੀਜਾ_pred(ggml_opt_ਨਤੀਜਾ_t ਨਤੀਜਾ, int32_t * pred) {
    ਲਈ (size_t i = 0; i < ਨਤੀਜਾ->pred.size(); ++i) {
        pred[i] = ਨਤੀਜਾ->pred[i];
    }
}

ਖਾਲੀ ggml_opt_ਨਤੀਜਾ_accuracy(ggml_opt_ਨਤੀਜਾ_t ਨਤੀਜਾ, ਦੁੱਗਣਾ * accuracy, ਦੁੱਗਣਾ * unc) {
    *accuracy = ਨਤੀਜਾ->ncorrect >= 0 ? ਦੁੱਗਣਾ(ਨਤੀਜਾ->ncorrect) / ਦੁੱਗਣਾ(ਨਤੀਜਾ->ndata) : NAN;

    ਜੇ (!unc) {
        ਵਾਪਸ;
    }

    *unc = ਨਤੀਜਾ->ncorrect >= 0 && ਨਤੀਜਾ->ndata >= 2 ?
        ਵਰਗਮੂਲ((*accuracy) * (1.0 - (*accuracy)) / ਦੁੱਗਣਾ(ਨਤੀਜਾ->ndata - 1)) : NAN;
}

// ====== Computation ======

ਖਾਲੀ ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        ਬਣਤਰ ggml_context * ctx_compute,
        ਬਣਤਰ ggml_cgraph  * gf,
        ਬਣਤਰ ggml_tensor  * inputs,
        ਬਣਤਰ ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

ਖਾਲੀ ggml_opt_alloc(ggml_opt_context_t opt_ctx, ਤਾਰਕਿਕ backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    ਜੇ (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_ਗ੍ਰਾਫ_reset(opt_ctx->gb_grad);
    }
    ਜੇ (backward) {
        ਅਚਲ int32_t opt_i_ਅਗਲਾ = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_ਅਗਲਾ == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } ਨਹੀਂਤਾਂ {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    ਜੇ (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    ਬਣਤਰ ggml_cgraph * graph = ਖਾਲੀ_ਸੂਚਕ;
    ਚੋਣ (opt_ctx->build_type) {
        ਹਾਲਤ GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } ਰੋਕੋ;
    }
    GGML_ASSERT(graph);

    ਜੇ (opt_ctx->allocated_ਗ੍ਰਾਫ == graph) {
        opt_ctx->eval_ready = ਸੱਚ;
        ਵਾਪਸ;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    ਜੇ (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_ਗ੍ਰਾਫ_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ ਖਾਲੀ_ਸੂਚਕ,
            /*.no_alloc   =*/ ਸੱਚ,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_ਗ੍ਰਾਫ_copy = dup_ਗ੍ਰਾਫ(opt_ctx->ctx_copy, graph);
    } ਨਹੀਂਤਾਂ {
        opt_ctx->allocated_ਗ੍ਰਾਫ_copy = graph;
    }

    ggml_backend_sched_alloc_ਗ੍ਰਾਫ(opt_ctx->backend_sched, opt_ctx->allocated_ਗ੍ਰਾਫ_copy);
    opt_ctx->allocated_ਗ੍ਰਾਫ = graph;

    opt_ctx->eval_ready = ਸੱਚ;
}

ਖਾਲੀ ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_ਨਤੀਜਾ_t ਨਤੀਜਾ) {
    GGML_ASSERT(opt_ctx->eval_ready);
    ਜੇ (opt_ctx->allocated_ਗ੍ਰਾਫ == opt_ctx->gb_opt) {
        ਅਚਲ ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        ਚੋਣ (opt_ctx->optimizer) {
            ਹਾਲਤ GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                ਅਚਲ ਦਸ਼ਮਲਵ beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                ਅਚਲ ਦਸ਼ਮਲਵ beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                ਦਸ਼ਮਲਵ * adamw_par_ਅੰਕੜੇ = ggml_get_ਅੰਕੜੇ_f32(opt_ctx->opt_ਕਦਮ_params);
                adamw_par_ਅੰਕੜੇ[0] = opt_pars.adamw.alpha;
                adamw_par_ਅੰਕੜੇ[1] = opt_pars.adamw.beta1;
                adamw_par_ਅੰਕੜੇ[2] = opt_pars.adamw.beta2;
                adamw_par_ਅੰਕੜੇ[3] = opt_pars.adamw.eps;
                adamw_par_ਅੰਕੜੇ[4] = opt_pars.adamw.wd;
                adamw_par_ਅੰਕੜੇ[5] = beta1h;
                adamw_par_ਅੰਕੜੇ[6] = beta2h;
            } ਰੋਕੋ;
            ਹਾਲਤ GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                ਦਸ਼ਮਲਵ * sgd = ggml_get_ਅੰਕੜੇ_f32(opt_ctx->opt_ਕਦਮ_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_ਗ੍ਰਾਫ_compute(opt_ctx->backend_sched, opt_ctx->allocated_ਗ੍ਰਾਫ_copy);
    opt_ctx->iter += opt_ctx->allocated_ਗ੍ਰਾਫ == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    ਜੇ (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = ਖਾਲੀ_ਸੂਚਕ;
        opt_ctx->gb_grad              = ਖਾਲੀ_ਸੂਚਕ;
        opt_ctx->gb_opt               = ਖਾਲੀ_ਸੂਚਕ;
        opt_ctx->allocated_ਗ੍ਰਾਫ      = ਖਾਲੀ_ਸੂਚਕ;
        opt_ctx->allocated_ਗ੍ਰਾਫ_copy = ਖਾਲੀ_ਸੂਚਕ;
    }

    opt_ctx->eval_ready = ਝੂਠ;

    ਜੇ (!ਨਤੀਜਾ) {
        ਵਾਪਸ;
    }

    ਜੇ (ਨਤੀਜਾ->ndata == 0) {
        ਨਤੀਜਾ->loss_ਪ੍ਰਤੀ_datapoint = opt_ctx->loss_ਪ੍ਰਤੀ_datapoint;
        ਨਤੀਜਾ->opt_period         = opt_ctx->opt_period;
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਨਤੀਜਾ->loss_ਪ੍ਰਤੀ_datapoint == opt_ctx->loss_ਪ੍ਰਤੀ_datapoint);
        GGML_ASSERT(ਨਤੀਜਾ->opt_period         == opt_ctx->opt_period);
    }

    ਅਚਲ int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(ਨਤੀਜਾ->ndata == ndata*int64_t(ਨਤੀਜਾ->loss.size()) && "varying batch size not supported");
    ਨਤੀਜਾ->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    ਦਸ਼ਮਲਵ loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    ਨਤੀਜਾ->loss.push_back(loss);

    ਜੇ (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.ਅੰਕੜੇ(), 0, ggml_nbytes(opt_ctx->pred));
        ਨਤੀਜਾ->pred.insert(ਨਤੀਜਾ->pred.end(), pred.begin(), pred.end());
    }

    ਜੇ (!opt_ctx->ncorrect || ਨਤੀਜਾ->ncorrect < 0) {
        ਨਤੀਜਾ->ncorrect = -1;
        ਵਾਪਸ;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    ਨਤੀਜਾ->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

ਖਾਲੀ ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_ਨਤੀਜਾ_t       ਨਤੀਜਾ_train,
        ggml_opt_ਨਤੀਜਾ_t       ਨਤੀਜਾ_eval,
        int64_t                 idata_ਵੰਡੋ,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    ਬਣਤਰ ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    ਬਣਤਰ ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    ਬਣਤਰ ggml_tensor * ਅੰਕੜੇ   = ggml_opt_dataset_ਅੰਕੜੇ(dataset);
    GGML_ASSERT(ਅੰਕੜੇ->ne[0] == inputs->ne[0]);

    ਅਚਲ int64_t ndata       =   ਅੰਕੜੇ->ne[1];
    ਅਚਲ int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(ਅੰਕੜੇ->ne[1] % inputs->ne[1] == 0);
    ਅਚਲ int64_t nbatches = ndata/ndata_batch;

    idata_ਵੰਡੋ = idata_ਵੰਡੋ < 0 ? ndata : idata_ਵੰਡੋ;
    GGML_ASSERT(idata_ਵੰਡੋ % ndata_batch == 0);
    ਅਚਲ int64_t ibatch_ਵੰਡੋ = idata_ਵੰਡੋ / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_ਸ਼ੁਰੂ = ggml_ਸਮਾਂ_us();
    ਲਈ (; ibatch < ibatch_ਵੰਡੋ; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ ਸੱਚ);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, ਨਤੀਜਾ_train);
        ਜੇ (callback_train) {
            callback_train(ਸੱਚ, opt_ctx, dataset, ਨਤੀਜਾ_train, ibatch+1, ibatch_ਵੰਡੋ, t_loop_ਸ਼ੁਰੂ);
        }
    }
    t_loop_ਸ਼ੁਰੂ = ggml_ਸਮਾਂ_us();
    ਲਈ (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ ਝੂਠ);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, ਨਤੀਜਾ_eval);
        ਜੇ (callback_eval) {
            callback_eval(ਝੂਠ, opt_ctx, dataset, ਨਤੀਜਾ_eval, ibatch+1-ibatch_ਵੰਡੋ, nbatches-ibatch_ਵੰਡੋ, t_loop_ਸ਼ੁਰੂ);
        }
    }
}

ਖਾਲੀ ggml_opt_epoch_callback_ਤਰੱਕੀ_bar(
        ਤਾਰਕਿਕ               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_ਨਤੀਜਾ_t  ਨਤੀਜਾ,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_ਸ਼ੁਰੂ_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_ਲੰਮਾਈ = 8;
    ਅਚਲ int64_t ibatch8 = 8 * ibatch;
    ਲਈ (int64_t j = 0; j < bar_ਲੰਮਾਈ; ++j) {
        ਜੇ        (ibatch_max * (8*j + 8) / bar_ਲੰਮਾਈ < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } ਨਹੀਂਤਾਂ ਜੇ (ibatch_max * (8*j + 7) / bar_ਲੰਮਾਈ < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } ਨਹੀਂਤਾਂ ਜੇ (ibatch_max * (8*j + 6) / bar_ਲੰਮਾਈ < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } ਨਹੀਂਤਾਂ ਜੇ (ibatch_max * (8*j + 5) / bar_ਲੰਮਾਈ < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } ਨਹੀਂਤਾਂ ਜੇ (ibatch_max * (8*j + 4) / bar_ਲੰਮਾਈ < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } ਨਹੀਂਤਾਂ ਜੇ (ibatch_max * (8*j + 3) / bar_ਲੰਮਾਈ < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } ਨਹੀਂਤਾਂ ਜੇ (ibatch_max * (8*j + 2) / bar_ਲੰਮਾਈ < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } ਨਹੀਂਤਾਂ ਜੇ (ibatch_max * (8*j + 1) / bar_ਲੰਮਾਈ < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } ਨਹੀਂਤਾਂ {
            fprintf(stderr, " ");
        }
    }

    ਅਚਲ int64_t batch_ਮਾਪ = ggml_opt_inputs(opt_ctx)->ne[1];
    ਅਚਲ int64_t idata      = ibatch*batch_ਮਾਪ;
    ਅਚਲ int64_t idata_max  = ibatch_max*batch_ਮਾਪ;

    ਦੁੱਗਣਾ loss;
    ਦੁੱਗਣਾ loss_unc;
    ggml_opt_ਨਤੀਜਾ_loss(ਨਤੀਜਾ, &loss, &loss_unc);

    ਦੁੱਗਣਾ accuracy;
    ਦੁੱਗਣਾ accuracy_unc;
    ggml_opt_ਨਤੀਜਾ_accuracy(ਨਤੀਜਾ, &accuracy, &accuracy_unc);

    ਅਚਲ int64_t t_ibatch_us = ggml_ਸਮਾਂ_us() - t_ਸ਼ੁਰੂ_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    ਅਚਲ int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    ਅਚਲ int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    ਅਚਲ int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    ਅਚਲ int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    ਅਚਲ int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    ਜੇ (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_ਅਣਵਰਤਿਆ(dataset);
}

ਖਾਲੀ ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        ਗਿਣਤੀ ggml_opt_loss_type         loss_type,
        ਗਿਣਤੀ ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        ਦਸ਼ਮਲਵ                           val_ਵੰਡੋ,
        ਤਾਰਕਿਕ                            silent) {
    ggml_ਸਮਾਂ_init();
    ਅਚਲ int64_t t_ਸ਼ੁਰੂ_us = ggml_ਸਮਾਂ_us();

    ਅਚਲ int64_t ndata           = ggml_opt_dataset_ਅੰਕੜੇ(dataset)->ne[1];
    ਅਚਲ int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    ਅਚਲ int64_t opt_period       = nbatch_logical / nbatch_physical;
    ਅਚਲ int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_ਵੰਡੋ >= 0.0f);
    GGML_ASSERT(val_ਵੰਡੋ <  1.0f);
    ਅਚਲ int64_t ibatch_ਵੰਡੋ = int64_t(((1.0f - val_ਵੰਡੋ) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    ਅਚਲ int64_t idata_ਵੰਡੋ  = ibatch_ਵੰਡੋ * nbatch_physical;

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
    ਜੇ (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_ਨਤੀਜਾ_t ਨਤੀਜਾ_train = ggml_opt_ਨਤੀਜਾ_init();
    ggml_opt_ਨਤੀਜਾ_t ਨਤੀਜਾ_val   = ggml_opt_ਨਤੀਜਾ_init();

    ggml_opt_epoch_callback epoch_callback = silent ? ਖਾਲੀ_ਸੂਚਕ : ggml_opt_epoch_callback_ਤਰੱਕੀ_bar;

    ਲਈ (; epoch <= nepoch; ++epoch) {
        ਜੇ (nbatch_logical < idata_ਵੰਡੋ) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_ਵੰਡੋ);
        }

        ggml_opt_ਨਤੀਜਾ_reset(ਨਤੀਜਾ_train);
        ggml_opt_ਨਤੀਜਾ_reset(ਨਤੀਜਾ_val);

        ਜੇ (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, ਨਤੀਜਾ_train, ਨਤੀਜਾ_val, idata_ਵੰਡੋ, epoch_callback, epoch_callback);
        ਜੇ (!silent) {
            fprintf(stderr, "\n");
        }
    }

    ਜੇ (!silent) {
        int64_t t_ਕੁੱਲ_s = (ggml_ਸਮਾਂ_us() - t_ਸ਼ੁਰੂ_us) / 1000000;
        ਅਚਲ int64_t t_ਕੁੱਲ_h = t_ਕੁੱਲ_s / 3600;
        t_ਕੁੱਲ_s -= t_ਕੁੱਲ_h * 3600;
        ਅਚਲ int64_t t_ਕੁੱਲ_m = t_ਕੁੱਲ_s / 60;
        t_ਕੁੱਲ_s -= t_ਕੁੱਲ_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_ਕੁੱਲ_h, t_ਕੁੱਲ_m, t_ਕੁੱਲ_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_ਨਤੀਜਾ_free(ਨਤੀਜਾ_train);
    ggml_opt_ਨਤੀਜਾ_free(ਨਤੀਜਾ_val);
}

ਗਿਣਤੀ ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    ਵਾਪਸ c->optimizer;
}

GGML_API ਅਚਲ ਅੱਖਰ * ggml_opt_optimizer_name(ਗਿਣਤੀ ggml_opt_optimizer_type o) {
    ਚੋਣ (o) {
        ਹਾਲਤ GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            ਵਾਪਸ "adamw";
        ਹਾਲਤ GGML_OPT_OPTIMIZER_TYPE_SGD:
            ਵਾਪਸ "sgd";
        ਮੂਲ:
            ਵਾਪਸ "undefined";
    };
}
