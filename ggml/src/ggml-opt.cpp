#ಅನ್ತರ್ಭಾವ "ggml-opt.h"

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-alloc.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cinttypes>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <random>
#ಅನ್ತರ್ಭಾವ <vector>

ರಚನೆ ggml_opt_dataset {
    ರಚನೆ ggml_context   * ctx    = ಶೂನ್ಯಸೂಚಕ;
    ggml_backend_ಬಫರ್_t   buf    = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_tensor    * ದತ್ತಾಂಶ   = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_tensor    * labels = ಶೂನ್ಯಸೂಚಕ;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_ದತ್ತಾಂಶ    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

ರಚನೆ ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = ಶೂನ್ಯಸೂಚಕ;
    ggml_cgraph              * allocated_ನಕ್ಷೆ      = ಶೂನ್ಯಸೂಚಕ;
    ggml_cgraph              * allocated_ನಕ್ಷೆ_copy = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_context      * ctx_static           = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_context      * ctx_cpu              = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_context      * ctx_compute          = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_context      * ctx_copy             = ಶೂನ್ಯಸೂಚಕ;
    ggml_backend_ಬಫರ್_t      buf_static           = ಶೂನ್ಯಸೂಚಕ;
    ggml_backend_ಬಫರ್_t      buf_cpu              = ಶೂನ್ಯಸೂಚಕ;
    std::mt19937               rng;
    ಎಣಿಕೆ ggml_opt_loss_type    loss_type;
    ಎಣಿಕೆ ggml_opt_build_type   build_type;
    ಎಣಿಕೆ ggml_opt_build_type   build_type_alloc;

    ರಚನೆ ggml_tensor * inputs  = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_tensor * outputs = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_tensor * labels  = ಶೂನ್ಯಸೂಚಕ;

    ರಚನೆ ggml_tensor * loss     = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_tensor * pred     = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_tensor * ncorrect = ಶೂನ್ಯಸೂಚಕ;

    ರಚನೆ ggml_cgraph * gf      = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_cgraph * gb_grad = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_cgraph * gb_opt  = ಶೂನ್ಯಸೂಚಕ;
    ತಾರ್ಕಿಕ static_graphs           = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ eval_ready              = ಸುಳ್ಳು;
    std::vector<ರಚನೆ ggml_tensor *> grad_accs;
    std::vector<ರಚನೆ ggml_tensor *> grad_m;
    std::vector<ರಚನೆ ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    ತಾರ್ಕಿಕ    loss_ಪ್ರತಿ_datapoint = ಸುಳ್ಳು;

    ggml_opt_get_optimizer_params get_opt_pars    = ಶೂನ್ಯಸೂಚಕ;
    ಶೂನ್ಯ *                        get_opt_pars_ud = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ggml_tensor *          opt_ಹೆಜ್ಜೆ_params = ಶೂನ್ಯಸೂಚಕ; // Stores output of get_opt_pars.

    ಎಣಿಕೆ ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

ರಚನೆ ggml_opt_ಫಲಿತಾಂಶ {
    int64_t              ndata    = 0;
    std::vector<ದಶಮಾಂಶ>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    ತಾರ್ಕಿಕ    loss_ಪ್ರತಿ_datapoint = ಸುಳ್ಳು;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        ಎಣಿಕೆ ggml_type type_ದತ್ತಾಂಶ,
        ಎಣಿಕೆ ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t ಫಲಿತಾಂಶ = ಹೊಸ ggml_opt_dataset;
    ಫಲಿತಾಂಶ->ndata       = ndata;
    ಫಲಿತಾಂಶ->ndata_shard = ndata_shard;

    {
        ರಚನೆ ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ ಶೂನ್ಯಸೂಚಕ,
            /*.no_alloc   =*/ ನಿಜ,
        };
        ಫಲಿತಾಂಶ->ctx = ggml_init(params);
    }

    ಫಲಿತಾಂಶ->ದತ್ತಾಂಶ = ggml_new_tensor_2d(ಫಲಿತಾಂಶ->ctx, type_ದತ್ತಾಂಶ, ne_datapoint, ndata);
    ಫಲಿತಾಂಶ->nbs_ದತ್ತಾಂಶ = ggml_nbytes(ಫಲಿತಾಂಶ->ದತ್ತಾಂಶ) * ndata_shard/ndata;

    ಆದರೆ (ne_label > 0) {
        ಫಲಿತಾಂಶ->labels = ggml_new_tensor_2d(ಫಲಿತಾಂಶ->ctx, type_label, ne_label, ndata);
        ಫಲಿತಾಂಶ->nbs_labels = ggml_nbytes(ಫಲಿತಾಂಶ->labels) * ndata_shard/ndata;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ->labels = ಶೂನ್ಯಸೂಚಕ;
        ಫಲಿತಾಂಶ->nbs_labels = 0;
    }

    ಫಲಿತಾಂಶ->buf = ggml_backend_alloc_ctx_tensors_from_buft(ಫಲಿತಾಂಶ->ctx, ggml_backend_cpu_ಬಫರ್_type());

    ಸ್ಥಿರಾಂಕ int64_t nshards = ndata/ndata_shard;
    ಫಲಿತಾಂಶ->permutation.resize(nshards);
    ಗಾಗಿ (int64_t i = 0; i < nshards; ++i) {
        ಫಲಿತಾಂಶ->permutation[i] = i;
    }
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಶೂನ್ಯ ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_ಬಫರ್_free(dataset->buf);
    ggml_free(dataset->ctx);
    ಅಳಿಸು dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    ಹಿಂತಿರುಗಿ dataset->ndata;
}

ರಚನೆ ggml_tensor * ggml_opt_dataset_ದತ್ತಾಂಶ(ggml_opt_dataset_t dataset) {
    ಹಿಂತಿರುಗಿ dataset->ದತ್ತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    ಹಿಂತಿರುಗಿ dataset->labels;
}

ಶೂನ್ಯ ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    ಆದರೆ (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    ಸ್ಥಿರಾಂಕ int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

ಶೂನ್ಯ ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, ರಚನೆ ggml_tensor * ದತ್ತಾಂಶ_batch, ರಚನೆ ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   ದತ್ತಾಂಶ_batch && ggml_is_contiguous(ದತ್ತಾಂಶ_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == ಶೂನ್ಯಸೂಚಕ) == (dataset->labels == ಶೂನ್ಯಸೂಚಕ));
    GGML_ASSERT(                   ದತ್ತಾಂಶ_batch->type == dataset->ದತ್ತಾಂಶ->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    ಸ್ಥಿರಾಂಕ size_t nb_ದತ್ತಾಂಶ_batch = ggml_nbytes(ದತ್ತಾಂಶ_batch);
    GGML_ASSERT(nb_ದತ್ತಾಂಶ_batch % dataset->nbs_ದತ್ತಾಂಶ == 0);
    ಸ್ಥಿರಾಂಕ int64_t shards_ಪ್ರತಿ_batch = nb_ದತ್ತಾಂಶ_batch / dataset->nbs_ದತ್ತಾಂಶ;

    ಆದರೆ (labels_batch) {
        ಸ್ಥಿರಾಂಕ size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_ಪ್ರತಿ_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_ಪ್ರತಿ_batch <= int64_t(dataset->permutation.size()));

    ಗಾಗಿ (int64_t ishard_batch = 0; ishard_batch < shards_ಪ್ರತಿ_batch; ++ishard_batch) {
        ಸ್ಥಿರಾಂಕ int64_t ishard = dataset->permutation[ibatch*shards_ಪ್ರತಿ_batch + ishard_batch];

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ptr_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) dataset->ದತ್ತಾಂಶ->ದತ್ತಾಂಶ + ishard*dataset->nbs_ದತ್ತಾಂಶ;
        ggml_backend_tensor_ಗಣ(ದತ್ತಾಂಶ_batch, ptr_ದತ್ತಾಂಶ, ishard_batch*dataset->nbs_ದತ್ತಾಂಶ, dataset->nbs_ದತ್ತಾಂಶ);

        ಆದರೆ (!labels_batch) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ptr_labels = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) dataset->labels->ದತ್ತಾಂಶ + ishard*dataset->nbs_labels;
        ggml_backend_tensor_ಗಣ(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

ಶೂನ್ಯ ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, ಶೂನ್ಯ * ದತ್ತಾಂಶ_batch, size_t nb_ದತ್ತಾಂಶ_batch, ಶೂನ್ಯ * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == ಶೂನ್ಯಸೂಚಕ) == (dataset->labels == ಶೂನ್ಯಸೂಚಕ));
    GGML_ASSERT(nb_ದತ್ತಾಂಶ_batch % dataset->nbs_ದತ್ತಾಂಶ == 0);

    ಸ್ಥಿರಾಂಕ int64_t shards_ಪ್ರತಿ_batch = nb_ದತ್ತಾಂಶ_batch / dataset->nbs_ದತ್ತಾಂಶ;

    GGML_ASSERT((ibatch + 1)*shards_ಪ್ರತಿ_batch <= int64_t(dataset->permutation.size()));

    ಗಾಗಿ (int64_t ishard_batch = 0; ishard_batch < shards_ಪ್ರತಿ_batch; ++ishard_batch) {
        ಸ್ಥಿರಾಂಕ int64_t ishard = dataset->permutation[ibatch*shards_ಪ್ರತಿ_batch + ishard_batch];

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ptr_ದತ್ತಾಂಶ       = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) dataset->ದತ್ತಾಂಶ->ದತ್ತಾಂಶ + ishard      *dataset->nbs_ದತ್ತಾಂಶ;
        ಅಕ್ಷರ       * ptr_ದತ್ತಾಂಶ_batch = (ಅಕ್ಷರ       *) ದತ್ತಾಂಶ_batch          + ishard_batch*dataset->nbs_ದತ್ತಾಂಶ;
        ಸ್ಮರಣೆ_ನಕಲು(ptr_ದತ್ತಾಂಶ_batch, ptr_ದತ್ತಾಂಶ, dataset->nbs_ದತ್ತಾಂಶ);

        ಆದರೆ (!labels_batch) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ptr_labels       = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) dataset->labels->ದತ್ತಾಂಶ + ishard      *dataset->nbs_labels;
        ಅಕ್ಷರ       * ptr_labels_batch = (ಅಕ್ಷರ       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        ಸ್ಮರಣೆ_ನಕಲು(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

ರಚನೆ ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(ಶೂನ್ಯ * userdata) {
    GGML_ಬಳಸದ(userdata);

    ggml_opt_optimizer_params ಫಲಿತಾಂಶ;

    ಫಲಿತಾಂಶ.adamw.alpha = 0.001f;
    ಫಲಿತಾಂಶ.adamw.beta1 = 0.9f;
    ಫಲಿತಾಂಶ.adamw.beta2 = 0.999f;
    ಫಲಿತಾಂಶ.adamw.eps   = 1e-8f;
    ಫಲಿತಾಂಶ.adamw.wd    = 0.0f;

    ಫಲಿತಾಂಶ.sgd.alpha   = 1e-3f;
    ಫಲಿತಾಂಶ.sgd.wd      = 0.0f;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}


ರಚನೆ ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(ಶೂನ್ಯ * userdata) {
    ಹಿಂತಿರುಗಿ *((ರಚನೆ ggml_opt_optimizer_params *) userdata);
}

ರಚನೆ ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        ಎಣಿಕೆ ggml_opt_loss_type   loss_type) {
    ಹಿಂತಿರುಗಿ {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ ಶೂನ್ಯಸೂಚಕ,
        /*inputs          =*/ ಶೂನ್ಯಸೂಚಕ,
        /*logits          =*/ ಶೂನ್ಯಸೂಚಕ,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ ಶೂನ್ಯಸೂಚಕ,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

ಸ್ಥಿರ ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    ಆದರೆ (!tensor) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಆದರೆ (tensor_map.find(tensor) != tensor_map.end()) {
        ಹಿಂತಿರುಗಿ tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    ಸ್ಮರಣೆ_ನಕಲು(new_tensor->op_params, tensor->op_params, ಗಾತ್ರ(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->ದತ್ತಾಂಶ = tensor->ದತ್ತಾಂಶ;
    new_tensor->ಬಫರ್ = tensor->ಬಫರ್;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    ಹಿಂತಿರುಗಿ new_tensor;
}

ಸ್ಥಿರ ggml_cgraph * dup_ನಕ್ಷೆ(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_ನಕ್ಷೆ_custom(ctx, src->size, /*grads =*/ ನಿಜ);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < src->n_nodes; ++i) {
        ಸ್ಥಿರಾಂಕ size_t igrad_src = ggml_hash_ಕಂಡುಹಿಡಿ(&src->ಭೇಟಿಯಾದ_hash_ಗಣ, src->nodes[i]);
        ಸ್ಥಿರಾಂಕ size_t igrad_dst = ggml_hash_ಕಂಡುಹಿಡಿ(&dst->ಭೇಟಿಯಾದ_hash_ಗಣ, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_ತುಂಬಿದ);
        GGML_ASSERT(ggml_bitset_get(src->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_ತುಂಬಿದ);
        GGML_ASSERT(ggml_bitset_get(dst->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    ಹಿಂತಿರುಗಿ dst;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->ದತ್ತಾಂಶ) && "when using static graphs the inputs must be allocated statically");

    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_ಗಣ_ಒಳಹರಿವು(opt_ctx->inputs);
    ggml_ಗಣ_output(opt_ctx->outputs);

    ಪೂರ್ಣಾಂಕ n_param = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಗಂಟು = opt_ctx->gf->nodes[i];
        ಆದರೆ (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(ಗಂಟು->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    ಆದರೆ (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        ಸ್ಥಿರಾಂಕ size_t tensors_ಪ್ರತಿ_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        ಸ್ಥಿರಾಂಕ size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_meta = (n_loss + tensors_ಪ್ರತಿ_param*n_param + tensors_const) * ggml_tensor_overhead();
        ರಚನೆ ggml_init_params params = {
            /*.mem_size   =*/ ಗಾತ್ರ_meta,
            /*.mem_buffer =*/ ಶೂನ್ಯಸೂಚಕ,
            /*.no_alloc   =*/ ನಿಜ,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_meta = 1 * ggml_tensor_overhead();
        ರಚನೆ ggml_init_params params = {
            /*.mem_size   =*/ ಗಾತ್ರ_meta,
            /*.mem_buffer =*/ ಶೂನ್ಯಸೂಚಕ,
            /*.no_alloc   =*/ ನಿಜ,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_ಬಫರ್_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = ಶೂನ್ಯಸೂಚಕ;
    }

    ರಚನೆ ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    ಆಯ್ಕೆ (opt_ctx->loss_type) {
        ಸ್ಥಿತಿ GGML_OPT_LOSS_TYPE_ಸರಾಸರಿ: {
            opt_ctx->loss = ggml_ಮೊತ್ತ(ctx_results, opt_ctx->outputs);
            ggml_ಗಣ_name(opt_ctx->loss, "loss_sum");
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_ಗಣ_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_ಪ್ರತಿ_datapoint = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_OPT_LOSS_TYPE_ಮೊತ್ತ: {
            opt_ctx->loss = ggml_ಮೊತ್ತ(ctx_results, opt_ctx->outputs);
            ggml_ಗಣ_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_ಪ್ರತಿ_datapoint = ಸುಳ್ಳು;
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_ಗಣ_ಒಳಹರಿವು(opt_ctx->labels);
            ggml_ಗಣ_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_ಗಣ_name(opt_ctx->loss, "loss_cross_entropy");
            ಆದರೆ (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_ಗಣ_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_ಪ್ರತಿ_datapoint = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_OPT_LOSS_TYPE_ಸರಾಸರಿ_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_ಗಣ_ಒಳಹರಿವು(opt_ctx->labels);
            ggml_ಗಣ_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_ಗಣ_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_ಗಣ_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_ಮೊತ್ತ(ctx_results, opt_ctx->loss);
            ggml_ಗಣ_name(opt_ctx->loss, "loss_sum_squared_error");
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_ಗಣ_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_ಪ್ರತಿ_datapoint = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        }
    }
    ggml_ಗಣ_output(opt_ctx->loss);
    ggml_ಗಣ_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    ಆದರೆ (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_ಗಣ_name(opt_ctx->pred, "pred");
        ggml_ಗಣ_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_ಎಣಿಕೆ_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_ಗಣ_name(opt_ctx->ncorrect, "ncorrect");
        ggml_ಗಣ_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    ಆದರೆ (opt_ctx->buf_static) {
        ಆದರೆ (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            ಹಿಂತಿರುಗಿ;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (opt_ctx->grad_accs.ಖಾಲಿಗಳು()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_nodes; ++i) {
            ggml_tensor * ಗಂಟು = opt_ctx->gf->nodes[i];
            ಆದರೆ ((accumulate && (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM)) || (ಗಂಟು->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ಗಂಟು->ne);
            } ಇಲ್ಲದಿದ್ದರೆ {
                opt_ctx->grad_accs[i] = ಶೂನ್ಯಸೂಚಕ;
            }
        }

        ಆದರೆ (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_nodes; ++i) {
                ggml_tensor * ಗಂಟು = opt_ctx->gf->nodes[i];
                ಆದರೆ (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ಗಂಟು->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, ಗಂಟು->ne);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    opt_ctx->grad_m[i] = ಶೂನ್ಯಸೂಚಕ;
                    opt_ctx->grad_v[i] = ಶೂನ್ಯಸೂಚಕ;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_ನಕ್ಷೆ_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ ನಿಜ);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.ದತ್ತಾಂಶ());

    ಆದರೆ (opt_ctx->buf_static) {
        ಆದರೆ (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            ಹಿಂತಿರುಗಿ;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_ನಕ್ಷೆ_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_ನಕ್ಷೆ_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ ನಿಜ);

    opt_ctx->opt_ಹೆಜ್ಜೆ_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_ಹೆಜ್ಜೆ_params;
    ggml_ಗಣ_ಒಳಹರಿವು(adamw_params);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        ರಚನೆ ggml_tensor * ಗಂಟು = opt_ctx->gb_opt->nodes[i];
        ರಚನೆ ggml_tensor * grad = ggml_ನಕ್ಷೆ_get_grad(opt_ctx->gb_opt, ಗಂಟು);

        ಆದರೆ (grad && (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM)) {
            ರಚನೆ ggml_tensor * m = ಶೂನ್ಯಸೂಚಕ;
            ರಚನೆ ggml_tensor * v = ಶೂನ್ಯಸೂಚಕ;
            ಆದರೆ (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", ಗಂಟು->name);
                ggml_format_name(v, "AdamW v for %s", ಗಂಟು->name);
            }
            ರಚನೆ ggml_tensor * opt_ಹೆಜ್ಜೆ;
            ಆಯ್ಕೆ (optimizer) {
                ಸ್ಥಿತಿ GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_ಹೆಜ್ಜೆ = ggml_opt_ಹೆಜ್ಜೆ_adamw(opt_ctx->ctx_compute, ಗಂಟು, grad, m, v, adamw_params);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_ಹೆಜ್ಜೆ = ggml_opt_ಹೆಜ್ಜೆ_sgd(opt_ctx->ctx_compute, ಗಂಟು, grad, adamw_params);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_ಹೆಜ್ಜೆ, "%s step for %s", optimizer_name, ಗಂಟು->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_ಹೆಜ್ಜೆ);
        }
    }

    ಆದರೆ (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_ನಕ್ಷೆ_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_ಬಫರ್_type());
}

ggml_opt_context_t ggml_opt_init(ರಚನೆ ggml_opt_params params) {
    ggml_opt_context_t ಫಲಿತಾಂಶ = ಹೊಸ ರಚನೆ ggml_opt_context;
    ಫಲಿತಾಂಶ->backend_sched    = params.backend_sched;
    ಫಲಿತಾಂಶ->ctx_compute      = params.ctx_compute;
    ಫಲಿತಾಂಶ->loss_type        = params.loss_type;
    ಫಲಿತಾಂಶ->build_type       = params.build_type;
    ಫಲಿತಾಂಶ->build_type_alloc = params.build_type;
    ಫಲಿತಾಂಶ->inputs           = params.inputs;
    ಫಲಿತಾಂಶ->outputs          = params.outputs;
    ಫಲಿತಾಂಶ->opt_period       = params.opt_period;
    ಫಲಿತಾಂಶ->get_opt_pars     = params.get_opt_pars;
    ಫಲಿತಾಂಶ->get_opt_pars_ud  = params.get_opt_pars_ud;
    ಫಲಿತಾಂಶ->optimizer        = params.optimizer;

    GGML_ASSERT(ಫಲಿತಾಂಶ->opt_period >= 1);

    ಫಲಿತಾಂಶ->static_graphs = ಫಲಿತಾಂಶ->ctx_compute;

    ಆದರೆ (!ಫಲಿತಾಂಶ->static_graphs) {
        GGML_ASSERT(!ಫಲಿತಾಂಶ->inputs);
        GGML_ASSERT(!ಫಲಿತಾಂಶ->outputs);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    GGML_ASSERT(ಫಲಿತಾಂಶ->inputs);
    GGML_ASSERT(ಫಲಿತಾಂಶ->outputs);

    ಫಲಿತಾಂಶ->gf = ggml_new_ನಕ್ಷೆ_custom(ಫಲಿತಾಂಶ->ctx_compute, GGML_DEFAULT_ನಕ್ಷೆ_ಗಾತ್ರ, /*grads =*/ ನಿಜ); // Forward pass.
    ggml_build_forward_expand(ಫಲಿತಾಂಶ->gf, ಫಲಿತಾಂಶ->outputs);

    ggml_opt_build(ಫಲಿತಾಂಶ);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಶೂನ್ಯ ggml_opt_free(ggml_opt_context_t opt_ctx) {
    ಆದರೆ (opt_ctx == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    ggml_backend_ಬಫರ್_free(opt_ctx->buf_static);
    ggml_backend_ಬಫರ್_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    ಅಳಿಸು opt_ctx;
}

ಶೂನ್ಯ ggml_opt_reset(ggml_opt_context_t opt_ctx, ತಾರ್ಕಿಕ optimizer) {
    ಆದರೆ (optimizer) {
        ggml_ನಕ್ಷೆ_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_ನಕ್ಷೆ_reset(opt_ctx->gb_grad);
    }
}

ತಾರ್ಕಿಕ ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    ಹಿಂತಿರುಗಿ opt_ctx->static_graphs;
}

ರಚನೆ ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    ಹಿಂತಿರುಗಿ opt_ctx->inputs;
}

ರಚನೆ ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    ಹಿಂತಿರುಗಿ opt_ctx->outputs;
}

ರಚನೆ ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    ಹಿಂತಿರುಗಿ opt_ctx->labels;
}

ರಚನೆ ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    ಹಿಂತಿರುಗಿ opt_ctx->loss;
}

ರಚನೆ ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    ಹಿಂತಿರುಗಿ opt_ctx->pred;
}

ರಚನೆ ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    ಹಿಂತಿರುಗಿ opt_ctx->ncorrect;
}

ರಚನೆ ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, ರಚನೆ ggml_tensor * ಗಂಟು) {
    ಹಿಂತಿರುಗಿ ggml_ನಕ್ಷೆ_get_grad_acc(opt_ctx->gb_opt, ಗಂಟು);
}

// ====== Optimization Result ======

ggml_opt_ಫಲಿತಾಂಶ_t ggml_opt_ಫಲಿತಾಂಶ_init() {
    ಹಿಂತಿರುಗಿ ಹೊಸ ggml_opt_ಫಲಿತಾಂಶ;
}

ಶೂನ್ಯ ggml_opt_ಫಲಿತಾಂಶ_free(ggml_opt_ಫಲಿತಾಂಶ_t ಫಲಿತಾಂಶ) {
    ಅಳಿಸು ಫಲಿತಾಂಶ;
}

ಶೂನ್ಯ ggml_opt_ಫಲಿತಾಂಶ_reset(ggml_opt_ಫಲಿತಾಂಶ_t ಫಲಿತಾಂಶ) {
    ಫಲಿತಾಂಶ->ndata = 0;
    ಫಲಿತಾಂಶ->loss.clear();
    ಫಲಿತಾಂಶ->pred.clear();
    ಫಲಿತಾಂಶ->ncorrect = 0;
}

ಶೂನ್ಯ ggml_opt_ಫಲಿತಾಂಶ_ndata(ggml_opt_ಫಲಿತಾಂಶ_t ಫಲಿತಾಂಶ, int64_t * ndata) {
    *ndata = ಫಲಿತಾಂಶ->ndata;
}

ಶೂನ್ಯ ggml_opt_ಫಲಿತಾಂಶ_loss(ggml_opt_ಫಲಿತಾಂಶ_t ಫಲಿತಾಂಶ, ದ್ವಿಗುಣ * loss, ದ್ವಿಗುಣ * unc) {
    ಸ್ಥಿರಾಂಕ int64_t nbatches = ಫಲಿತಾಂಶ->loss.size(); // Number of physical batches.

    ಆದರೆ (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        ಹಿಂತಿರುಗಿ;
    }

    ದ್ವಿಗುಣ ಮೊತ್ತ         = 0.0;
    ದ್ವಿಗುಣ ಮೊತ್ತ_squared = 0.0;

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ & loss : ಫಲಿತಾಂಶ->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ loss_scaled = ಫಲಿತಾಂಶ->loss_ಪ್ರತಿ_datapoint ? loss*ಫಲಿತಾಂಶ->opt_period : loss;
        ಮೊತ್ತ         += loss_scaled;
        ಮೊತ್ತ_squared += loss_scaled*loss_scaled;
    }

    ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ ಸರಾಸರಿ_2 = ಮೊತ್ತ/nbatches;
    *loss = ಫಲಿತಾಂಶ->loss_ಪ್ರತಿ_datapoint ? ಸರಾಸರಿ_2 : ಮೊತ್ತ;

    ಆದರೆ (!unc) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (nbatches < 2) {
        *unc = NAN;
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ var_ಮೊತ್ತ = ಮೊತ್ತ_squared/nbatches - ಸರಾಸರಿ_2*ಸರಾಸರಿ_2; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = ಫಲಿತಾಂಶ->loss_ಪ್ರತಿ_datapoint ? ವರ್ಗಮೂಲ(var_ಮೊತ್ತ / (nbatches - 1)) : ವರ್ಗಮೂಲ(var_ಮೊತ್ತ * nbatches/(nbatches - 1));
}

ಶೂನ್ಯ ggml_opt_ಫಲಿತಾಂಶ_pred(ggml_opt_ಫಲಿತಾಂಶ_t ಫಲಿತಾಂಶ, int32_t * pred) {
    ಗಾಗಿ (size_t i = 0; i < ಫಲಿತಾಂಶ->pred.size(); ++i) {
        pred[i] = ಫಲಿತಾಂಶ->pred[i];
    }
}

ಶೂನ್ಯ ggml_opt_ಫಲಿತಾಂಶ_accuracy(ggml_opt_ಫಲಿತಾಂಶ_t ಫಲಿತಾಂಶ, ದ್ವಿಗುಣ * accuracy, ದ್ವಿಗುಣ * unc) {
    *accuracy = ಫಲಿತಾಂಶ->ncorrect >= 0 ? ದ್ವಿಗುಣ(ಫಲಿತಾಂಶ->ncorrect) / ದ್ವಿಗುಣ(ಫಲಿತಾಂಶ->ndata) : NAN;

    ಆದರೆ (!unc) {
        ಹಿಂತಿರುಗಿ;
    }

    *unc = ಫಲಿತಾಂಶ->ncorrect >= 0 && ಫಲಿತಾಂಶ->ndata >= 2 ?
        ವರ್ಗಮೂಲ((*accuracy) * (1.0 - (*accuracy)) / ದ್ವಿಗುಣ(ಫಲಿತಾಂಶ->ndata - 1)) : NAN;
}

// ====== Computation ======

ಶೂನ್ಯ ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        ರಚನೆ ggml_context * ctx_compute,
        ರಚನೆ ggml_cgraph  * gf,
        ರಚನೆ ggml_tensor  * inputs,
        ರಚನೆ ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

ಶೂನ್ಯ ggml_opt_alloc(ggml_opt_context_t opt_ctx, ತಾರ್ಕಿಕ backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    ಆದರೆ (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_ನಕ್ಷೆ_reset(opt_ctx->gb_grad);
    }
    ಆದರೆ (backward) {
        ಸ್ಥಿರಾಂಕ int32_t opt_i_ಮುಂದಿನ = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_ಮುಂದಿನ == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } ಇಲ್ಲದಿದ್ದರೆ {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    ಆದರೆ (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    ರಚನೆ ggml_cgraph * graph = ಶೂನ್ಯಸೂಚಕ;
    ಆಯ್ಕೆ (opt_ctx->build_type) {
        ಸ್ಥಿತಿ GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } ನಿಲ್ಲಿಸು;
    }
    GGML_ASSERT(graph);

    ಆದರೆ (opt_ctx->allocated_ನಕ್ಷೆ == graph) {
        opt_ctx->eval_ready = ನಿಜ;
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    ಆದರೆ (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_ನಕ್ಷೆ_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ ಶೂನ್ಯಸೂಚಕ,
            /*.no_alloc   =*/ ನಿಜ,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_ನಕ್ಷೆ_copy = dup_ನಕ್ಷೆ(opt_ctx->ctx_copy, graph);
    } ಇಲ್ಲದಿದ್ದರೆ {
        opt_ctx->allocated_ನಕ್ಷೆ_copy = graph;
    }

    ggml_backend_sched_alloc_ನಕ್ಷೆ(opt_ctx->backend_sched, opt_ctx->allocated_ನಕ್ಷೆ_copy);
    opt_ctx->allocated_ನಕ್ಷೆ = graph;

    opt_ctx->eval_ready = ನಿಜ;
}

ಶೂನ್ಯ ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_ಫಲಿತಾಂಶ_t ಫಲಿತಾಂಶ) {
    GGML_ASSERT(opt_ctx->eval_ready);
    ಆದರೆ (opt_ctx->allocated_ನಕ್ಷೆ == opt_ctx->gb_opt) {
        ಸ್ಥಿರಾಂಕ ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        ಆಯ್ಕೆ (opt_ctx->optimizer) {
            ಸ್ಥಿತಿ GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                ದಶಮಾಂಶ * adamw_par_ದತ್ತಾಂಶ = ggml_get_ದತ್ತಾಂಶ_f32(opt_ctx->opt_ಹೆಜ್ಜೆ_params);
                adamw_par_ದತ್ತಾಂಶ[0] = opt_pars.adamw.alpha;
                adamw_par_ದತ್ತಾಂಶ[1] = opt_pars.adamw.beta1;
                adamw_par_ದತ್ತಾಂಶ[2] = opt_pars.adamw.beta2;
                adamw_par_ದತ್ತಾಂಶ[3] = opt_pars.adamw.eps;
                adamw_par_ದತ್ತಾಂಶ[4] = opt_pars.adamw.wd;
                adamw_par_ದತ್ತಾಂಶ[5] = beta1h;
                adamw_par_ದತ್ತಾಂಶ[6] = beta2h;
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                ದಶಮಾಂಶ * sgd = ggml_get_ದತ್ತಾಂಶ_f32(opt_ctx->opt_ಹೆಜ್ಜೆ_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_ನಕ್ಷೆ_compute(opt_ctx->backend_sched, opt_ctx->allocated_ನಕ್ಷೆ_copy);
    opt_ctx->iter += opt_ctx->allocated_ನಕ್ಷೆ == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    ಆದರೆ (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = ಶೂನ್ಯಸೂಚಕ;
        opt_ctx->gb_grad              = ಶೂನ್ಯಸೂಚಕ;
        opt_ctx->gb_opt               = ಶೂನ್ಯಸೂಚಕ;
        opt_ctx->allocated_ನಕ್ಷೆ      = ಶೂನ್ಯಸೂಚಕ;
        opt_ctx->allocated_ನಕ್ಷೆ_copy = ಶೂನ್ಯಸೂಚಕ;
    }

    opt_ctx->eval_ready = ಸುಳ್ಳು;

    ಆದರೆ (!ಫಲಿತಾಂಶ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ಫಲಿತಾಂಶ->ndata == 0) {
        ಫಲಿತಾಂಶ->loss_ಪ್ರತಿ_datapoint = opt_ctx->loss_ಪ್ರತಿ_datapoint;
        ಫಲಿತಾಂಶ->opt_period         = opt_ctx->opt_period;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(ಫಲಿತಾಂಶ->loss_ಪ್ರತಿ_datapoint == opt_ctx->loss_ಪ್ರತಿ_datapoint);
        GGML_ASSERT(ಫಲಿತಾಂಶ->opt_period         == opt_ctx->opt_period);
    }

    ಸ್ಥಿರಾಂಕ int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(ಫಲಿತಾಂಶ->ndata == ndata*int64_t(ಫಲಿತಾಂಶ->loss.size()) && "varying batch size not supported");
    ಫಲಿತಾಂಶ->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    ದಶಮಾಂಶ loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    ಫಲಿತಾಂಶ->loss.push_back(loss);

    ಆದರೆ (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.ದತ್ತಾಂಶ(), 0, ggml_nbytes(opt_ctx->pred));
        ಫಲಿತಾಂಶ->pred.insert(ಫಲಿತಾಂಶ->pred.end(), pred.begin(), pred.end());
    }

    ಆದರೆ (!opt_ctx->ncorrect || ಫಲಿತಾಂಶ->ncorrect < 0) {
        ಫಲಿತಾಂಶ->ncorrect = -1;
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    ಫಲಿತಾಂಶ->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

ಶೂನ್ಯ ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_ಫಲಿತಾಂಶ_t       ಫಲಿತಾಂಶ_train,
        ggml_opt_ಫಲಿತಾಂಶ_t       ಫಲಿತಾಂಶ_eval,
        int64_t                 idata_ವಿಭಜಿಸು,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    ರಚನೆ ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    ರಚನೆ ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    ರಚನೆ ggml_tensor * ದತ್ತಾಂಶ   = ggml_opt_dataset_ದತ್ತಾಂಶ(dataset);
    GGML_ASSERT(ದತ್ತಾಂಶ->ne[0] == inputs->ne[0]);

    ಸ್ಥಿರಾಂಕ int64_t ndata       =   ದತ್ತಾಂಶ->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(ದತ್ತಾಂಶ->ne[1] % inputs->ne[1] == 0);
    ಸ್ಥಿರಾಂಕ int64_t nbatches = ndata/ndata_batch;

    idata_ವಿಭಜಿಸು = idata_ವಿಭಜಿಸು < 0 ? ndata : idata_ವಿಭಜಿಸು;
    GGML_ASSERT(idata_ವಿಭಜಿಸು % ndata_batch == 0);
    ಸ್ಥಿರಾಂಕ int64_t ibatch_ವಿಭಜಿಸು = idata_ವಿಭಜಿಸು / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_ಆರಂಭ = ggml_ಸಮಯ_us();
    ಗಾಗಿ (; ibatch < ibatch_ವಿಭಜಿಸು; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ ನಿಜ);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, ಫಲಿತಾಂಶ_train);
        ಆದರೆ (callback_train) {
            callback_train(ನಿಜ, opt_ctx, dataset, ಫಲಿತಾಂಶ_train, ibatch+1, ibatch_ವಿಭಜಿಸು, t_loop_ಆರಂಭ);
        }
    }
    t_loop_ಆರಂಭ = ggml_ಸಮಯ_us();
    ಗಾಗಿ (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ ಸುಳ್ಳು);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, ಫಲಿತಾಂಶ_eval);
        ಆದರೆ (callback_eval) {
            callback_eval(ಸುಳ್ಳು, opt_ctx, dataset, ಫಲಿತಾಂಶ_eval, ibatch+1-ibatch_ವಿಭಜಿಸು, nbatches-ibatch_ವಿಭಜಿಸು, t_loop_ಆರಂಭ);
        }
    }
}

ಶೂನ್ಯ ggml_opt_epoch_callback_ಪ್ರಗತಿ_bar(
        ತಾರ್ಕಿಕ               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_ಫಲಿತಾಂಶ_t  ಫಲಿತಾಂಶ,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_ಆರಂಭ_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_ಉದ್ದಳತೆ = 8;
    ಸ್ಥಿರಾಂಕ int64_t ibatch8 = 8 * ibatch;
    ಗಾಗಿ (int64_t j = 0; j < bar_ಉದ್ದಳತೆ; ++j) {
        ಆದರೆ        (ibatch_max * (8*j + 8) / bar_ಉದ್ದಳತೆ < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ibatch_max * (8*j + 7) / bar_ಉದ್ದಳತೆ < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ibatch_max * (8*j + 6) / bar_ಉದ್ದಳತೆ < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ibatch_max * (8*j + 5) / bar_ಉದ್ದಳತೆ < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ibatch_max * (8*j + 4) / bar_ಉದ್ದಳತೆ < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ibatch_max * (8*j + 3) / bar_ಉದ್ದಳತೆ < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ibatch_max * (8*j + 2) / bar_ಉದ್ದಳತೆ < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ibatch_max * (8*j + 1) / bar_ಉದ್ದಳತೆ < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } ಇಲ್ಲದಿದ್ದರೆ {
            fprintf(stderr, " ");
        }
    }

    ಸ್ಥಿರಾಂಕ int64_t batch_ಗಾತ್ರ = ggml_opt_inputs(opt_ctx)->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t idata      = ibatch*batch_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t idata_max  = ibatch_max*batch_ಗಾತ್ರ;

    ದ್ವಿಗುಣ loss;
    ದ್ವಿಗುಣ loss_unc;
    ggml_opt_ಫಲಿತಾಂಶ_loss(ಫಲಿತಾಂಶ, &loss, &loss_unc);

    ದ್ವಿಗುಣ accuracy;
    ದ್ವಿಗುಣ accuracy_unc;
    ggml_opt_ಫಲಿತಾಂಶ_accuracy(ಫಲಿತಾಂಶ, &accuracy, &accuracy_unc);

    ಸ್ಥಿರಾಂಕ int64_t t_ibatch_us = ggml_ಸಮಯ_us() - t_ಆರಂಭ_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    ಸ್ಥಿರಾಂಕ int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    ಸ್ಥಿರಾಂಕ int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    ಸ್ಥಿರಾಂಕ int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    ಸ್ಥಿರಾಂಕ int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    ಸ್ಥಿರಾಂಕ int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    ಆದರೆ (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_ಬಳಸದ(dataset);
}

ಶೂನ್ಯ ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        ಎಣಿಕೆ ggml_opt_loss_type         loss_type,
        ಎಣಿಕೆ ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        ದಶಮಾಂಶ                           val_ವಿಭಜಿಸು,
        ತಾರ್ಕಿಕ                            silent) {
    ggml_ಸಮಯ_init();
    ಸ್ಥಿರಾಂಕ int64_t t_ಆರಂಭ_us = ggml_ಸಮಯ_us();

    ಸ್ಥಿರಾಂಕ int64_t ndata           = ggml_opt_dataset_ದತ್ತಾಂಶ(dataset)->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    ಸ್ಥಿರಾಂಕ int64_t opt_period       = nbatch_logical / nbatch_physical;
    ಸ್ಥಿರಾಂಕ int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_ವಿಭಜಿಸು >= 0.0f);
    GGML_ASSERT(val_ವಿಭಜಿಸು <  1.0f);
    ಸ್ಥಿರಾಂಕ int64_t ibatch_ವಿಭಜಿಸು = int64_t(((1.0f - val_ವಿಭಜಿಸು) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    ಸ್ಥಿರಾಂಕ int64_t idata_ವಿಭಜಿಸು  = ibatch_ವಿಭಜಿಸು * nbatch_physical;

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
    ಆದರೆ (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_ಫಲಿತಾಂಶ_t ಫಲಿತಾಂಶ_train = ggml_opt_ಫಲಿತಾಂಶ_init();
    ggml_opt_ಫಲಿತಾಂಶ_t ಫಲಿತಾಂಶ_val   = ggml_opt_ಫಲಿತಾಂಶ_init();

    ggml_opt_epoch_callback epoch_callback = silent ? ಶೂನ್ಯಸೂಚಕ : ggml_opt_epoch_callback_ಪ್ರಗತಿ_bar;

    ಗಾಗಿ (; epoch <= nepoch; ++epoch) {
        ಆದರೆ (nbatch_logical < idata_ವಿಭಜಿಸು) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_ವಿಭಜಿಸು);
        }

        ggml_opt_ಫಲಿತಾಂಶ_reset(ಫಲಿತಾಂಶ_train);
        ggml_opt_ಫಲಿತಾಂಶ_reset(ಫಲಿತಾಂಶ_val);

        ಆದರೆ (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, ಫಲಿತಾಂಶ_train, ಫಲಿತಾಂಶ_val, idata_ವಿಭಜಿಸು, epoch_callback, epoch_callback);
        ಆದರೆ (!silent) {
            fprintf(stderr, "\n");
        }
    }

    ಆದರೆ (!silent) {
        int64_t t_ಒಟ್ಟು_s = (ggml_ಸಮಯ_us() - t_ಆರಂಭ_us) / 1000000;
        ಸ್ಥಿರಾಂಕ int64_t t_ಒಟ್ಟು_h = t_ಒಟ್ಟು_s / 3600;
        t_ಒಟ್ಟು_s -= t_ಒಟ್ಟು_h * 3600;
        ಸ್ಥಿರಾಂಕ int64_t t_ಒಟ್ಟು_m = t_ಒಟ್ಟು_s / 60;
        t_ಒಟ್ಟು_s -= t_ಒಟ್ಟು_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_ಒಟ್ಟು_h, t_ಒಟ್ಟು_m, t_ಒಟ್ಟು_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_ಫಲಿತಾಂಶ_free(ಫಲಿತಾಂಶ_train);
    ggml_opt_ಫಲಿತಾಂಶ_free(ಫಲಿತಾಂಶ_val);
}

ಎಣಿಕೆ ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    ಹಿಂತಿರುಗಿ c->optimizer;
}

GGML_API ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_opt_optimizer_name(ಎಣಿಕೆ ggml_opt_optimizer_type o) {
    ಆಯ್ಕೆ (o) {
        ಸ್ಥಿತಿ GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            ಹಿಂತಿರುಗಿ "adamw";
        ಸ್ಥಿತಿ GGML_OPT_OPTIMIZER_TYPE_SGD:
            ಹಿಂತಿರುಗಿ "sgd";
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ "undefined";
    };
}
