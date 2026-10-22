#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/decoder.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/partial_shape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <optional>
#ਅਨ੍ਤਰ੍ਭਾਵ <set>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਬਣਤਰ ModelParams {
    ਪੂਰਨ ctx = -1;
    ਪੂਰਨ ctx_ਪ੍ਰਤੀ_seq = -1;
    ਪੂਰਨ ctx_ਪ੍ਰਤੀ_seq_swa = -1;
    ਪੂਰਨ n_seq = 1;
    ਪੂਰਨ n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<ਪੂਰਨ, ਪੂਰਨ> n_heads_kv_ਪ੍ਰਤੀ_layer;
    ਪੂਰਨ ਸਿਰ_ਮਾਪ = -1;
    ਪੂਰਨ state_ਮਾਪ = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    ਪੂਰਨ n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::ਸਦਿਸ਼<ਪੂਰਨ> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    ਅਚਲ ggml_tensor * swa_ਨਕਾਬ = nullptr;

    std::ਸਦਿਸ਼<std::string> kv_ਨਾਂ;
    size_t kv_ਬਫਰ_ctx_id = 0;

    bool same_rope_params(ਅਚਲ ModelParams & other) ਅਚਲ {
        ਵਾਪਸ mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, ਆਕਾਰ(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(ਅਚਲ ModelParams & other) ਅਚਲ {
        ਵਾਪਸ same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(ਅਚਲ ModelParams & other) ਅਚਲ {
        ਵਾਪਸ same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_ਬਫਰ_ਬਦਲਿਆ(ਅਚਲ ModelParams & other) ਅਚਲ { ਵਾਪਸ kv_ਬਫਰ_ctx_id != other.kv_ਬਫਰ_ctx_id; }
};

ਬਣਤਰ ComputeParams {
    ਪੂਰਨ n_seq_ਸਰਗਰਮ = 1;
    ਪੂਰਨ seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ = 0;
    ਪੂਰਨ attention_ਮਾਪ = -1;
    ਪੂਰਨ attention_ਮਾਪ_swa = -1;
    ਪੂਰਨ attention_ਮਾਪ_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    ਪੂਰਨ swa_ਖਿੜਕੀ = -1;
    ਪੂਰਨ ਨਿਵੇਸ਼_len = -1;
    ਪੂਰਨ token_len_ਪ੍ਰਤੀ_seq = -1;
    ਪੂਰਨ past_kv_len = -1;
    ਪੂਰਨ output_len = 1;

    ਪੂਰਨ cache_rs_reset_idx = -1;
    ਪੂਰਨ cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    ਪੂਰਨ s_copy_ਸਰਗਰਮ_ਥਾਂ_len = -1;
    // SSM/DeltaNet models otionally reorder slots of state cache, to make the active slots contiguous
    // leaf_5 is the inp->s_copy in llama-graph.cpp, eg if there are 8 slots in total and slot 3 and 7
    // are active in the current batch, leaf_5 will be [3, 7, 5, 6, 4]
    //  6: [     2,     1,     1,     1] VIEW                  (view)
    //      [     2,     1,     1,     1]            0: NONE        leaf_5
    //  7: [ 18432,     2,     1,     1] GET_ROWS             conv_states-0
    //      [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    //      [     2,     1,     1,     1]            1: VIEW         (view)
    //  8: [     0,     1,     1,     1] VIEW                  (view)
    //      [     2,     1,     1,     1]            0: NONE        leaf_5
    //  9: [ 18432,     0,     1,     1] GET_ROWS             node_9
    //      [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    //      [     0,     1,     1,     1]            1: VIEW         (view)
    // 10: [ 18432,     0,     1,     1] VIEW                 cache_r_l0 (view)
    //      [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 11: [ 18432,     0,     1,     1] CPY                  cache_r_l0 (view) (copy of )
    //      [ 18432,     0,     1,     1]            0: GET_ROWS    node_9
    //      [ 18432,     0,     1,     1]            1: VIEW        cache_r_l0 (view)

    ਬਣਤਰ RsWriteback {
        ਪੂਰਨ ਥਾਂ_begin = 0;  // first cache slot written by the CPY
        ਪੂਰਨ src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<ਪੂਰਨ> ਕੱਢੋ_layer_from_name(ਅਚਲ std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_ਜੋੜ_ਜੋੜੋ(ਅਚਲ ggml_tensor * ਗੰਢ);

class GgmlOvDecoder : ਜਨਤਕ ov::frontend::ggml::GgmlDecoder {
ਜਨਤਕ:
    ਸਥਿਰ std::string get_tensor_name(ਅਚਲ ggml_cgraph * cgraph, ਅਚਲ ggml_tensor * tensor);
    ਬਣਤਰ ਗੰਢ_Info {
        ggml_tensor * ਗੰਢ;
        std::string ਗੰਢ_name;
        std::string ਗੰਢ_op_type;
        std::map<std::string, ggml_tensor *> ਗੰਢ_inputs;
        std::map<std::string, std::ਸਦਿਸ਼<std::pair<std::string, ggml_tensor *>>> ਗੰਢ_inputs_views;
        std::ਸਦਿਸ਼<std::string> ਗੰਢ_inputs_ਨਾਂ;
        ਪੂਰਨ ਗੰਢ_op_case = 0;
        ਖਾਲੀ * ਅੰਕੜੇ_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::ਗੰਢ_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  ਪੂਰਨ prefill_chunk_ਮਾਪ = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::ਗੰਢ_2>> & model_weights);

    ਆਭਾਸੀ ov::Any get_attribute(ਅਚਲ std::string & name) ਅਚਲ override {
        ਵਾਪਸ nullptr;
        GGML_ਅਣਵਰਤਿਆ(name);
    }

    ਆਭਾਸੀ ov::PartialShape get_ਨਿਵੇਸ਼_shape(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ override;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<size_t> get_ਨਿਵੇਸ਼_stride(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ override;

    ਆਭਾਸੀ size_t get_view_ਨਿਵੇਸ਼_ਮਾਪ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ override;

    ਆਭਾਸੀ size_t get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ override;

    ਆਭਾਸੀ size_t get_view_ਨਿਵੇਸ਼_src_ਵਿੱਥ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ override;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<size_t> get_view_ਨਿਵੇਸ਼_stride(ਪੂਰਨ ਗੰਢ_idx,
                                                      ਅਚਲ std::string & name,
                                                      size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ override;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<size_t> get_view_ਨਿਵੇਸ਼_src_stride(ਪੂਰਨ ਗੰਢ_idx,
                                                          ਅਚਲ std::string & name,
                                                          size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ override;

    ਆਭਾਸੀ ov::Shape get_view_ਨਿਵੇਸ਼_ggml_shape(ਪੂਰਨ ਗੰਢ_idx,
                                                ਅਚਲ std::string & name,
                                                size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ override;

    ਆਭਾਸੀ ov::Shape get_view_ਨਿਵੇਸ਼_src_ggml_shape(ਪੂਰਨ ਗੰਢ_idx,
                                                    ਅਚਲ std::string & name,
                                                    size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ override;

    ਆਭਾਸੀ ov::PartialShape get_view_ਨਿਵੇਸ਼_ov_shape(ਪੂਰਨ ਗੰਢ_idx,
                                                     ਅਚਲ std::string & name,
                                                     size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ override;

    ਆਭਾਸੀ ov::PartialShape get_view_ਨਿਵੇਸ਼_src_ov_shape(ਪੂਰਨ ਗੰਢ_idx,
                                                         ਅਚਲ std::string & name,
                                                         size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ override;

    ਆਭਾਸੀ std::string get_view_ਨਿਵੇਸ਼_name(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ override;

    ਆਭਾਸੀ std::string get_view_ਨਿਵੇਸ਼_src_name(ਪੂਰਨ ਗੰਢ_idx,
                                                ਅਚਲ std::string & name,
                                                size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ override;

    ਆਭਾਸੀ ov::element::Type get_ਨਿਵੇਸ਼_type(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ override;

    ਆਭਾਸੀ size_t get_ਨਿਵੇਸ਼_ਮਾਪ() ਅਚਲ override;

    ਆਭਾਸੀ size_t get_ਨਿਵੇਸ਼_ਮਾਪ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ ਖਾਲੀ get_ਨਿਵੇਸ਼_ਗੰਢ(size_t ਨਿਵੇਸ਼_ਪੋਰਟ_idx,
                                std::string & ਉਤਪਾਦਕ_name,
                                std::string & ਉਤਪਾਦਕ_output_ਪੋਰਟ_name,
                                size_t & ਉਤਪਾਦਕ_output_ਪੋਰਟ_ਸੂਚਕਾਂਕ) ਅਚਲ override {
        GGML_ਅਣਵਰਤਿਆ(ਨਿਵੇਸ਼_ਪੋਰਟ_idx);
        GGML_ਅਣਵਰਤਿਆ(ਉਤਪਾਦਕ_name);
        GGML_ਅਣਵਰਤਿਆ(ਉਤਪਾਦਕ_output_ਪੋਰਟ_name);
        GGML_ਅਣਵਰਤਿਆ(ਉਤਪਾਦਕ_output_ਪੋਰਟ_ਸੂਚਕਾਂਕ);
    }

    ਆਭਾਸੀ std::ਸਦਿਸ਼<std::string> get_ਨਿਵੇਸ਼_ਨਾਂ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ ov::PartialShape get_output_shape(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ ov::element::Type get_output_type(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<size_t> get_output_stride(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ int32_t * get_ਨਿਵੇਸ਼_op_params(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ override;

    ਆਭਾਸੀ int32_t * get_output_op_params(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ size_t get_output_op_ਵਿੱਥ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<std::string> get_output_ਨਾਂ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ std::string get_inplace_op_src(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ bool is_view_like_alias_ਦਾ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & view_src_name) ਅਚਲ override;

    ਆਭਾਸੀ ਅਚਲ std::string & get_op_type() ਅਚਲ override;

    ਆਭਾਸੀ ਅਚਲ std::string & get_op_type(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ ਅਚਲ std::string & get_op_name() ਅਚਲ override;

    ਆਭਾਸੀ ਅਚਲ std::string & get_op_name(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ int32_t get_op_dynamic_dim(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override;

    ਆਭਾਸੀ ਖਾਲੀ ਦੇਖੋ_subgraph(
        std::function<ਖਾਲੀ(std::shared_ptr<GgmlDecoder>, ਪੂਰਨ ਗੰਢ_idx)> ਗੰਢ_visitor) ਅਚਲ override;

    ggml_tensor * get_ਨਿਵੇਸ਼_ggml_tensor(ਅਚਲ std::string & name) ਅਚਲ { ਵਾਪਸ m_inputs.at(name); }

    ਆਭਾਸੀ ਪੂਰਨ get_op_case(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ override { ਵਾਪਸ m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_op_case; }

    ਆਭਾਸੀ ਅਚਲ std::map<std::string, ov::frontend::ggml::Model_ਨਿਵੇਸ਼_Info> & get_model_inputs() ਅਚਲ override {
        ਵਾਪਸ m_model_inputs;
    }

    ਆਭਾਸੀ ਅਚਲ std::map<std::string, ov::frontend::ggml::Model_Extra_ਨਿਵੇਸ਼_Info> & get_model_extra_inputs() ਅਚਲ override {
        ਵਾਪਸ m_model_extra_inputs;
    }

    ਆਭਾਸੀ ਅਚਲ std::map<std::string, std::shared_ptr<ov::ਗੰਢ_2>> & get_model_weights() ਅਚਲ override {
        ਵਾਪਸ m_model_weights;
    }

    ਆਭਾਸੀ std::ਸੈੱਟ<std::string> get_model_output_ਨਾਂ() ਅਚਲ override { ਵਾਪਸ m_model_output_ਨਾਂ; }

    ਅਚਲ std::map<std::string, ggml_tensor *> & get_model_outputs() ਅਚਲ { ਵਾਪਸ m_model_outputs; }

    ਆਭਾਸੀ ਪੂਰਨ get_ctx_ਮਾਪ() ਅਚਲ { ਵਾਪਸ m_model_params.ctx; }

    ਆਭਾਸੀ ਪੂਰਨ get_ctx_ਪ੍ਰਤੀ_seq() ਅਚਲ { ਵਾਪਸ m_model_params.ctx_ਪ੍ਰਤੀ_seq; }

    ਆਭਾਸੀ ਪੂਰਨ get_ctx_ਪ੍ਰਤੀ_seq_swa() ਅਚਲ { ਵਾਪਸ m_model_params.ctx_ਪ੍ਰਤੀ_seq_swa; }

    ਆਭਾਸੀ ਪੂਰਨ get_n_seq() ਅਚਲ { ਵਾਪਸ m_model_params.n_seq; }

    ਆਭਾਸੀ ਪੂਰਨ is_swa_layer(ਪੂਰਨ layer) ਅਚਲ override {
        ਵਾਪਸ std::ਲੱਭੋ(m_model_params.swa_layers.begin(), m_model_params.swa_layers.ਸਿਰਾ(), layer) !=
               m_model_params.swa_layers.ਸਿਰਾ();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    ਪੂਰਨ get_n_heads_kv_for_layer(ਪੂਰਨ layer) ਅਚਲ {
        ਸਵੈਚਾਲਿਤ it = m_model_params.n_heads_kv_ਪ੍ਰਤੀ_layer.ਲੱਭੋ(layer);
        ਵਾਪਸ it != m_model_params.n_heads_kv_ਪ੍ਰਤੀ_layer.ਸਿਰਾ() ? it->ਦੂਜਾ : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    ਪੂਰਨ get_n_heads_kv_for_tensor(ਅਚਲ ggml_tensor * kv_tensor) ਅਚਲ {
        ਜੇ (ਸਵੈਚਾਲਿਤ layer = ਕੱਢੋ_layer_from_name(std::string(kv_tensor->name)); layer.has_ਮੁੱਲ()) {
            ਵਾਪਸ get_n_heads_kv_for_layer(layer.ਮੁੱਲ());
        }
        ਵਾਪਸ m_model_params.n_heads_kv;
    }

    ਪੂਰਨ get_past_kv_len() ਅਚਲ { ਵਾਪਸ m_compute_params.past_kv_len; }

    ਪੂਰਨ get_ਨਿਵੇਸ਼_len() ਅਚਲ { ਵਾਪਸ m_compute_params.ਨਿਵੇਸ਼_len; }

    ਆਭਾਸੀ int32_t * get_rope_params() ਅਚਲ override { ਵਾਪਸ const_cast<int32_t *>(m_model_params.rope_params); }

    ਆਭਾਸੀ bool has_mixed_rope_params() ਅਚਲ override { ਵਾਪਸ m_model_params.mixed_rope_params; }

    ਆਭਾਸੀ ਪੂਰਨ get_ssm_state_ਮਾਪ() ਅਚਲ override { ਵਾਪਸ m_model_params.state_ਮਾਪ; }

    ਆਭਾਸੀ std::map<std::string, std::string> get_kv_param_res_ਨਾਂ() ਅਚਲ override;

    ਆਭਾਸੀ bool is_static() ਅਚਲ override { ਵਾਪਸ m_is_static; }

    ਆਭਾਸੀ bool is_stateful() ਅਚਲ override { ਵਾਪਸ m_is_stateful; }

    ਪੂਰਨ get_static_n_ਟੋਕਨ() ਅਚਲ { ਵਾਪਸ m_is_prefill ? m_prefill_chunk_ਮਾਪ : 1; }

    ਆਭਾਸੀ bool is_splited_model() ਅਚਲ override { ਵਾਪਸ m_model_is_splitted; }

    ov::PartialShape get_ਗ੍ਰਾਫ_ਨਿਵੇਸ਼_shape(ਅਚਲ ggml_tensor * op,
                                           ਅਚਲ ggml_tensor * ਨਿਵੇਸ਼,
                                           ਪੂਰਨ dynamic_dim_ਸੂਚਕਾਂਕ = -1) ਅਚਲ;

    ਸਥਿਰ ਖਾਲੀ dump_cgraph(ਅਚਲ ggml_cgraph * cgraph, std::string & filename);

    ਸਥਿਰ std::shared_ptr<ov::ਗੰਢ_2> create_ਭਾਰ_ਗੰਢ(ggml_tensor * tensor, bool naive = false);

    ਸਥਿਰ std::map<std::string, std::shared_ptr<ov::ਗੰਢ_2>> create_ਭਾਰ_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    ਸਥਿਰ std::ਸੈੱਟ<std::string> collect_ਭਾਰ_ਨਾਂ(ggml_cgraph * cgraph);

    ਅਚਲ ggml_tensor * get_tensor_ਵਰਤਿਆ_op(ਅਚਲ ggml_tensor * tensor) ਅਚਲ;

    ਅਚਲ ggml_tensor * get_tensor_from_name(ਅਚਲ std::string & name) ਅਚਲ;

    ਖਾਲੀ clear_model_weights() { m_model_weights.clear(); }

    ਸਥਿਰ std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() ਅਚਲ { ਵਾਪਸ m_model_params; }

    ComputeParams get_compute_params() ਅਚਲ { ਵਾਪਸ m_compute_params; }

    ਖਾਲੀ ਸੈੱਟ_model_params(ਅਚਲ ModelParams & model_params) { m_model_params = model_params; }

    ਖਾਲੀ ਸੈੱਟ_compute_params(ਅਚਲ ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    ਪੂਰਨ m_prefill_chunk_ਮਾਪ = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    ਸਥਿਰ ov::Shape get_shape(ਅਚਲ ggml_tensor * tensor);
    ਸਥਿਰ std::ਸਦਿਸ਼<size_t> get_stride(ਅਚਲ ggml_tensor * tensor);
    ਸਥਿਰ ov::element::Type get_ov_type(ਅਚਲ ggml_tensor * tensor);
    ਸਥਿਰ std::string compute_op_type(ਅਚਲ ggml_tensor * ਗੰਢ);
    ਖਾਲੀ ਜੋੜੋ_extra_inputs();

    ਖਾਲੀ update_io(ggml_cgraph * cgraph);
    bool is_bound_ਮੰਜ਼ਿਲ(ਅਚਲ ggml_cgraph * cgraph) ਅਚਲ;

    ਸਥਿਰ bool is_inp_tok(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ op->op == GGML_OP_GET_ਪੰਕਤੀਆਂ && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    ਸਥਿਰ bool is_inp_pos(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    ਸਥਿਰ ਪੂਰਨ get_inp_pos_n_planes(ਅਚਲ ggml_tensor * op) {
        ਅਚਲ ਪੂਰਨ ਬਹੁਲਕ = op->op_params[2];
        ਵਾਪਸ (ਬਹੁਲਕ == GGML_ROPE_TYPE_IMROPE || ਬਹੁਲਕ == GGML_ROPE_TYPE_VISION || (ਬਹੁਲਕ & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    ਸਥਿਰ bool is_inp_emb(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ tensor->op == GGML_OP_GET_ਪੰਕਤੀਆਂ && op->op == GGML_OP_RMS_NORM;
    }

    ਸਥਿਰ bool is_inp_ਨਕਾਬ(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    ਸਥਿਰ bool is_inp_ਔਸਤ(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    ਸਥਿਰ bool is_inp_scale_ਪੰਕਤੀਆਂ(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    ਸਥਿਰ bool is_rope_freqs_ਭਾਰ(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    ਸਤਰਅੰਦਰ ਸਥਿਰ bool is_recurrent_cache(ਅਚਲ ggml_tensor * tensor) {
        ਵਾਪਸ tensor != nullptr && (ਲੜੀ_ਤੁਲਨਾ(tensor->name, "cache_r_l", ਲੜੀ_ਲੰਬਾਈ("cache_r_l")) == 0 ||
                                     ਲੜੀ_ਤੁਲਨਾ(tensor->name, "cache_s_l", ਲੜੀ_ਲੰਬਾਈ("cache_s_l")) == 0 ||
                                     ਲੜੀ_ਤੁਲਨਾ(tensor->name, "cache_ple_r_l", ਲੜੀ_ਲੰਬਾਈ("cache_ple_r_l")) == 0);
    }

    ਸਤਰਅੰਦਰ ਸਥਿਰ bool is_cache(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    ਸਤਰਅੰਦਰ ਸਥਿਰ bool is_kvcache(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਜੇ (tensor == nullptr || is_recurrent_cache(tensor)) {
            ਵਾਪਸ false;
        }
        ਵਾਪਸ (tensor->ਬਫਰ != nullptr && tensor->ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ && op->src[2] == tensor);
    }

    ਸਤਰਅੰਦਰ ਸਥਿਰ bool is_conv_state_writeback(ਅਚਲ ggml_tensor * ਗੰਢ) {
        ਵਾਪਸ ਗੰਢ->op == GGML_OP_CPY && ਗੰਢ->view_src != nullptr && is_recurrent_cache(ਗੰਢ->view_src) &&
               ਗੰਢ->src[0] != nullptr && ਗੰਢ->src[0]->op == GGML_OP_VIEW && ਗੰਢ->src[0]->src[0] != nullptr &&
               ਗੰਢ->src[0]->src[0]->op == GGML_OP_CONCAT && ਗੰਢ->src[1] != nullptr &&
               ਗੰਢ->src[1]->op == GGML_OP_VIEW && ਗੰਢ->src[1]->view_src == ਗੰਢ->view_src;
    }

    ਸਥਿਰ bool is_kv_idx(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ op->op == GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ && op->src[1] == tensor;
    }

    bool is_swa_ਨਕਾਬ(ਅਚਲ ggml_tensor * tensor) ਅਚਲ {
        ਵਾਪਸ m_model_params.swa_ਨਕਾਬ != nullptr && tensor == m_model_params.swa_ਨਕਾਬ;
    }

    ਸਥਿਰ bool is_output_idx(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ op->op == GGML_OP_GET_ਪੰਕਤੀਆਂ && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    ਸਥਿਰ bool is_inp_s_copy(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) {
        ਵਾਪਸ op->op == GGML_OP_GET_ਪੰਕਤੀਆਂ && tensor == op->src[1] &&
               op->src[0]->ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_ANY;
    }

    std::string get_ਗ੍ਰਾਫ_ਨਿਵੇਸ਼_ov_name(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ggml_tensor * op) ਅਚਲ {
        ਜੇ (is_inp_pos(tensor, op)) {
            ਵਾਪਸ "inp_pos";
        }
        ਜੇ (is_inp_emb(tensor, op)) {
            ਵਾਪਸ "embd";
        }
        ਜੇ (is_inp_ਨਕਾਬ(tensor, op)) {
            // Give the two attention masks distinct OV parameter names.
            //
            // An interleaved-SWA model builds one full-attention mask and one sliding-window mask,
            // but build_attn_inp_kq_mask() names them identically, so keying a parameter off
            // tensor->name alone makes the second mask OVERWRITE the first in m_model_inputs: both
            // attention types then read a single parameter, and the windowed layers silently run
            // against an unbanded mask. Disambiguate using the SWA layer set computed in
            // compute_llm_params(), which classifies by mask tensor identity rather than by name.
            //
            // When no SWA layer was found there is only one mask in play, so the plain name is
            // correct and no _swa parameter is created.
            ਜੇ (m_model_params.swa_layers.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਵਾਪਸ "self_kq_mask";
            }
            ਵਾਪਸ is_swa_ਨਕਾਬ(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        ਵਾਪਸ tensor->name;
    }

ਨਿਜੀ:
    ਖਾਲੀ ਸੈੱਟ_ਨਿਵੇਸ਼_output();
    ਪੂਰਨ compute_op_case(ਅਚਲ ggml_tensor * ਗੰਢ) ਅਚਲ;
    bool ਗੰਢ_is_ਵਰਤਿਆ_as_src(ਅਚਲ ਪੂਰਨ ਗੰਢ_idx);
    ਖਾਲੀ compute_model_inputs();
    ਖਾਲੀ compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_ਪੱਤੀ(ਅਚਲ ggml_tensor * tensor) ਅਚਲ;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    ਖਾਲੀ compute_ਗੰਢ_dynamic_dims();

    ਖਾਲੀ validate_cgraph() ਅਚਲ;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_ਨਿਵੇਸ਼_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_ਨਿਵੇਸ਼_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::ਗੰਢ_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::ਸੈੱਟ<std::string> m_model_output_ਨਾਂ;
    std::ਸਦਿਸ਼<ਗੰਢ_Info> m_ਗੰਢ_info_list;
    std::map<ggml_tensor *, ਪੂਰਨ> m_ਗੰਢ_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

ਖਾਲੀ print_tensor_ਪਤਾ_map(ਅਚਲ ggml_cgraph * cgraph);
