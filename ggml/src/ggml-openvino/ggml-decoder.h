#आशय once

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml.h"
#समावेश "openvino/decoder.h"

#समावेश <cstdint>
#समावेश <cstring>
#समावेश <map>
#समावेश <memory>
#समावेश <openvino/core/partial_shape.hpp>
#समावेश <optional>
#समावेश <set>
#समावेश <string>
#समावेश <vector>

काष्ठा ModelParams {
    पूर्णांक ctx = -1;
    पूर्णांक ctx_per_seq = -1;
    पूर्णांक ctx_per_seq_swa = -1;
    पूर्णांक n_seq = 1;
    पूर्णांक n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<पूर्णांक, पूर्णांक> n_heads_kv_per_layer;
    पूर्णांक शीर्ष_आकार = -1;
    पूर्णांक state_आकार = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    पूर्णांक n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::vector<पूर्णांक> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    स्थिर ggml_tensor * swa_mask = nullptr;

    std::vector<std::string> kv_names;
    माप_प्रकार kv_बफर_ctx_id = 0;

    bool same_rope_params(स्थिर ModelParams & other) स्थिर {
        वापस mixed_rope_params == other.mixed_rope_params &&
               स_भेद(rope_params, other.rope_params, माप(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(स्थिर ModelParams & other) स्थिर {
        वापस same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(स्थिर ModelParams & other) स्थिर {
        वापस same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_बफर_changed(स्थिर ModelParams & other) स्थिर { वापस kv_बफर_ctx_id != other.kv_बफर_ctx_id; }
};

काष्ठा ComputeParams {
    पूर्णांक n_seq_active = 1;
    पूर्णांक seq_active_start = 0;
    पूर्णांक attention_आकार = -1;
    पूर्णांक attention_आकार_swa = -1;
    पूर्णांक attention_आकार_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    पूर्णांक swa_window = -1;
    पूर्णांक निवेश_len = -1;
    पूर्णांक token_len_per_seq = -1;
    पूर्णांक past_kv_len = -1;
    पूर्णांक output_len = 1;

    पूर्णांक cache_rs_reset_idx = -1;
    पूर्णांक cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    पूर्णांक s_copy_active_slot_len = -1;
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

    काष्ठा RsWriteback {
        पूर्णांक slot_begin = 0;  // first cache slot written by the CPY
        पूर्णांक src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<पूर्णांक> extract_layer_from_name(स्थिर std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_योग_add(स्थिर ggml_tensor * गाँठ);

class GgmlOvDecoder : खुला ov::frontend::ggml::GgmlDecoder {
खुला:
    अटल std::string get_tensor_name(स्थिर ggml_cgraph * cgraph, स्थिर ggml_tensor * tensor);
    काष्ठा गाँठ_Info {
        ggml_tensor * गाँठ;
        std::string गाँठ_name;
        std::string गाँठ_op_type;
        std::map<std::string, ggml_tensor *> गाँठ_inputs;
        std::map<std::string, std::vector<std::pair<std::string, ggml_tensor *>>> गाँठ_inputs_views;
        std::vector<std::string> गाँठ_inputs_names;
        पूर्णांक गाँठ_op_case = 0;
        व्योम * तथ्य_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::गाँठ_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  पूर्णांक prefill_chunk_आकार = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::गाँठ_2>> & model_weights);

    भव ov::Any get_attribute(स्थिर std::string & name) स्थिर override {
        वापस nullptr;
        GGML_UNUSED(name);
    }

    भव ov::PartialShape get_निवेश_shape(पूर्णांक गाँठ_idx, स्थिर std::string & name) स्थिर override;

    भव std::vector<माप_प्रकार> get_निवेश_stride(पूर्णांक गाँठ_idx, स्थिर std::string & name) स्थिर override;

    भव माप_प्रकार get_view_निवेश_आकार(पूर्णांक गाँठ_idx, स्थिर std::string & name) स्थिर override;

    भव माप_प्रकार get_view_निवेश_offset(पूर्णांक गाँठ_idx, स्थिर std::string & name, माप_प्रकार view_सूचकांक) स्थिर override;

    भव माप_प्रकार get_view_निवेश_src_offset(पूर्णांक गाँठ_idx, स्थिर std::string & name, माप_प्रकार view_सूचकांक) स्थिर override;

    भव std::vector<माप_प्रकार> get_view_निवेश_stride(पूर्णांक गाँठ_idx,
                                                      स्थिर std::string & name,
                                                      माप_प्रकार view_सूचकांक) स्थिर override;

    भव std::vector<माप_प्रकार> get_view_निवेश_src_stride(पूर्णांक गाँठ_idx,
                                                          स्थिर std::string & name,
                                                          माप_प्रकार view_सूचकांक) स्थिर override;

    भव ov::Shape get_view_निवेश_ggml_shape(पूर्णांक गाँठ_idx,
                                                स्थिर std::string & name,
                                                माप_प्रकार view_सूचकांक) स्थिर override;

    भव ov::Shape get_view_निवेश_src_ggml_shape(पूर्णांक गाँठ_idx,
                                                    स्थिर std::string & name,
                                                    माप_प्रकार view_सूचकांक) स्थिर override;

    भव ov::PartialShape get_view_निवेश_ov_shape(पूर्णांक गाँठ_idx,
                                                     स्थिर std::string & name,
                                                     माप_प्रकार view_सूचकांक) स्थिर override;

    भव ov::PartialShape get_view_निवेश_src_ov_shape(पूर्णांक गाँठ_idx,
                                                         स्थिर std::string & name,
                                                         माप_प्रकार view_सूचकांक) स्थिर override;

    भव std::string get_view_निवेश_name(पूर्णांक गाँठ_idx, स्थिर std::string & name, माप_प्रकार view_सूचकांक) स्थिर override;

    भव std::string get_view_निवेश_src_name(पूर्णांक गाँठ_idx,
                                                स्थिर std::string & name,
                                                माप_प्रकार view_सूचकांक) स्थिर override;

    भव ov::element::Type get_निवेश_type(पूर्णांक गाँठ_idx, स्थिर std::string & name) स्थिर override;

    भव माप_प्रकार get_निवेश_आकार() स्थिर override;

    भव माप_प्रकार get_निवेश_आकार(पूर्णांक गाँठ_idx) स्थिर override;

    भव व्योम get_निवेश_गाँठ(माप_प्रकार निवेश_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                माप_प्रकार & producer_output_port_सूचकांक) स्थिर override {
        GGML_UNUSED(निवेश_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_सूचकांक);
    }

    भव std::vector<std::string> get_निवेश_names(पूर्णांक गाँठ_idx) स्थिर override;

    भव ov::PartialShape get_output_shape(पूर्णांक गाँठ_idx) स्थिर override;

    भव ov::element::Type get_output_type(पूर्णांक गाँठ_idx) स्थिर override;

    भव std::vector<माप_प्रकार> get_output_stride(पूर्णांक गाँठ_idx) स्थिर override;

    भव int32_t * get_निवेश_op_params(पूर्णांक गाँठ_idx, स्थिर std::string & name) स्थिर override;

    भव int32_t * get_output_op_params(पूर्णांक गाँठ_idx) स्थिर override;

    भव माप_प्रकार get_output_op_offset(पूर्णांक गाँठ_idx) स्थिर override;

    भव std::vector<std::string> get_output_names(पूर्णांक गाँठ_idx) स्थिर override;

    भव std::string get_inplace_op_src(पूर्णांक गाँठ_idx) स्थिर override;

    भव bool is_view_like_alias_of(पूर्णांक गाँठ_idx, स्थिर std::string & view_src_name) स्थिर override;

    भव स्थिर std::string & get_op_type() स्थिर override;

    भव स्थिर std::string & get_op_type(पूर्णांक गाँठ_idx) स्थिर override;

    भव स्थिर std::string & get_op_name() स्थिर override;

    भव स्थिर std::string & get_op_name(पूर्णांक गाँठ_idx) स्थिर override;

    भव int32_t get_op_dynamic_dim(पूर्णांक गाँठ_idx) स्थिर override;

    भव व्योम visit_subgraph(
        std::function<व्योम(std::shared_ptr<GgmlDecoder>, पूर्णांक गाँठ_idx)> गाँठ_visitor) स्थिर override;

    ggml_tensor * get_निवेश_ggml_tensor(स्थिर std::string & name) स्थिर { वापस m_inputs.at(name); }

    भव पूर्णांक get_op_case(पूर्णांक गाँठ_idx) स्थिर override { वापस m_गाँठ_info_list[गाँठ_idx].गाँठ_op_case; }

    भव स्थिर std::map<std::string, ov::frontend::ggml::Model_निवेश_Info> & get_model_inputs() स्थिर override {
        वापस m_model_inputs;
    }

    भव स्थिर std::map<std::string, ov::frontend::ggml::Model_Extra_निवेश_Info> & get_model_extra_inputs() स्थिर override {
        वापस m_model_extra_inputs;
    }

    भव स्थिर std::map<std::string, std::shared_ptr<ov::गाँठ_2>> & get_model_weights() स्थिर override {
        वापस m_model_weights;
    }

    भव std::set<std::string> get_model_output_names() स्थिर override { वापस m_model_output_names; }

    स्थिर std::map<std::string, ggml_tensor *> & get_model_outputs() स्थिर { वापस m_model_outputs; }

    भव पूर्णांक get_ctx_आकार() स्थिर { वापस m_model_params.ctx; }

    भव पूर्णांक get_ctx_per_seq() स्थिर { वापस m_model_params.ctx_per_seq; }

    भव पूर्णांक get_ctx_per_seq_swa() स्थिर { वापस m_model_params.ctx_per_seq_swa; }

    भव पूर्णांक get_n_seq() स्थिर { वापस m_model_params.n_seq; }

    भव पूर्णांक is_swa_layer(पूर्णांक layer) स्थिर override {
        वापस std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    पूर्णांक get_n_heads_kv_for_layer(पूर्णांक layer) स्थिर {
        स्वतः it = m_model_params.n_heads_kv_per_layer.find(layer);
        वापस it != m_model_params.n_heads_kv_per_layer.end() ? it->second : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    पूर्णांक get_n_heads_kv_for_tensor(स्थिर ggml_tensor * kv_tensor) स्थिर {
        अगर (स्वतः layer = extract_layer_from_name(std::string(kv_tensor->name)); layer.has_मूल्य()) {
            वापस get_n_heads_kv_for_layer(layer.मूल्य());
        }
        वापस m_model_params.n_heads_kv;
    }

    पूर्णांक get_past_kv_len() स्थिर { वापस m_compute_params.past_kv_len; }

    पूर्णांक get_निवेश_len() स्थिर { वापस m_compute_params.निवेश_len; }

    भव int32_t * get_rope_params() स्थिर override { वापस const_cast<int32_t *>(m_model_params.rope_params); }

    भव bool has_mixed_rope_params() स्थिर override { वापस m_model_params.mixed_rope_params; }

    भव पूर्णांक get_ssm_state_आकार() स्थिर override { वापस m_model_params.state_आकार; }

    भव std::map<std::string, std::string> get_kv_param_res_names() स्थिर override;

    भव bool is_static() स्थिर override { वापस m_is_static; }

    भव bool is_stateful() स्थिर override { वापस m_is_stateful; }

    पूर्णांक get_static_n_tokens() स्थिर { वापस m_is_prefill ? m_prefill_chunk_आकार : 1; }

    भव bool is_splited_model() स्थिर override { वापस m_model_is_splitted; }

    ov::PartialShape get_graph_निवेश_shape(स्थिर ggml_tensor * op,
                                           स्थिर ggml_tensor * निवेश,
                                           पूर्णांक dynamic_dim_सूचकांक = -1) स्थिर;

    अटल व्योम dump_cgraph(स्थिर ggml_cgraph * cgraph, std::string & filename);

    अटल std::shared_ptr<ov::गाँठ_2> create_weight_गाँठ(ggml_tensor * tensor, bool naive = false);

    अटल std::map<std::string, std::shared_ptr<ov::गाँठ_2>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    अटल std::set<std::string> collect_weight_names(ggml_cgraph * cgraph);

    स्थिर ggml_tensor * get_tensor_used_op(स्थिर ggml_tensor * tensor) स्थिर;

    स्थिर ggml_tensor * get_tensor_from_name(स्थिर std::string & name) स्थिर;

    व्योम clear_model_weights() { m_model_weights.clear(); }

    अटल std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() स्थिर { वापस m_model_params; }

    ComputeParams get_compute_params() स्थिर { वापस m_compute_params; }

    व्योम set_model_params(स्थिर ModelParams & model_params) { m_model_params = model_params; }

    व्योम set_compute_params(स्थिर ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    पूर्णांक m_prefill_chunk_आकार = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    अटल ov::Shape get_shape(स्थिर ggml_tensor * tensor);
    अटल std::vector<माप_प्रकार> get_stride(स्थिर ggml_tensor * tensor);
    अटल ov::element::Type get_ov_type(स्थिर ggml_tensor * tensor);
    अटल std::string compute_op_type(स्थिर ggml_tensor * गाँठ);
    व्योम add_extra_inputs();

    व्योम update_io(ggml_cgraph * cgraph);
    bool is_bound_to(स्थिर ggml_cgraph * cgraph) स्थिर;

    अटल bool is_inp_tok(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    अटल bool is_inp_pos(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    अटल पूर्णांक get_inp_pos_n_planes(स्थिर ggml_tensor * op) {
        स्थिर पूर्णांक mode = op->op_params[2];
        वापस (mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION || (mode & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    अटल bool is_inp_emb(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    अटल bool is_inp_mask(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    अटल bool is_inp_mean(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_निवेश) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    अटल bool is_inp_scale_rows(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस op->op == GGML_OP_MUL && tensor == op->src[1] && म_भेद(tensor->name, "inp_scale_rows") == 0;
    }

    अटल bool is_rope_freqs_weight(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    अंतरभूत अटल bool is_recurrent_cache(स्थिर ggml_tensor * tensor) {
        वापस tensor != nullptr && (म_भेदन(tensor->name, "cache_r_l", म_माप("cache_r_l")) == 0 ||
                                     म_भेदन(tensor->name, "cache_s_l", म_माप("cache_s_l")) == 0 ||
                                     म_भेदन(tensor->name, "cache_ple_r_l", म_माप("cache_ple_r_l")) == 0);
    }

    अंतरभूत अटल bool is_cache(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    अंतरभूत अटल bool is_kvcache(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        अगर (tensor == nullptr || is_recurrent_cache(tensor)) {
            वापस false;
        }
        वापस (tensor->बफर != nullptr && tensor->बफर->usage == GGML_BACKEND_बफर_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    अंतरभूत अटल bool is_conv_state_writeback(स्थिर ggml_tensor * गाँठ) {
        वापस गाँठ->op == GGML_OP_CPY && गाँठ->view_src != nullptr && is_recurrent_cache(गाँठ->view_src) &&
               गाँठ->src[0] != nullptr && गाँठ->src[0]->op == GGML_OP_VIEW && गाँठ->src[0]->src[0] != nullptr &&
               गाँठ->src[0]->src[0]->op == GGML_OP_CONCAT && गाँठ->src[1] != nullptr &&
               गाँठ->src[1]->op == GGML_OP_VIEW && गाँठ->src[1]->view_src == गाँठ->view_src;
    }

    अटल bool is_kv_idx(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    bool is_swa_mask(स्थिर ggml_tensor * tensor) स्थिर {
        वापस m_model_params.swa_mask != nullptr && tensor == m_model_params.swa_mask;
    }

    अटल bool is_output_idx(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    अटल bool is_inp_s_copy(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) {
        वापस op->op == GGML_OP_GET_ROWS && tensor == op->src[1] &&
               op->src[0]->बफर->usage == GGML_BACKEND_बफर_USAGE_ANY;
    }

    std::string get_graph_निवेश_ov_name(स्थिर ggml_tensor * tensor, स्थिर ggml_tensor * op) स्थिर {
        अगर (is_inp_pos(tensor, op)) {
            वापस "inp_pos";
        }
        अगर (is_inp_emb(tensor, op)) {
            वापस "embd";
        }
        अगर (is_inp_mask(tensor, op)) {
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
            अगर (m_model_params.swa_layers.empty()) {
                वापस "self_kq_mask";
            }
            वापस is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        वापस tensor->name;
    }

निजी:
    व्योम set_निवेश_output();
    पूर्णांक compute_op_case(स्थिर ggml_tensor * गाँठ) स्थिर;
    bool गाँठ_is_used_as_src(स्थिर पूर्णांक गाँठ_idx);
    व्योम compute_model_inputs();
    व्योम compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_leaf(स्थिर ggml_tensor * tensor) स्थिर;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    व्योम compute_गाँठ_dynamic_dims();

    व्योम validate_cgraph() स्थिर;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_निवेश_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_निवेश_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::गाँठ_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::set<std::string> m_model_output_names;
    std::vector<गाँठ_Info> m_गाँठ_info_list;
    std::map<ggml_tensor *, पूर्णांक> m_गाँठ_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

व्योम print_tensor_address_map(स्थिर ggml_cgraph * cgraph);
