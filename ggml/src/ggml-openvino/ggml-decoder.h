#رهنمود once

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml.h"
#ضمن "openvino/decoder.h"

#ضمن <cstdint>
#ضمن <cstring>
#ضمن <map>
#ضمن <memory>
#ضمن <openvino/core/partial_shape.hpp>
#ضمن <optional>
#ضمن <set>
#ضمن <string>
#ضمن <vector>

بنية ModelParams {
    صحيح ctx = -1;
    صحيح ctx_در_هر_seq = -1;
    صحيح ctx_در_هر_seq_swa = -1;
    صحيح n_seq = 1;
    صحيح n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<صحيح, صحيح> n_heads_kv_در_هر_layer;
    صحيح سر_اندازه = -1;
    صحيح state_اندازه = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    صحيح n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::وکتور<صحيح> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    ثابت ggml_tensor * swa_ماسک = nullptr;

    std::وکتور<std::string> kv_نامها;
    size_t kv_میانگیر_ctx_id = 0;

    bool same_rope_params(ثابت ModelParams & other) ثابت {
        أرجع mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, حجم(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(ثابت ModelParams & other) ثابت {
        أرجع same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(ثابت ModelParams & other) ثابت {
        أرجع same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_میانگیر_تغییریافته(ثابت ModelParams & other) ثابت { أرجع kv_میانگیر_ctx_id != other.kv_میانگیر_ctx_id; }
};

بنية ComputeParams {
    صحيح n_seq_فعال = 1;
    صحيح seq_فعال_شروع = 0;
    صحيح attention_اندازه = -1;
    صحيح attention_اندازه_swa = -1;
    صحيح attention_اندازه_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    صحيح swa_پنجره = -1;
    صحيح درون_داد_len = -1;
    صحيح token_len_در_هر_seq = -1;
    صحيح past_kv_len = -1;
    صحيح output_len = 1;

    صحيح cache_rs_reset_idx = -1;
    صحيح cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    صحيح s_copy_فعال_جایگاه_len = -1;
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

    بنية RsWriteback {
        صحيح جایگاه_begin = 0;  // first cache slot written by the CPY
        صحيح src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<صحيح> استخراج_layer_from_name(ثابت std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_جمع_جمع_کن(ثابت ggml_tensor * گره);

class GgmlOvDecoder : عام ov::frontend::ggml::GgmlDecoder {
عام:
    ساكن std::string get_tensor_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor);
    بنية گره_Info {
        ggml_tensor * گره;
        std::string گره_name;
        std::string گره_op_type;
        std::map<std::string, ggml_tensor *> گره_inputs;
        std::map<std::string, std::وکتور<std::pair<std::string, ggml_tensor *>>> گره_inputs_views;
        std::وکتور<std::string> گره_inputs_نامها;
        صحيح گره_op_case = 0;
        فراغ * داده_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::گره_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  صحيح prefill_chunk_اندازه = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::گره_2>> & model_weights);

    مجازی ov::Any get_attribute(ثابت std::string & name) ثابت override {
        أرجع nullptr;
        GGML_استفادهنشده(name);
    }

    مجازی ov::PartialShape get_درون_داد_shape(صحيح گره_idx, ثابت std::string & name) ثابت override;

    مجازی std::وکتور<size_t> get_درون_داد_stride(صحيح گره_idx, ثابت std::string & name) ثابت override;

    مجازی size_t get_view_درون_داد_اندازه(صحيح گره_idx, ثابت std::string & name) ثابت override;

    مجازی size_t get_view_درون_داد_جابه_جایی(صحيح گره_idx, ثابت std::string & name, size_t view_شاخص) ثابت override;

    مجازی size_t get_view_درون_داد_src_جابه_جایی(صحيح گره_idx, ثابت std::string & name, size_t view_شاخص) ثابت override;

    مجازی std::وکتور<size_t> get_view_درون_داد_stride(صحيح گره_idx,
                                                      ثابت std::string & name,
                                                      size_t view_شاخص) ثابت override;

    مجازی std::وکتور<size_t> get_view_درون_داد_src_stride(صحيح گره_idx,
                                                          ثابت std::string & name,
                                                          size_t view_شاخص) ثابت override;

    مجازی ov::Shape get_view_درون_داد_ggml_shape(صحيح گره_idx,
                                                ثابت std::string & name,
                                                size_t view_شاخص) ثابت override;

    مجازی ov::Shape get_view_درون_داد_src_ggml_shape(صحيح گره_idx,
                                                    ثابت std::string & name,
                                                    size_t view_شاخص) ثابت override;

    مجازی ov::PartialShape get_view_درون_داد_ov_shape(صحيح گره_idx,
                                                     ثابت std::string & name,
                                                     size_t view_شاخص) ثابت override;

    مجازی ov::PartialShape get_view_درون_داد_src_ov_shape(صحيح گره_idx,
                                                         ثابت std::string & name,
                                                         size_t view_شاخص) ثابت override;

    مجازی std::string get_view_درون_داد_name(صحيح گره_idx, ثابت std::string & name, size_t view_شاخص) ثابت override;

    مجازی std::string get_view_درون_داد_src_name(صحيح گره_idx,
                                                ثابت std::string & name,
                                                size_t view_شاخص) ثابت override;

    مجازی ov::element::Type get_درون_داد_type(صحيح گره_idx, ثابت std::string & name) ثابت override;

    مجازی size_t get_درون_داد_اندازه() ثابت override;

    مجازی size_t get_درون_داد_اندازه(صحيح گره_idx) ثابت override;

    مجازی فراغ get_درون_داد_گره(size_t درون_داد_پورت_idx,
                                std::string & تولیدکننده_name,
                                std::string & تولیدکننده_output_پورت_name,
                                size_t & تولیدکننده_output_پورت_شاخص) ثابت override {
        GGML_استفادهنشده(درون_داد_پورت_idx);
        GGML_استفادهنشده(تولیدکننده_name);
        GGML_استفادهنشده(تولیدکننده_output_پورت_name);
        GGML_استفادهنشده(تولیدکننده_output_پورت_شاخص);
    }

    مجازی std::وکتور<std::string> get_درون_داد_نامها(صحيح گره_idx) ثابت override;

    مجازی ov::PartialShape get_output_shape(صحيح گره_idx) ثابت override;

    مجازی ov::element::Type get_output_type(صحيح گره_idx) ثابت override;

    مجازی std::وکتور<size_t> get_output_stride(صحيح گره_idx) ثابت override;

    مجازی int32_t * get_درون_داد_op_params(صحيح گره_idx, ثابت std::string & name) ثابت override;

    مجازی int32_t * get_output_op_params(صحيح گره_idx) ثابت override;

    مجازی size_t get_output_op_جابه_جایی(صحيح گره_idx) ثابت override;

    مجازی std::وکتور<std::string> get_output_نامها(صحيح گره_idx) ثابت override;

    مجازی std::string get_inplace_op_src(صحيح گره_idx) ثابت override;

    مجازی bool is_view_like_alias_از(صحيح گره_idx, ثابت std::string & view_src_name) ثابت override;

    مجازی ثابت std::string & get_op_type() ثابت override;

    مجازی ثابت std::string & get_op_type(صحيح گره_idx) ثابت override;

    مجازی ثابت std::string & get_op_name() ثابت override;

    مجازی ثابت std::string & get_op_name(صحيح گره_idx) ثابت override;

    مجازی int32_t get_op_dynamic_dim(صحيح گره_idx) ثابت override;

    مجازی فراغ بازدید_subgraph(
        std::function<فراغ(std::shared_ptr<GgmlDecoder>, صحيح گره_idx)> گره_visitor) ثابت override;

    ggml_tensor * get_درون_داد_ggml_tensor(ثابت std::string & name) ثابت { أرجع m_inputs.at(name); }

    مجازی صحيح get_op_case(صحيح گره_idx) ثابت override { أرجع m_گره_info_list[گره_idx].گره_op_case; }

    مجازی ثابت std::map<std::string, ov::frontend::ggml::Model_درون_داد_Info> & get_model_inputs() ثابت override {
        أرجع m_model_inputs;
    }

    مجازی ثابت std::map<std::string, ov::frontend::ggml::Model_Extra_درون_داد_Info> & get_model_extra_inputs() ثابت override {
        أرجع m_model_extra_inputs;
    }

    مجازی ثابت std::map<std::string, std::shared_ptr<ov::گره_2>> & get_model_weights() ثابت override {
        أرجع m_model_weights;
    }

    مجازی std::مجموعه<std::string> get_model_output_نامها() ثابت override { أرجع m_model_output_نامها; }

    ثابت std::map<std::string, ggml_tensor *> & get_model_outputs() ثابت { أرجع m_model_outputs; }

    مجازی صحيح get_ctx_اندازه() ثابت { أرجع m_model_params.ctx; }

    مجازی صحيح get_ctx_در_هر_seq() ثابت { أرجع m_model_params.ctx_در_هر_seq; }

    مجازی صحيح get_ctx_در_هر_seq_swa() ثابت { أرجع m_model_params.ctx_در_هر_seq_swa; }

    مجازی صحيح get_n_seq() ثابت { أرجع m_model_params.n_seq; }

    مجازی صحيح is_swa_layer(صحيح layer) ثابت override {
        أرجع std::پیدا_کن(m_model_params.swa_layers.begin(), m_model_params.swa_layers.انتها(), layer) !=
               m_model_params.swa_layers.انتها();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    صحيح get_n_heads_kv_for_layer(صحيح layer) ثابت {
        خودکار it = m_model_params.n_heads_kv_در_هر_layer.پیدا_کن(layer);
        أرجع it != m_model_params.n_heads_kv_در_هر_layer.انتها() ? it->دومی : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    صحيح get_n_heads_kv_for_tensor(ثابت ggml_tensor * kv_tensor) ثابت {
        إذا (خودکار layer = استخراج_layer_from_name(std::string(kv_tensor->name)); layer.has_مقدار()) {
            أرجع get_n_heads_kv_for_layer(layer.مقدار());
        }
        أرجع m_model_params.n_heads_kv;
    }

    صحيح get_past_kv_len() ثابت { أرجع m_compute_params.past_kv_len; }

    صحيح get_درون_داد_len() ثابت { أرجع m_compute_params.درون_داد_len; }

    مجازی int32_t * get_rope_params() ثابت override { أرجع const_cast<int32_t *>(m_model_params.rope_params); }

    مجازی bool has_mixed_rope_params() ثابت override { أرجع m_model_params.mixed_rope_params; }

    مجازی صحيح get_ssm_state_اندازه() ثابت override { أرجع m_model_params.state_اندازه; }

    مجازی std::map<std::string, std::string> get_kv_param_res_نامها() ثابت override;

    مجازی bool is_static() ثابت override { أرجع m_is_static; }

    مجازی bool is_stateful() ثابت override { أرجع m_is_stateful; }

    صحيح get_static_n_نشانهها() ثابت { أرجع m_is_prefill ? m_prefill_chunk_اندازه : 1; }

    مجازی bool is_splited_model() ثابت override { أرجع m_model_is_splitted; }

    ov::PartialShape get_گراف_درون_داد_shape(ثابت ggml_tensor * op,
                                           ثابت ggml_tensor * درون_داد,
                                           صحيح dynamic_dim_شاخص = -1) ثابت;

    ساكن فراغ dump_cgraph(ثابت ggml_cgraph * cgraph, std::string & filename);

    ساكن std::shared_ptr<ov::گره_2> create_وزن_گره(ggml_tensor * tensor, bool naive = false);

    ساكن std::map<std::string, std::shared_ptr<ov::گره_2>> create_وزن_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    ساكن std::مجموعه<std::string> collect_وزن_نامها(ggml_cgraph * cgraph);

    ثابت ggml_tensor * get_tensor_استفاده_شده_op(ثابت ggml_tensor * tensor) ثابت;

    ثابت ggml_tensor * get_tensor_from_name(ثابت std::string & name) ثابت;

    فراغ clear_model_weights() { m_model_weights.clear(); }

    ساكن std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() ثابت { أرجع m_model_params; }

    ComputeParams get_compute_params() ثابت { أرجع m_compute_params; }

    فراغ مجموعه_model_params(ثابت ModelParams & model_params) { m_model_params = model_params; }

    فراغ مجموعه_compute_params(ثابت ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    صحيح m_prefill_chunk_اندازه = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    ساكن ov::Shape get_shape(ثابت ggml_tensor * tensor);
    ساكن std::وکتور<size_t> get_stride(ثابت ggml_tensor * tensor);
    ساكن ov::element::Type get_ov_type(ثابت ggml_tensor * tensor);
    ساكن std::string compute_op_type(ثابت ggml_tensor * گره);
    فراغ جمع_کن_extra_inputs();

    فراغ update_io(ggml_cgraph * cgraph);
    bool is_bound_مقصد(ثابت ggml_cgraph * cgraph) ثابت;

    ساكن bool is_inp_tok(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_GET_ردیفها && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    ساكن bool is_inp_pos(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    ساكن صحيح get_inp_pos_n_planes(ثابت ggml_tensor * op) {
        ثابت صحيح نما = op->op_params[2];
        أرجع (نما == GGML_ROPE_TYPE_IMROPE || نما == GGML_ROPE_TYPE_VISION || (نما & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    ساكن bool is_inp_emb(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع tensor->op == GGML_OP_GET_ردیفها && op->op == GGML_OP_RMS_NORM;
    }

    ساكن bool is_inp_ماسک(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    ساكن bool is_inp_میانگین(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_درون_داد) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    ساكن bool is_inp_scale_ردیفها(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    ساكن bool is_rope_freqs_وزن(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    مضمن ساكن bool is_recurrent_cache(ثابت ggml_tensor * tensor) {
        أرجع tensor != nullptr && (مقایسه_رشته(tensor->name, "cache_r_l", طول_رشته("cache_r_l")) == 0 ||
                                     مقایسه_رشته(tensor->name, "cache_s_l", طول_رشته("cache_s_l")) == 0 ||
                                     مقایسه_رشته(tensor->name, "cache_ple_r_l", طول_رشته("cache_ple_r_l")) == 0);
    }

    مضمن ساكن bool is_cache(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    مضمن ساكن bool is_kvcache(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        إذا (tensor == nullptr || is_recurrent_cache(tensor)) {
            أرجع false;
        }
        أرجع (tensor->میانگیر != nullptr && tensor->میانگیر->usage == GGML_BACKEND_میانگیر_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_مجموعه_ردیفها && op->src[2] == tensor);
    }

    مضمن ساكن bool is_conv_state_writeback(ثابت ggml_tensor * گره) {
        أرجع گره->op == GGML_OP_CPY && گره->view_src != nullptr && is_recurrent_cache(گره->view_src) &&
               گره->src[0] != nullptr && گره->src[0]->op == GGML_OP_VIEW && گره->src[0]->src[0] != nullptr &&
               گره->src[0]->src[0]->op == GGML_OP_CONCAT && گره->src[1] != nullptr &&
               گره->src[1]->op == GGML_OP_VIEW && گره->src[1]->view_src == گره->view_src;
    }

    ساكن bool is_kv_idx(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_مجموعه_ردیفها && op->src[1] == tensor;
    }

    bool is_swa_ماسک(ثابت ggml_tensor * tensor) ثابت {
        أرجع m_model_params.swa_ماسک != nullptr && tensor == m_model_params.swa_ماسک;
    }

    ساكن bool is_output_idx(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_GET_ردیفها && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    ساكن bool is_inp_s_copy(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_GET_ردیفها && tensor == op->src[1] &&
               op->src[0]->میانگیر->usage == GGML_BACKEND_میانگیر_USAGE_ANY;
    }

    std::string get_گراف_درون_داد_ov_name(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) ثابت {
        إذا (is_inp_pos(tensor, op)) {
            أرجع "inp_pos";
        }
        إذا (is_inp_emb(tensor, op)) {
            أرجع "embd";
        }
        إذا (is_inp_ماسک(tensor, op)) {
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
            إذا (m_model_params.swa_layers.خالی()) {
                أرجع "self_kq_mask";
            }
            أرجع is_swa_ماسک(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        أرجع tensor->name;
    }

خاص:
    فراغ مجموعه_درون_داد_output();
    صحيح compute_op_case(ثابت ggml_tensor * گره) ثابت;
    bool گره_is_استفاده_شده_as_src(ثابت صحيح گره_idx);
    فراغ compute_model_inputs();
    فراغ compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_برگ(ثابت ggml_tensor * tensor) ثابت;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    فراغ compute_گره_dynamic_dims();

    فراغ validate_cgraph() ثابت;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_درون_داد_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_درون_داد_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::گره_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::مجموعه<std::string> m_model_output_نامها;
    std::وکتور<گره_Info> m_گره_info_list;
    std::map<ggml_tensor *, صحيح> m_گره_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

فراغ print_tensor_آدرس_map(ثابت ggml_cgraph * cgraph);
