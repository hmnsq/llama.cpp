#ضمن "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// Moves the sequence axis of the stateful KV cache from dim 1 to dim 2, i.e. from
// [1, seq, n_heads_kv, head_size] to [1, n_heads_kv, seq, head_size], and updates the
// Concat that appends to it. Two wins: the GPU plugin only appends new tokens in place
// when the growing axis is a spatial axis, and the reader no longer has to transpose the
// whole accumulated state every token (that cost grows with context length, so it is the
// larger win at depth for a model with several KV heads). Only rewrites states that still
// match the frontend layout, so it no-ops if that layout ever changes.
class KVStateSeqAxis : عام ov::pass::ModelPass {
عام:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::ggml::pass::KVStateSeqAxis")
    bool پیاپی_on_model(ثابت std::shared_ptr<ov::Model> & model) override;
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
