#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <memory>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/reduce_sum.hpp>
#समावेश <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_add(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    अगर (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        माप_प्रकार view_आकार = context.get_view_निवेश_आकार(1);
        स्वतः base_name = context.get_view_निवेश_src_name(1, view_आकार - 1);
        स्वतः base = context.get_निवेश(base_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        स्थिर स्वतः base_rank = base.get_partial_shape().rank();
        FRONT_END_OP_CONVERSION_CHECK(base_rank.is_static() && base_rank.get_लंबाई() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        स्थिर int64_t rank = base_rank.get_लंबाई();

        स्वतः reduced = std::make_shared<ov::op::v1::Reduce_योग>(
            base, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), false);
        स्वतः res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        वापस rename_outputs_with_suffix({res}, context.get_name());
    }

    स्वतः निवेश_0 = process_view_निवेश_new(context, 0);
    स्वतः निवेश_1 = process_view_निवेश_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    स्वतः output_type = context.get_output_type();
    अगर (निवेश_0.get_element_type() != निवेश_1.get_element_type()) {
        अगर (निवेश_0.get_element_type() != ov::element::f32) {
            निवेश_0 = std::make_shared<ov::op::v0::Convert>(निवेश_0, ov::element::f32);
        }
        अगर (निवेश_1.get_element_type() != ov::element::f32) {
            निवेश_1 = std::make_shared<ov::op::v0::Convert>(निवेश_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(निवेश_0, निवेश_1);
    अगर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
