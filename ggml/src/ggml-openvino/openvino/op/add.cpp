#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/reduce_sum.hpp>
#شامل <openvino/op/unsqueeze.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_جوڑو(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);

    اگر (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        size_t view_سائز = context.get_view_داخلہ_سائز(1);
        خودکار بنیاد_name = context.get_view_داخلہ_src_name(1, view_سائز - 1);
        خودکار بنیاد = context.get_داخلہ(بنیاد_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        ثابت خودکار بنیاد_rank = بنیاد.get_partial_shape().rank();
        FRONT_انتہا_OP_CONVERSION_CHECK(بنیاد_rank.is_static() && بنیاد_rank.get_لمبائی() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        ثابت int64_t rank = بنیاد_rank.get_لمبائی();

        خودکار reduced = std::make_shared<ov::op::v1::Reduce_مجموع>(
            بنیاد, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), جھوٹ);
        خودکار res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        واپس rename_outputs_with_suffix({res}, context.get_name());
    }

    خودکار داخلہ_0 = process_view_داخلہ_new(context, 0);
    خودکار داخلہ_1 = process_view_داخلہ_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    خودکار output_type = context.get_output_type();
    اگر (داخلہ_0.get_element_type() != داخلہ_1.get_element_type()) {
        اگر (داخلہ_0.get_element_type() != ov::element::f32) {
            داخلہ_0 = std::make_shared<ov::op::v0::Convert>(داخلہ_0, ov::element::f32);
        }
        اگر (داخلہ_1.get_element_type() != ov::element::f32) {
            داخلہ_1 = std::make_shared<ov::op::v0::Convert>(داخلہ_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::جوڑو>(داخلہ_0, داخلہ_1);
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
