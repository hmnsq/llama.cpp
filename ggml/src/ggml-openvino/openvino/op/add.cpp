#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reduce_sum.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_ಕೂಡಿಸು(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ಆದರೆ (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        size_t view_ಗಾತ್ರ = context.get_view_ಒಳಹರಿವು_ಗಾತ್ರ(1);
        ಸ್ವಯಂಚಾಲಿತ ತಳ_name = context.get_view_ಒಳಹರಿವು_src_name(1, view_ಗಾತ್ರ - 1);
        ಸ್ವಯಂಚಾಲಿತ ತಳ = context.get_ಒಳಹರಿವು(ತಳ_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ತಳ_rank = ತಳ.get_partial_shape().rank();
        FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ತಳ_rank.is_static() && ತಳ_rank.get_ಉದ್ದಳತೆ() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        ಸ್ಥಿರಾಂಕ int64_t rank = ತಳ_rank.get_ಉದ್ದಳತೆ();

        ಸ್ವಯಂಚಾಲಿತ reduced = std::make_shared<ov::op::v1::Reduce_ಮೊತ್ತ>(
            ತಳ, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), ಸುಳ್ಳು);
        ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
    }

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_0 = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_1 = process_view_ಒಳಹರಿವು_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (ಒಳಹರಿವು_0.get_element_type() != ಒಳಹರಿವು_1.get_element_type()) {
        ಆದರೆ (ಒಳಹರಿವು_0.get_element_type() != ov::element::f32) {
            ಒಳಹರಿವು_0 = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು_0, ov::element::f32);
        }
        ಆದರೆ (ಒಳಹರಿವು_1.get_element_type() != ov::element::f32) {
            ಒಳಹರಿವು_1 = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(ಒಳಹರಿವು_0, ಒಳಹರಿವು_1);
    ಆದರೆ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
