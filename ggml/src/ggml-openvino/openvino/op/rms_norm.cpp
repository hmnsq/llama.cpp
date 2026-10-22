#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/divide.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/negative.hpp>
#شامل <openvino/op/power.hpp>
#شامل <openvino/op/reduce_mean.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/sqrt.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_rms_norm(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار op_case = context.get_op_case();

    ov::Output<ov::Node> داخلہ_گرہ;
    اگر (op_case == 3) {
        // Flatten sequence and token dimensions to match the gate layout.
        خودکار داخلہ_shape = context.get_داخلہ_shape(0).مقصد_shape();
        داخلہ_گرہ = std::make_shared<ov::op::v1::Reshape>(
            context.get_داخلہ(0),
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) داخلہ_shape[2], (int64_t) داخلہ_shape[3]}),
            جھوٹ);
    } ورنہ اگر (op_case == 1) {
        داخلہ_گرہ = process_view_داخلہ_new(context, 0);
    } ورنہ اگر (op_case == 2) {
        خودکار ssm_state_سائز = context.get_ssm_state_سائز();
        // The GDN op packs [attn | new_state] along the row axis; the state occupies the last
        // ssm_state_size * n_seqs rows. Slice it off (scaling by the active sequence count) to keep
        // just the attention output.
        ov::Output<ov::Node> state_انتہا;
        اگر (context.has_داخلہ("s_copy_active_slot_len")) {
            خودکار len = context.get_داخلہ("s_copy_active_slot_len");
            خودکار state_صفیں = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {ssm_state_سائز}), len);
            state_انتہا = std::make_shared<ov::op::v0::Negative>(state_صفیں);
        } ورنہ {
            state_انتہا = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ssm_state_سائز});
        }
        خودکار gdn_attn_output = std::make_shared<ov::op::v8::قاش>(
            context.get_داخلہ(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {0}), state_انتہا,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));

        خودکار داخلہ_shape = context.get_داخلہ_shape(0).مقصد_shape();
        داخلہ_گرہ = std::make_shared<ov::op::v1::Reshape>(
            gdn_attn_output,
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) داخلہ_shape[2], (int64_t) داخلہ_shape[3]}),
            جھوٹ);

    } ورنہ {
        داخلہ_گرہ = process_view_داخلہ_new(context, 0);
    }
    خودکار مربع = std::make_shared<ov::op::v1::Multiply>(داخلہ_گرہ, داخلہ_گرہ);

    خودکار اوسط_2 = std::make_shared<ov::op::v1::Reduce_اوسط>(
        مربع, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), سچ);

    اعشاری eps;
    حافظہ_نقل(&eps, context.get_output_op_params(), حجم(اعشاری));

    خودکار rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::جوڑو>(اوسط_2, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    خودکار reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    خودکار res = std::make_shared<ov::op::v1::Multiply>(داخلہ_گرہ, reciprocal);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
