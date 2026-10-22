#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/negative.hpp>
#ضمن <openvino/op/power.hpp>
#ضمن <openvino/op/reduce_mean.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/sqrt.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_وکتور translate_rms_norm(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    خودکار op_case = context.get_op_case();

    ov::Output<ov::Node> درون_داد_گره;
    إذا (op_case == 3) {
        // Flatten sequence and token dimensions to match the gate layout.
        خودکار درون_داد_shape = context.get_درون_داد_shape(0).مقصد_shape();
        درون_داد_گره = std::make_shared<ov::op::v1::Reshape>(
            context.get_درون_داد(0),
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) درون_داد_shape[2], (int64_t) درون_داد_shape[3]}),
            خطأ);
    } وإلا إذا (op_case == 1) {
        درون_داد_گره = process_view_درون_داد_new(context, 0);
    } وإلا إذا (op_case == 2) {
        خودکار ssm_state_اندازه = context.get_ssm_state_اندازه();
        // The GDN op packs [attn | new_state] along the row axis; the state occupies the last
        // ssm_state_size * n_seqs rows. Slice it off (scaling by the active sequence count) to keep
        // just the attention output.
        ov::Output<ov::Node> state_انتها;
        إذا (context.has_درون_داد("s_copy_active_slot_len")) {
            خودکار len = context.get_درون_داد("s_copy_active_slot_len");
            خودکار state_ردیفها = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {ssm_state_اندازه}), len);
            state_انتها = std::make_shared<ov::op::v0::Negative>(state_ردیفها);
        } وإلا {
            state_انتها = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ssm_state_اندازه});
        }
        خودکار gdn_attn_output = std::make_shared<ov::op::v8::برش>(
            context.get_درون_داد(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {0}), state_انتها,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));

        خودکار درون_داد_shape = context.get_درون_داد_shape(0).مقصد_shape();
        درون_داد_گره = std::make_shared<ov::op::v1::Reshape>(
            gdn_attn_output,
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) درون_داد_shape[2], (int64_t) درون_داد_shape[3]}),
            خطأ);

    } وإلا {
        درون_داد_گره = process_view_درون_داد_new(context, 0);
    }
    خودکار مربع = std::make_shared<ov::op::v1::Multiply>(درون_داد_گره, درون_داد_گره);

    خودکار میانگین_2 = std::make_shared<ov::op::v1::Reduce_میانگین>(
        مربع, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), درست);

    عائم eps;
    کپی_حافظه(&eps, context.get_output_op_params(), حجم(عائم));

    خودکار rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::جمع_کن>(میانگین_2, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    خودکار reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    خودکار res = std::make_shared<ov::op::v1::Multiply>(درون_داد_گره, reciprocal);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
