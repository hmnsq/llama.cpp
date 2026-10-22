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

OutputVector translate_rms_norm(ثابت عقدة_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto op_case = context.get_op_case();

    ov::Output<ov::Node> مدخل_عقدة;
    إذا (op_case == 3) {
        // Flatten sequence and token dimensions to match the gate layout.
        auto مدخل_shape = context.get_مدخل_shape(0).to_shape();
        مدخل_عقدة = std::make_shared<ov::op::v1::Reshape>(
            context.get_مدخل(0),
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) مدخل_shape[2], (int64_t) مدخل_shape[3]}),
            خطأ);
    } وإلا إذا (op_case == 1) {
        مدخل_عقدة = process_view_مدخل_new(context, 0);
    } وإلا إذا (op_case == 2) {
        auto ssm_state_size = context.get_ssm_state_size();
        // The GDN op packs [attn | new_state] along the row axis; the state occupies the last
        // ssm_state_size * n_seqs rows. Slice it off (scaling by the active sequence count) to keep
        // just the attention output.
        ov::Output<ov::Node> state_end;
        إذا (context.has_مدخل("s_copy_active_slot_len")) {
            auto len = context.get_مدخل("s_copy_active_slot_len");
            auto state_rows = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {ssm_state_size}), len);
            state_end = std::make_shared<ov::op::v0::Negative>(state_rows);
        } وإلا {
            state_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ssm_state_size});
        }
        auto gdn_attn_output = std::make_shared<ov::op::v8::Slice>(
            context.get_مدخل(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {0}), state_end,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));

        auto مدخل_shape = context.get_مدخل_shape(0).to_shape();
        مدخل_عقدة = std::make_shared<ov::op::v1::Reshape>(
            gdn_attn_output,
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) مدخل_shape[2], (int64_t) مدخل_shape[3]}),
            خطأ);

    } وإلا {
        مدخل_عقدة = process_view_مدخل_new(context, 0);
    }
    auto square = std::make_shared<ov::op::v1::Multiply>(مدخل_عقدة, مدخل_عقدة);

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    عائم eps;
    memcpy(&eps, context.get_output_op_params(), حجم(عائم));

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(مدخل_عقدة, reciprocal);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
