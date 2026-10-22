#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/op/add.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/divide.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/multiply.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/negative.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/power.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reduce_mean.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/slice.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/sqrt.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_rms_norm(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    auto op_case = context.get_op_case();

    ov::Output<ov::Node> ନିବେଶ_ଗଣ୍ଠି;
    ଯଦି (op_case == 3) {
        // Flatten sequence and token dimensions to match the gate layout.
        auto ନିବେଶ_shape = context.get_ନିବେଶ_shape(0).to_shape();
        ନିବେଶ_ଗଣ୍ଠି = std::make_shared<ov::op::v1::Reshape>(
            context.get_ନିବେଶ(0),
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) ନିବେଶ_shape[2], (int64_t) ନିବେଶ_shape[3]}),
            ମିଥ୍ୟା);
    } ଅନ୍ୟଥା ଯଦି (op_case == 1) {
        ନିବେଶ_ଗଣ୍ଠି = process_view_ନିବେଶ_new(context, 0);
    } ଅନ୍ୟଥା ଯଦି (op_case == 2) {
        auto ssm_state_size = context.get_ssm_state_size();
        // The GDN op packs [attn | new_state] along the row axis; the state occupies the last
        // ssm_state_size * n_seqs rows. Slice it off (scaling by the active sequence count) to keep
        // just the attention output.
        ov::Output<ov::Node> state_end;
        ଯଦି (context.has_ନିବେଶ("s_copy_active_slot_len")) {
            auto len = context.get_ନିବେଶ("s_copy_active_slot_len");
            auto state_rows = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {ssm_state_size}), len);
            state_end = std::make_shared<ov::op::v0::Negative>(state_rows);
        } ଅନ୍ୟଥା {
            state_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ssm_state_size});
        }
        auto gdn_attn_output = std::make_shared<ov::op::v8::Slice>(
            context.get_ନିବେଶ(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {0}), state_end,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));

        auto ନିବେଶ_shape = context.get_ନିବେଶ_shape(0).to_shape();
        ନିବେଶ_ଗଣ୍ଠି = std::make_shared<ov::op::v1::Reshape>(
            gdn_attn_output,
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) ନିବେଶ_shape[2], (int64_t) ନିବେଶ_shape[3]}),
            ମିଥ୍ୟା);

    } ଅନ୍ୟଥା {
        ନିବେଶ_ଗଣ୍ଠି = process_view_ନିବେଶ_new(context, 0);
    }
    auto square = std::make_shared<ov::op::v1::Multiply>(ନିବେଶ_ଗଣ୍ଠି, ନିବେଶ_ଗଣ୍ଠି);

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), ସତ୍ୟ);

    ଦଶମିକ eps;
    memcpy(&eps, context.get_output_op_params(), ଆକାର(ଦଶମିକ));

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(ନିବେଶ_ଗଣ୍ଠି, reciprocal);

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
