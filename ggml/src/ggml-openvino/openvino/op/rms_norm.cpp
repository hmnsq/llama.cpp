#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/divide.hpp>
#અન્તર્ભાવ <openvino/op/multiply.hpp>
#અન્તર્ભાવ <openvino/op/negative.hpp>
#અન્તર્ભાવ <openvino/op/power.hpp>
#અન્તર્ભાવ <openvino/op/reduce_mean.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/sqrt.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_rms_norm(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    auto op_case = context.get_op_case();

    ov::Output<ov::Node> આગત_ગાંઠ;
    જો (op_case == 3) {
        // Flatten sequence and token dimensions to match the gate layout.
        auto આગત_shape = context.get_આગત_shape(0).to_shape();
        આગત_ગાંઠ = std::make_shared<ov::op::v1::Reshape>(
            context.get_આગત(0),
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) આગત_shape[2], (int64_t) આગત_shape[3]}),
            ખોટું);
    } નહીંતર જો (op_case == 1) {
        આગત_ગાંઠ = process_view_આગત_new(context, 0);
    } નહીંતર જો (op_case == 2) {
        auto ssm_state_size = context.get_ssm_state_size();
        // The GDN op packs [attn | new_state] along the row axis; the state occupies the last
        // ssm_state_size * n_seqs rows. Slice it off (scaling by the active sequence count) to keep
        // just the attention output.
        ov::Output<ov::Node> state_end;
        જો (context.has_આગત("s_copy_active_slot_len")) {
            auto len = context.get_આગત("s_copy_active_slot_len");
            auto state_rows = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {ssm_state_size}), len);
            state_end = std::make_shared<ov::op::v0::Negative>(state_rows);
        } નહીંતર {
            state_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ssm_state_size});
        }
        auto gdn_attn_output = std::make_shared<ov::op::v8::Slice>(
            context.get_આગત(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {0}), state_end,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));

        auto આગત_shape = context.get_આગત_shape(0).to_shape();
        આગત_ગાંઠ = std::make_shared<ov::op::v1::Reshape>(
            gdn_attn_output,
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) આગત_shape[2], (int64_t) આગત_shape[3]}),
            ખોટું);

    } નહીંતર {
        આગત_ગાંઠ = process_view_આગત_new(context, 0);
    }
    auto square = std::make_shared<ov::op::v1::Multiply>(આગત_ગાંઠ, આગત_ગાંઠ);

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), સાચું);

    દશાંશ eps;
    memcpy(&eps, context.get_output_op_params(), કદ(દશાંશ));

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(આગત_ગાંઠ, reciprocal);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
