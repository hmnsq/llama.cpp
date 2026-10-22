#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/divide.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/negative.hpp>
#அந்தர்பாவ <openvino/op/power.hpp>
#அந்தர்பாவ <openvino/op/reduce_mean.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/sqrt.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_rms_norm(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    auto op_case = context.get_op_case();

    ov::Output<ov::Node> உள்ளீடு_முனை;
    எனில் (op_case == 3) {
        // Flatten sequence and token dimensions to match the gate layout.
        auto உள்ளீடு_shape = context.get_உள்ளீடு_shape(0).to_shape();
        உள்ளீடு_முனை = std::make_shared<ov::op::v1::Reshape>(
            context.get_உள்ளீடு(0),
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) உள்ளீடு_shape[2], (int64_t) உள்ளீடு_shape[3]}),
            பொய்);
    } இல்லையெனில் எனில் (op_case == 1) {
        உள்ளீடு_முனை = process_view_உள்ளீடு_new(context, 0);
    } இல்லையெனில் எனில் (op_case == 2) {
        auto ssm_state_size = context.get_ssm_state_size();
        // The GDN op packs [attn | new_state] along the row axis; the state occupies the last
        // ssm_state_size * n_seqs rows. Slice it off (scaling by the active sequence count) to keep
        // just the attention output.
        ov::Output<ov::Node> state_end;
        எனில் (context.has_உள்ளீடு("s_copy_active_slot_len")) {
            auto len = context.get_உள்ளீடு("s_copy_active_slot_len");
            auto state_rows = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {ssm_state_size}), len);
            state_end = std::make_shared<ov::op::v0::Negative>(state_rows);
        } இல்லையெனில் {
            state_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ssm_state_size});
        }
        auto gdn_attn_output = std::make_shared<ov::op::v8::Slice>(
            context.get_உள்ளீடு(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {0}), state_end,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));

        auto உள்ளீடு_shape = context.get_உள்ளீடு_shape(0).to_shape();
        உள்ளீடு_முனை = std::make_shared<ov::op::v1::Reshape>(
            gdn_attn_output,
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) உள்ளீடு_shape[2], (int64_t) உள்ளீடு_shape[3]}),
            பொய்);

    } இல்லையெனில் {
        உள்ளீடு_முனை = process_view_உள்ளீடு_new(context, 0);
    }
    auto square = std::make_shared<ov::op::v1::Multiply>(உள்ளீடு_முனை, உள்ளீடு_முனை);

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), உண்மை);

    தசம eps;
    memcpy(&eps, context.get_output_op_params(), அளவு(தசம));

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(உள்ளீடு_முனை, reciprocal);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
