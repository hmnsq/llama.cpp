#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/divide.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/negative.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/power.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reduce_mean.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/sqrt.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_rms_norm(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਸਵੈਚਾਲਿਤ op_case = context.get_op_case();

    ov::Output<ov::Node> ਨਿਵੇਸ਼_ਗੰਢ;
    ਜੇ (op_case == 3) {
        // Flatten sequence and token dimensions to match the gate layout.
        ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();
        ਨਿਵੇਸ਼_ਗੰਢ = std::make_shared<ov::op::v1::Reshape>(
            context.get_ਨਿਵੇਸ਼(0),
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) ਨਿਵੇਸ਼_shape[2], (int64_t) ਨਿਵੇਸ਼_shape[3]}),
            ਝੂਠ);
    } ਨਹੀਂਤਾਂ ਜੇ (op_case == 1) {
        ਨਿਵੇਸ਼_ਗੰਢ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    } ਨਹੀਂਤਾਂ ਜੇ (op_case == 2) {
        ਸਵੈਚਾਲਿਤ ssm_state_ਮਾਪ = context.get_ssm_state_ਮਾਪ();
        // The GDN op packs [attn | new_state] along the row axis; the state occupies the last
        // ssm_state_size * n_seqs rows. Slice it off (scaling by the active sequence count) to keep
        // just the attention output.
        ov::Output<ov::Node> state_ਸਿਰਾ;
        ਜੇ (context.has_ਨਿਵੇਸ਼("s_copy_active_slot_len")) {
            ਸਵੈਚਾਲਿਤ len = context.get_ਨਿਵੇਸ਼("s_copy_active_slot_len");
            ਸਵੈਚਾਲਿਤ state_ਪੰਕਤੀਆਂ = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {ssm_state_ਮਾਪ}), len);
            state_ਸਿਰਾ = std::make_shared<ov::op::v0::Negative>(state_ਪੰਕਤੀਆਂ);
        } ਨਹੀਂਤਾਂ {
            state_ਸਿਰਾ = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ssm_state_ਮਾਪ});
        }
        ਸਵੈਚਾਲਿਤ gdn_attn_output = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(
            context.get_ਨਿਵੇਸ਼(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {0}), state_ਸਿਰਾ,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));

        ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();
        ਨਿਵੇਸ਼_ਗੰਢ = std::make_shared<ov::op::v1::Reshape>(
            gdn_attn_output,
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) ਨਿਵੇਸ਼_shape[2], (int64_t) ਨਿਵੇਸ਼_shape[3]}),
            ਝੂਠ);

    } ਨਹੀਂਤਾਂ {
        ਨਿਵੇਸ਼_ਗੰਢ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    }
    ਸਵੈਚਾਲਿਤ ਵਰਗਫਲ = std::make_shared<ov::op::v1::Multiply>(ਨਿਵੇਸ਼_ਗੰਢ, ਨਿਵੇਸ਼_ਗੰਢ);

    ਸਵੈਚਾਲਿਤ ਔਸਤ_2 = std::make_shared<ov::op::v1::Reduce_ਔਸਤ>(
        ਵਰਗਫਲ, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), ਸੱਚ);

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, context.get_output_op_params(), ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਸਵੈਚਾਲਿਤ rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::ਜੋੜੋ>(ਔਸਤ_2, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    ਸਵੈਚਾਲਿਤ reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Multiply>(ਨਿਵੇਸ਼_ਗੰਢ, reciprocal);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
