#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <array>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/pad.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <utility>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

ਨਾਮਾਕਾਸ਼ {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> ਨਿਵੇਸ਼,
                                            ਅਚਲ std::array<int32_t, 8> & pads,
                                            ਅਚਲ ov::Shape & ਨਿਵੇਸ਼_shape) {
    ov::Output<ov::Node> ਨਤੀਜਾ = std::move(ਨਿਵੇਸ਼);

    ਅਚਲ std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ਅਚਲ std::array<int32_t, 4> pads_ਸਿਰਾ = {pads[7], pads[5], pads[3], pads[1]};

    ਲਈ (size_t axis = 0; axis < ਨਿਵੇਸ਼_shape.size(); ++axis) {
        ਅਚਲ int64_t ਨਿਵੇਸ਼_dim = static_cast<int64_t>(ਨਿਵੇਸ਼_shape[axis]);
        ਅਚਲ int64_t pad_begin = pads_begin[axis];
        ਅਚਲ int64_t pad_ਸਿਰਾ = pads_ਸਿਰਾ[axis];

        ਜੇ (pad_begin == 0 && pad_ਸਿਰਾ == 0) {
            ਜਾਰੀ;
        }

        FRONT_ਸਿਰਾ_CHECK_IMPLEMENTED(ਨਿਵੇਸ਼_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(ਨਿਵੇਸ਼_dim + pad_begin + pad_ਸਿਰਾ));
        ਲਈ (int64_t ਸੂਚਕਾਂਕ = 0; ਸੂਚਕਾਂਕ < static_cast<int64_t>(indices.size()); ++ਸੂਚਕਾਂਕ) {
            int64_t wrapped = (ਸੂਚਕਾਂਕ - pad_begin) % ਨਿਵੇਸ਼_dim;
            ਜੇ (wrapped < 0) {
                wrapped += ਨਿਵੇਸ਼_dim;
            }
            indices[static_cast<size_t>(ਸੂਚਕਾਂਕ)] = wrapped;
        }

        ਸਵੈਚਾਲਿਤ gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        ਸਵੈਚਾਲਿਤ gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        ਨਤੀਜਾ = std::make_shared<ov::op::v8::Gather>(ਨਤੀਜਾ, gather_indices, gather_axis);
    }

    ਵਾਪਸ ਨਤੀਜਾ;
}

}  // namespace

Output_ਸਦਿਸ਼ translate_pad(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਜੇ (context.get_ਨਿਵੇਸ਼_shape(0) == context.get_output_shape()) {
        ਵਾਪਸ rename_outputs_with_suffix({ਨਿਵੇਸ਼}, context.get_name());
    }

    ਅਚਲ int32_t * op_params = context.get_output_op_params();
    FRONT_ਸਿਰਾ_CHECK_IMPLEMENTED(op_params != ਖਾਲੀ_ਸੂਚਕ, "PAD requires output op params");

    ਅਚਲ std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ਅਚਲ ਤਾਰਕਿਕ circular = op_params[8] != 0;

    ਜੇ (circular) {
        ਸਵੈਚਾਲਿਤ res = translate_circular_pad(ਨਿਵੇਸ਼, pads, context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape());
        ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
    }

    ਅਚਲ std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ਅਚਲ std::vector<int64_t> pads_ਸਿਰਾ = {pads[7], pads[5], pads[3], pads[1]};

    ਸਵੈਚਾਲਿਤ pads_begin_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    ਸਵੈਚਾਲਿਤ pads_ਸਿਰਾ_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {pads_ਸਿਰਾ.size()}, pads_ਸਿਰਾ);
    ਸਵੈਚਾਲਿਤ pad_ਮੁੱਲ = ov::op::v0::Constant::create(context.get_ਨਿਵੇਸ਼_type(0), ov::Shape{}, {0});
    ਸਵੈਚਾਲਿਤ res =
        std::make_shared<ov::op::v1::Pad>(ਨਿਵੇਸ਼, pads_begin_ਗੰਢ, pads_ਸਿਰਾ_ਗੰਢ, pad_ਮੁੱਲ, ov::op::Pad_ਬਹੁਲਕ::CONSTANT);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
