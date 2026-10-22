#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/pad.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <utility>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

ನಾಮಾಕಾಶ {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> ಒಳಹರಿವು,
                                            ಸ್ಥಿರಾಂಕ std::array<int32_t, 8> & pads,
                                            ಸ್ಥಿರಾಂಕ ov::Shape & ಒಳಹರಿವು_shape) {
    ov::Output<ov::Node> ಫಲಿತಾಂಶ = std::move(ಒಳಹರಿವು);

    ಸ್ಥಿರಾಂಕ std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ಸ್ಥಿರಾಂಕ std::array<int32_t, 4> pads_ಕೊನೆ = {pads[7], pads[5], pads[3], pads[1]};

    ಗಾಗಿ (size_t axis = 0; axis < ಒಳಹರಿವು_shape.size(); ++axis) {
        ಸ್ಥಿರಾಂಕ int64_t ಒಳಹರಿವು_dim = static_cast<int64_t>(ಒಳಹರಿವು_shape[axis]);
        ಸ್ಥಿರಾಂಕ int64_t pad_begin = pads_begin[axis];
        ಸ್ಥಿರಾಂಕ int64_t pad_ಕೊನೆ = pads_ಕೊನೆ[axis];

        ಆದರೆ (pad_begin == 0 && pad_ಕೊನೆ == 0) {
            ಮುಂದುವರಿ;
        }

        FRONT_ಕೊನೆ_CHECK_IMPLEMENTED(ಒಳಹರಿವು_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(ಒಳಹರಿವು_dim + pad_begin + pad_ಕೊನೆ));
        ಗಾಗಿ (int64_t ಸೂಚ್ಯಂಕ = 0; ಸೂಚ್ಯಂಕ < static_cast<int64_t>(indices.size()); ++ಸೂಚ್ಯಂಕ) {
            int64_t wrapped = (ಸೂಚ್ಯಂಕ - pad_begin) % ಒಳಹರಿವು_dim;
            ಆದರೆ (wrapped < 0) {
                wrapped += ಒಳಹರಿವು_dim;
            }
            indices[static_cast<size_t>(ಸೂಚ್ಯಂಕ)] = wrapped;
        }

        ಸ್ವಯಂಚಾಲಿತ gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        ಸ್ವಯಂಚಾಲಿತ gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        ಫಲಿತಾಂಶ = std::make_shared<ov::op::v8::Gather>(ಫಲಿತಾಂಶ, gather_indices, gather_axis);
    }

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

}  // namespace

Output_ಸದಿಶ translate_pad(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಆದರೆ (context.get_ಒಳಹರಿವು_shape(0) == context.get_output_shape()) {
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({ಒಳಹರಿವು}, context.get_name());
    }

    ಸ್ಥಿರಾಂಕ int32_t * op_params = context.get_output_op_params();
    FRONT_ಕೊನೆ_CHECK_IMPLEMENTED(op_params != ಶೂನ್ಯಸೂಚಕ, "PAD requires output op params");

    ಸ್ಥಿರಾಂಕ std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ circular = op_params[8] != 0;

    ಆದರೆ (circular) {
        ಸ್ವಯಂಚಾಲಿತ res = translate_circular_pad(ಒಳಹರಿವು, pads, context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape());
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
    }

    ಸ್ಥಿರಾಂಕ std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ಸ್ಥಿರಾಂಕ std::vector<int64_t> pads_ಕೊನೆ = {pads[7], pads[5], pads[3], pads[1]};

    ಸ್ವಯಂಚಾಲಿತ pads_begin_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    ಸ್ವಯಂಚಾಲಿತ pads_ಕೊನೆ_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {pads_ಕೊನೆ.size()}, pads_ಕೊನೆ);
    ಸ್ವಯಂಚಾಲಿತ pad_ಮೌಲ್ಯ = ov::op::v0::Constant::create(context.get_ಒಳಹರಿವು_type(0), ov::Shape{}, {0});
    ಸ್ವಯಂಚಾಲಿತ res =
        std::make_shared<ov::op::v1::Pad>(ಒಳಹರಿವು, pads_begin_ಗಂಟು, pads_ಕೊನೆ_ಗಂಟು, pad_ಮೌಲ್ಯ, ov::op::Pad_ಬಹುಲಕ::CONSTANT);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
