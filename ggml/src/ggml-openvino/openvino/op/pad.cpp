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
                                            ಧ್ರುವ std::array<int32_t, 8> & pads,
                                            ಧ್ರುವ ov::Shape & ಒಳಹರಿವು_shape) {
    ov::Output<ov::Node> result = std::move(ಒಳಹರಿವು);

    ಧ್ರುವ std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ಧ್ರುವ std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    ಪ್ರತಿ (size_t axis = 0; axis < ಒಳಹರಿವು_shape.size(); ++axis) {
        ಧ್ರುವ int64_t ಒಳಹರಿವು_dim = static_cast<int64_t>(ಒಳಹರಿವು_shape[axis]);
        ಧ್ರುವ int64_t pad_begin = pads_begin[axis];
        ಧ್ರುವ int64_t pad_end = pads_end[axis];

        ಯದಿ (pad_begin == 0 && pad_end == 0) {
            ಅನುವರ್ತ;
        }

        FRONT_END_CHECK_IMPLEMENTED(ಒಳಹರಿವು_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(ಒಳಹರಿವು_dim + pad_begin + pad_end));
        ಪ್ರತಿ (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % ಒಳಹರಿವು_dim;
            ಯದಿ (wrapped < 0) {
                wrapped += ಒಳಹರಿವು_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    ಪ್ರತ್ಯಾಗಮ result;
}

}  // namespace

OutputVector translate_pad(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಯದಿ (context.get_ಒಳಹರಿವು_shape(0) == context.get_output_shape()) {
        ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({ಒಳಹರಿವು}, context.get_name());
    }

    ಧ್ರುವ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    ಧ್ರುವ std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ಧ್ರುವ ತಾರ್ಕಿಕ circular = op_params[8] != 0;

    ಯದಿ (circular) {
        auto res = translate_circular_pad(ಒಳಹರಿವು, pads, context.get_ಒಳಹರಿವು_shape(0).to_shape());
        ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
    }

    ಧ್ರುವ std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ಧ್ರುವ std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_ಮೌಲ್ಯ = ov::op::v0::Constant::create(context.get_ಒಳಹರಿವು_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(ಒಳಹರಿವು, pads_begin_ಗಂಟು, pads_end_ಗಂಟು, pad_ಮೌಲ್ಯ, ov::op::PadMode::CONSTANT);

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
