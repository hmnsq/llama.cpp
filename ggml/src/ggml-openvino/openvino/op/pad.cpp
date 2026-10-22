#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <array>
#समावेश <cstdint>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/gather.hpp>
#समावेश <openvino/op/pad.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <utility>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> निवेश,
                                            स्थिर std::array<int32_t, 8> & pads,
                                            स्थिर ov::Shape & निवेश_shape) {
    ov::Output<ov::Node> परिणाम = std::move(निवेश);

    स्थिर std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    स्थिर std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    क्रम (माप_प्रकार axis = 0; axis < निवेश_shape.size(); ++axis) {
        स्थिर int64_t निवेश_dim = static_cast<int64_t>(निवेश_shape[axis]);
        स्थिर int64_t pad_begin = pads_begin[axis];
        स्थिर int64_t pad_end = pads_end[axis];

        अगर (pad_begin == 0 && pad_end == 0) {
            जारी;
        }

        FRONT_END_CHECK_IMPLEMENTED(निवेश_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<माप_प्रकार>(निवेश_dim + pad_begin + pad_end));
        क्रम (int64_t सूचकांक = 0; सूचकांक < static_cast<int64_t>(indices.size()); ++सूचकांक) {
            int64_t wrapped = (सूचकांक - pad_begin) % निवेश_dim;
            अगर (wrapped < 0) {
                wrapped += निवेश_dim;
            }
            indices[static_cast<माप_प्रकार>(सूचकांक)] = wrapped;
        }

        स्वतः gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        स्वतः gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        परिणाम = std::make_shared<ov::op::v8::Gather>(परिणाम, gather_indices, gather_axis);
    }

    वापस परिणाम;
}

}  // namespace

OutputVector translate_pad(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः निवेश = process_view_निवेश_new(context, 0);
    अगर (context.get_निवेश_shape(0) == context.get_output_shape()) {
        वापस rename_outputs_with_suffix({निवेश}, context.get_name());
    }

    स्थिर int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    स्थिर std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    स्थिर bool circular = op_params[8] != 0;

    अगर (circular) {
        स्वतः res = translate_circular_pad(निवेश, pads, context.get_निवेश_shape(0).to_shape());
        वापस rename_outputs_with_suffix({res}, context.get_name());
    }

    स्थिर std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    स्थिर std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    स्वतः pads_begin_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    स्वतः pads_end_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    स्वतः pad_मूल्य = ov::op::v0::Constant::create(context.get_निवेश_type(0), ov::Shape{}, {0});
    स्वतः res =
        std::make_shared<ov::op::v1::Pad>(निवेश, pads_begin_गाँठ, pads_end_गाँठ, pad_मूल्य, ov::op::PadMode::CONSTANT);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
