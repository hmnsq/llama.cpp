#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <array>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/pad.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <utility>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

नामाकाश {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> आगत,
                                            अचर std::array<int32_t, 8> & pads,
                                            अचर ov::Shape & आगत_shape) {
    ov::Output<ov::Node> result = std::move(आगत);

    अचर std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    अचर std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    लागि (माप_प्रकार axis = 0; axis < आगत_shape.size(); ++axis) {
        अचर int64_t आगत_dim = static_cast<int64_t>(आगत_shape[axis]);
        अचर int64_t pad_begin = pads_begin[axis];
        अचर int64_t pad_end = pads_end[axis];

        यदि (pad_begin == 0 && pad_end == 0) {
            जारी;
        }

        FRONT_END_CHECK_IMPLEMENTED(आगत_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<माप_प्रकार>(आगत_dim + pad_begin + pad_end));
        लागि (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % आगत_dim;
            यदि (wrapped < 0) {
                wrapped += आगत_dim;
            }
            indices[static_cast<माप_प्रकार>(index)] = wrapped;
        }

        स्वतः gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        स्वतः gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    फर्काऊ result;
}

}  // namespace

OutputVector translate_pad(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 1, 1);

    स्वतः आगत = process_view_आगत_new(context, 0);
    यदि (context.get_आगत_shape(0) == context.get_output_shape()) {
        फर्काऊ rename_outputs_with_suffix({आगत}, context.get_name());
    }

    अचर int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    अचर std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    अचर तार्किक circular = op_params[8] != 0;

    यदि (circular) {
        स्वतः res = translate_circular_pad(आगत, pads, context.get_आगत_shape(0).to_shape());
        फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
    }

    अचर std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    अचर std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    स्वतः pads_begin_गाँठो = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    स्वतः pads_end_गाँठो = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    स्वतः pad_मान = ov::op::v0::Constant::create(context.get_आगत_type(0), ov::Shape{}, {0});
    स्वतः res =
        std::make_shared<ov::op::v1::Pad>(आगत, pads_begin_गाँठो, pads_end_गाँठो, pad_मान, ov::op::PadMode::CONSTANT);

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
