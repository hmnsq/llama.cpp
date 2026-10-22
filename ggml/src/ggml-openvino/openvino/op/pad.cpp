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

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> आदान,
                                            ध्रुव std::array<int32_t, 8> & pads,
                                            ध्रुव ov::Shape & आदान_shape) {
    ov::Output<ov::Node> result = std::move(आदान);

    ध्रुव std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ध्रुव std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    प्रति (size_t axis = 0; axis < आदान_shape.size(); ++axis) {
        ध्रुव int64_t आदान_dim = static_cast<int64_t>(आदान_shape[axis]);
        ध्रुव int64_t pad_begin = pads_begin[axis];
        ध्रुव int64_t pad_end = pads_end[axis];

        यदि (pad_begin == 0 && pad_end == 0) {
            अनुवर्त;
        }

        FRONT_END_CHECK_IMPLEMENTED(आदान_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(आदान_dim + pad_begin + pad_end));
        प्रति (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % आदान_dim;
            यदि (wrapped < 0) {
                wrapped += आदान_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    प्रत्यागम result;
}

}  // namespace

OutputVector translate_pad(ध्रुव गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    auto आदान = process_view_आदान_new(context, 0);
    यदि (context.get_आदान_shape(0) == context.get_output_shape()) {
        प्रत्यागम rename_outputs_with_suffix({आदान}, context.get_name());
    }

    ध्रुव int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    ध्रुव std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ध्रुव तार्किक circular = op_params[8] != 0;

    यदि (circular) {
        auto res = translate_circular_pad(आदान, pads, context.get_आदान_shape(0).to_shape());
        प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
    }

    ध्रुव std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ध्रुव std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_गाठ = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_गाठ = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_मूल्य = ov::op::v0::Constant::create(context.get_आदान_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(आदान, pads_begin_गाठ, pads_end_गाठ, pad_मूल्य, ov::op::PadMode::CONSTANT);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
