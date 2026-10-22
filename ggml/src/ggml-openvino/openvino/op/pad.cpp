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

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> निवेशः,
                                            ध्रुव std::array<int32_t, 8> & pads,
                                            ध्रुव ov::Shape & निवेशः_shape) {
    ov::Output<ov::Node> परिणाम = std::move(निवेशः);

    ध्रुव std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ध्रुव std::array<int32_t, 4> pads_अन्तः = {pads[7], pads[5], pads[3], pads[1]};

    प्रति (size_t axis = 0; axis < निवेशः_shape.size(); ++axis) {
        ध्रुव int64_t निवेशः_dim = static_cast<int64_t>(निवेशः_shape[axis]);
        ध्रुव int64_t pad_begin = pads_begin[axis];
        ध्रुव int64_t pad_अन्तः = pads_अन्तः[axis];

        यदि (pad_begin == 0 && pad_अन्तः == 0) {
            अनुवर्त;
        }

        FRONT_अन्तः_CHECK_IMPLEMENTED(निवेशः_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(निवेशः_dim + pad_begin + pad_अन्तः));
        प्रति (int64_t सूचकांक = 0; सूचकांक < static_cast<int64_t>(indices.size()); ++सूचकांक) {
            int64_t wrapped = (सूचकांक - pad_begin) % निवेशः_dim;
            यदि (wrapped < 0) {
                wrapped += निवेशः_dim;
            }
            indices[static_cast<size_t>(सूचकांक)] = wrapped;
        }

        स्वचालित gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        स्वचालित gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        परिणाम = std::make_shared<ov::op::v8::Gather>(परिणाम, gather_indices, gather_axis);
    }

    प्रत्यागम परिणाम;
}

}  // namespace

Output_सदिशम् translate_pad(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    यदि (context.get_निवेशः_shape(0) == context.get_output_shape()) {
        प्रत्यागम rename_outputs_with_suffix({निवेशः}, context.get_name());
    }

    ध्रुव int32_t * op_params = context.get_output_op_params();
    FRONT_अन्तः_CHECK_IMPLEMENTED(op_params != शून्यसूचकः, "PAD requires output op params");

    ध्रुव std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ध्रुव तार्किक circular = op_params[8] != 0;

    यदि (circular) {
        स्वचालित res = translate_circular_pad(निवेशः, pads, context.get_निवेशः_shape(0).गन्तव्यम्_shape());
        प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
    }

    ध्रुव std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ध्रुव std::vector<int64_t> pads_अन्तः = {pads[7], pads[5], pads[3], pads[1]};

    स्वचालित pads_begin_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    स्वचालित pads_अन्तः_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::i64, {pads_अन्तः.size()}, pads_अन्तः);
    स्वचालित pad_मानम् = ov::op::v0::Constant::create(context.get_निवेशः_type(0), ov::Shape{}, {0});
    स्वचालित res =
        std::make_shared<ov::op::v1::Pad>(निवेशः, pads_begin_ग्रन्थिः, pads_अन्तः_ग्रन्थिः, pad_मानम्, ov::op::Pad_बहुलकम्::CONSTANT);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
