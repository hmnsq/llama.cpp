#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <array>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/gather.hpp>
#അന്തര്ഭാവ <openvino/op/pad.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/shape_of.hpp>
#അന്തര്ഭാവ <utility>
#അന്തര്ഭാവ <vector>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

നാമാകാശ {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> നിവേശം,
                                            സ്ഥിരം std::array<int32_t, 8> & pads,
                                            സ്ഥിരം ov::Shape & നിവേശം_shape) {
    ov::Output<ov::Node> result = std::move(നിവേശം);

    സ്ഥിരം std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    സ്ഥിരം std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    വേണ്ടി (size_t axis = 0; axis < നിവേശം_shape.size(); ++axis) {
        സ്ഥിരം int64_t നിവേശം_dim = static_cast<int64_t>(നിവേശം_shape[axis]);
        സ്ഥിരം int64_t pad_begin = pads_begin[axis];
        സ്ഥിരം int64_t pad_end = pads_end[axis];

        എങ്കിൽ (pad_begin == 0 && pad_end == 0) {
            തുടരുക;
        }

        FRONT_END_CHECK_IMPLEMENTED(നിവേശം_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(നിവേശം_dim + pad_begin + pad_end));
        വേണ്ടി (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % നിവേശം_dim;
            എങ്കിൽ (wrapped < 0) {
                wrapped += നിവേശം_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    തിരികെ result;
}

}  // namespace

OutputVector translate_pad(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    എങ്കിൽ (context.get_നിവേശം_shape(0) == context.get_output_shape()) {
        തിരികെ rename_outputs_with_suffix({നിവേശം}, context.get_name());
    }

    സ്ഥിരം int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    സ്ഥിരം std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    സ്ഥിരം തർക്കം circular = op_params[8] != 0;

    എങ്കിൽ (circular) {
        auto res = translate_circular_pad(നിവേശം, pads, context.get_നിവേശം_shape(0).to_shape());
        തിരികെ rename_outputs_with_suffix({res}, context.get_name());
    }

    സ്ഥിരം std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    സ്ഥിരം std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_മൂല്യം = ov::op::v0::Constant::create(context.get_നിവേശം_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(നിവേശം, pads_begin_കണ്ണി, pads_end_കണ്ണി, pad_മൂല്യം, ov::op::PadMode::CONSTANT);

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
