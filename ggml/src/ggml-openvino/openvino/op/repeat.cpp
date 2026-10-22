#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/tile.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
Output_सदिशम् translate_repeat(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 2);

    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);

    ध्रुव स्वचालित निवेशः_shape = context.get_निवेशः_shape(0).गन्तव्यम्_shape();
    ध्रुव स्वचालित output_shape = context.get_output_shape().गन्तव्यम्_shape();

    std::vector<int64_t> repeats(4, 1);
    प्रति (size_t axis = 0; axis < 4; ++axis) {
        ध्रुव int64_t निवेशः_dim = निवेशः_shape[axis];
        ध्रुव int64_t output_dim = output_shape[axis];

        FRONT_अन्तः_OP_CONVERSION_CHECK(निवेशः_dim > 0 && output_dim > 0 && output_dim % निवेशः_dim == 0,
                                      "REPEAT input shape ", निवेशः_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / निवेशः_dim;
    }

    स्वचालित repeats_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(निवेशः, repeats_ग्रन्थिः);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
