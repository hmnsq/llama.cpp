#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/divide.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/tile.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
Output_ಸದಿಶ translate_repeat(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 2);

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_shape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_shape = context.get_output_shape().ಗಮ್ಯ_shape();

    std::vector<int64_t> repeats(4, 1);
    ಗಾಗಿ (size_t axis = 0; axis < 4; ++axis) {
        ಸ್ಥಿರಾಂಕ int64_t ಒಳಹರಿವು_dim = ಒಳಹರಿವು_shape[axis];
        ಸ್ಥಿರಾಂಕ int64_t output_dim = output_shape[axis];

        FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಒಳಹರಿವು_dim > 0 && output_dim > 0 && output_dim % ಒಳಹರಿವು_dim == 0,
                                      "REPEAT input shape ", ಒಳಹರಿವು_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / ಒಳಹರಿವು_dim;
    }

    ಸ್ವಯಂಚಾಲಿತ repeats_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(ಒಳಹರಿವು, repeats_ಗಂಟು);
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
