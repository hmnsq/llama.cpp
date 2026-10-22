#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/exception.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_reshape(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಆದರೆ (context.get_ಒಳಹರಿವು(0).get_partial_shape().is_static() &&
        context.get_ಒಳಹರಿವು_shape(0) == context.get_output_shape()) {
        ಹಿಂತಿರುಗಿ {context.get_ಒಳಹರಿವು(0)};
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ op_case = context.get_op_case();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_shape = context.get_output_shape().ಗಮ್ಯ_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_ಗಂಟು;
    ಆಯ್ಕೆ (op_case) {
    ಸ್ಥಿತಿ 0:
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ 1:
    ಸ್ಥಿತಿ 9:
        shape[1] = -1;
        ಆದರೆ (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ 2:
    ಸ್ಥಿತಿ 3:
        shape[2] = -1;
        ಆದರೆ (op_case == 3) {
            shape[3] = 1;
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ 4:
        ಹಿಂತಿರುಗಿ {context.get_ಒಳಹರಿವು(0).get_ಗಂಟು_shared_ptr()->ಒಳಹರಿವು_ಮೌಲ್ಯ(0)};
    ಸ್ಥಿತಿ 5:
    ಸ್ಥಿತಿ 7:
        shape = {1, 1, -1, shape[3]};
        ಆದರೆ (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        ಆದರೆ (context.has_ಒಳಹರಿವು("s_copy_active_slot_len")) {
            ಸ್ವಯಂಚಾಲಿತ n_ಸ್ಥಾನ_ಸಕ್ರಿಯ_len = context.get_ಒಳಹರಿವು("s_copy_active_slot_len");
            ಸ್ವಯಂಚಾಲಿತ emb_ಗಾತ್ರ = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            ಸ್ವಯಂಚಾಲಿತ neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_ಗಂಟು =
                std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{one, n_ಸ್ಥಾನ_ಸಕ್ರಿಯ_len, neg_one, emb_ಗಾತ್ರ}, 0);
        } ಇಲ್ಲದಿದ್ದರೆ {
            shape = {1, 1, -1, shape[3]};
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ 8:
        shape[0] = -1;
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಸುಳ್ಳು, "Unsupported RESHAPE case: ", op_case);
    }
    ಆದರೆ (!new_shape_ಗಂಟು) {
        new_shape_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v1::Reshape>(context.get_ಒಳಹರಿವು(0), new_shape_ಗಂಟು, ಸುಳ್ಳು);
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
