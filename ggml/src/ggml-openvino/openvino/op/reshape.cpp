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

OutputVector translate_reshape(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಯದಿ (context.get_ಒಳಹರಿವು(0).get_partial_shape().is_static() &&
        context.get_ಒಳಹರಿವು_shape(0) == context.get_output_shape()) {
        ಪ್ರತ್ಯಾಗಮ {context.get_ಒಳಹರಿವು(0)};
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ op_case = context.get_op_case();
    ಧ್ರುವ auto output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_ಗಂಟು;
    ವಿಕಲ್ಪ (op_case) {
    ಪಕ್ಷ 0:
        ಭಙ್ಗ;
    ಪಕ್ಷ 1:
    ಪಕ್ಷ 9:
        shape[1] = -1;
        ಯದಿ (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ 2:
    ಪಕ್ಷ 3:
        shape[2] = -1;
        ಯದಿ (op_case == 3) {
            shape[3] = 1;
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ 4:
        ಪ್ರತ್ಯಾಗಮ {context.get_ಒಳಹರಿವು(0).get_ಗಂಟು_shared_ptr()->ಒಳಹರಿವು_ಮೌಲ್ಯ(0)};
    ಪಕ್ಷ 5:
    ಪಕ್ಷ 7:
        shape = {1, 1, -1, shape[3]};
        ಯದಿ (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        ಯದಿ (context.has_ಒಳಹರಿವು("s_copy_active_slot_len")) {
            auto n_slot_active_len = context.get_ಒಳಹರಿವು("s_copy_active_slot_len");
            auto emb_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_ಗಂಟು =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_size}, 0);
        } ಅನ್ಯಥಾ {
            shape = {1, 1, -1, shape[3]};
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ 8:
        shape[0] = -1;
        ಭಙ್ಗ;
    ಸ್ವಭಾವ:
        FRONT_END_OP_CONVERSION_CHECK(ಅಸತ್ಯ, "Unsupported RESHAPE case: ", op_case);
    }
    ಯದಿ (!new_shape_ಗಂಟು) {
        new_shape_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_ಒಳಹರಿವು(0), new_shape_ಗಂಟು, ಅಸತ್ಯ);
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
