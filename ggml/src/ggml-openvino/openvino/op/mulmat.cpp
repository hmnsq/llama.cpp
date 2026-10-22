#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <climits>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/util/op_types.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_mulmat(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ಪೂರ್ಣಾಂಕ op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<ov::Node> A = process_view_ಒಳಹರಿವು_new(context, 1);

    ಆದರೆ (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_ಒಳಹರಿವು_type(1));
    }

    ಸ್ವಯಂಚಾಲಿತ B_shape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();
    ಸ್ವಯಂಚಾಲಿತ A_shape = context.get_ಒಳಹರಿವು_shape(1).ಗಮ್ಯ_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    ಸ್ವಯಂಚಾಲಿತ A_batch_larger = A_batch > B_batch;
    ಸ್ವಯಂಚಾಲಿತ batch_large = A_batch_larger ? A_batch : B_batch;
    ಸ್ವಯಂಚಾಲಿತ batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    ಸ್ವಯಂಚಾಲಿತ Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t ಗುಣಕ = batch_large / batch_small;
    ಆದರೆ (ಗುಣಕ > 1 && batch_small > 1) {
        ಸ್ವಯಂಚಾಲಿತ batch_large_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        ಸ್ವಯಂಚಾಲಿತ batch_small_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        ಸ್ವಯಂಚಾಲಿತ ಗುಣಕ_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{ಗುಣಕ});

        ಸ್ವಯಂಚಾಲಿತ unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        ಸ್ವಯಂಚಾಲಿತ Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        ಸ್ವಯಂಚಾಲಿತ ಪ್ರಸಾರ_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, ಗುಣಕ, (int64_t) 1, (int64_t) 1});
        ಸ್ವಯಂಚಾಲಿತ new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        ಆದರೆ (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        ಸ್ವಯಂಚಾಲಿತ Z_broadcasted = std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(Z_unsqueezed, ಪ್ರಸಾರ_shape,
                                                                     ov::op::ಪ್ರಸಾರ_Type::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, ನಿಜ);
    }
    ಆದರೆ (A_batch_larger) {
        B = Z;
    } ಇಲ್ಲದಿದ್ದರೆ {
        A = Z;
    }

    ತಾರ್ಕಿಕ transpose_b = ನಿಜ;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, ಸುಳ್ಳು, transpose_b);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
