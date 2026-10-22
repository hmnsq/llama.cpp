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

OutputVector translate_mulmat(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ಪೂರ್ಣಾಙ್ಕ op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<ov::Node> A = process_view_ಒಳಹರಿವು_new(context, 1);

    ಯದಿ (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_ಒಳಹರಿವು_type(1));
    }

    auto B_shape = context.get_ಒಳಹರಿವು_shape(0).to_shape();
    auto A_shape = context.get_ಒಳಹರಿವು_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    auto Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    ಯದಿ (factor > 1 && batch_small > 1) {
        auto batch_large_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        ಯದಿ (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, ಸತ್ಯ);
    }
    ಯದಿ (A_batch_larger) {
        B = Z;
    } ಅನ್ಯಥಾ {
        A = Z;
    }

    ತಾರ್ಕಿಕ transpose_b = ಸತ್ಯ;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, ಅಸತ್ಯ, transpose_b);

    ಧ್ರುವ auto output_type = context.get_output_type();
    ಯದಿ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
