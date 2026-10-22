#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <climits>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <openvino/op/util/op_types.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_mulmat(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    पूर्णाङ्क op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_निवेश_new(context, 0);
    ov::Output<ov::Node> A = process_view_निवेश_new(context, 1);

    यदि (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_निवेश_type(1));
    }

    स्वतः B_shape = context.get_निवेश_shape(0).to_shape();
    स्वतः A_shape = context.get_निवेश_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    स्वतः A_batch_larger = A_batch > B_batch;
    स्वतः batch_large = A_batch_larger ? A_batch : B_batch;
    स्वतः batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    स्वतः Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    यदि (factor > 1 && batch_small > 1) {
        स्वतः batch_large_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        स्वतः batch_small_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        स्वतः factor_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        स्वतः unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        स्वतः Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        स्वतः broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        स्वतः new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        यदि (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        स्वतः Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, सत्य);
    }
    यदि (A_batch_larger) {
        B = Z;
    } अन्यथा {
        A = Z;
    }

    तार्किक transpose_b = सत्य;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, असत्य, transpose_b);

    ध्रुव स्वतः output_type = context.get_output_type();
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
