#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <climits>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/matmul.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/util/op_types.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_mulmat(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ਪੂਰਨ op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ov::Output<ov::Node> A = process_view_ਨਿਵੇਸ਼_new(context, 1);

    ਜੇ (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_ਨਿਵੇਸ਼_type(1));
    }

    ਸਵੈਚਾਲਿਤ B_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();
    ਸਵੈਚਾਲਿਤ A_shape = context.get_ਨਿਵੇਸ਼_shape(1).ਮੰਜ਼ਿਲ_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    ਸਵੈਚਾਲਿਤ A_batch_larger = A_batch > B_batch;
    ਸਵੈਚਾਲਿਤ batch_large = A_batch_larger ? A_batch : B_batch;
    ਸਵੈਚਾਲਿਤ batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    ਸਵੈਚਾਲਿਤ Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t ਗੁਣਕ = batch_large / batch_small;
    ਜੇ (ਗੁਣਕ > 1 && batch_small > 1) {
        ਸਵੈਚਾਲਿਤ batch_large_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        ਸਵੈਚਾਲਿਤ batch_small_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        ਸਵੈਚਾਲਿਤ ਗੁਣਕ_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{ਗੁਣਕ});

        ਸਵੈਚਾਲਿਤ unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        ਸਵੈਚਾਲਿਤ Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        ਸਵੈਚਾਲਿਤ ਪ੍ਰਸਾਰਣ_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, ਗੁਣਕ, (int64_t) 1, (int64_t) 1});
        ਸਵੈਚਾਲਿਤ new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        ਜੇ (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        ਸਵੈਚਾਲਿਤ Z_broadcasted = std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(Z_unsqueezed, ਪ੍ਰਸਾਰਣ_shape,
                                                                     ov::op::ਪ੍ਰਸਾਰਣ_Type::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, ਸੱਚ);
    }
    ਜੇ (A_batch_larger) {
        B = Z;
    } ਨਹੀਂਤਾਂ {
        A = Z;
    }

    ਤਾਰਕਿਕ transpose_b = ਸੱਚ;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, ਝੂਠ, transpose_b);

    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
