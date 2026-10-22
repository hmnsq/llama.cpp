#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <cassert>
#समावेश <cstdint>
#समावेश <memory>
#समावेश <openvino/core/node.hpp>
#समावेश <openvino/core/node_output.hpp>
#समावेश <openvino/core/type.hpp>
#समावेश <openvino/frontend/exception.hpp>
#समावेश <openvino/op/broadcast.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/gather.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/scatter_elements_update.hpp>
#समावेश <openvino/op/scatter_update.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/squeeze.hpp>
#समावेश <openvino/op/transpose.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_set_rows(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 3, 3);

    स्वतः तथ्य = process_view_निवेश_new(context, 0);
    स्वतः indices = process_view_निवेश_new(context, 1);
    स्वतः dst = process_view_निवेश_new(context, 2);

    तथ्य = std::make_shared<ov::op::v0::Convert>(तथ्य, context.get_output_type());

    स्थिर स्वतः indices_shape = context.get_निवेश_shape(1);
    स्थिर bool multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_लंबाई() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_लंबाई() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_लंबाई() > 1));

    स्वतः axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    अगर (context.is_stateful()) {
        पूर्णांक concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_लंबाई();
        int64_t dim3 = dst.get_partial_shape()[3].get_लंबाई();
        तथ्य = std::make_shared<ov::op::v1::Reshape>(
            तथ्य, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), false);
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{dst, तथ्य}, concat_axis);
    } अन्यथा अगर (multidim_indices) {
        स्वतः updates_shape = std::make_shared<ov::op::v3::ShapeOf>(तथ्य, ov::element::i64);

        स्वतः indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः indices_rank4_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{get_dimensions(updates_shape, {0, 1, 2}), one}, 0);
        स्वतः indices_rank4 = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rank4_shape, false);
        स्वतः broadcasted_indices = std::make_shared<ov::op::v3::Broadcast>(indices_rank4, updates_shape);

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, तथ्य, axes);
    } अन्यथा {
        स्वतः row_आकार = context.get_निवेश_shape(2)[3].get_लंबाई();
        स्वतः ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        स्वतः तथ्य_reshaped = std::make_shared<ov::op::v1::Reshape>(
            तथ्य,
            ov::op::v0::Constant::create(ov::element::i64, {4},
                                         {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) row_आकार}),
            false);
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, तथ्य_reshaped, axes);
    }

    स्वतः dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_गाँठ_shared_ptr());
    अगर (!multidim_indices && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        स्वतः dst_shape_partial = dst_reshape->get_निवेश_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_लंबाई(), dst_shape_partial[1].get_लंबाई(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_लंबाई() : -1,
                                          dst_shape_partial[3].get_लंबाई()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    false);
    }
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
