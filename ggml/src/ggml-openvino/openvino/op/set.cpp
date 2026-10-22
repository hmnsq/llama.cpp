#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <cstdint>
#समावेश <openvino/frontend/exception.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/range.hpp>
#समावेश <openvino/op/reduce_prod.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/scatter_update.hpp>
#समावेश <openvino/op/shape_of.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
OutputVector translate_set(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    स्वतः dst = process_view_निवेश_new(context, 0);
    स्वतः src = process_view_निवेश_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    स्थिर स्वतः dst_stride = context.get_निवेश_stride(0);
    FRONT_END_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    स्थिर स्वतः * op_params = reinterpret_cast<स्थिर uint32_t *>(context.get_output_op_params());
    स्थिर माप_प्रकार offset = static_cast<माप_प्रकार>(op_params[3]);

    स्थिर माप_प्रकार elem_आकार = dst_stride.back();
    FRONT_END_OP_CONVERSION_CHECK(elem_आकार != 0 && offset % elem_आकार == 0,
                                  "SET offset must be aligned to destination element size");

    स्थिर int64_t offset_elems = static_cast<int64_t>(offset / elem_आकार);

    स्वतः dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        false);

    स्वतः src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        false);

    स्वतः src_shape = std::make_shared<ov::op::v3::ShapeOf>(src_flat, ov::element::i64);
    स्वतः src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        false);

    स्वतः start = ov::op::v0::Constant::create(ov::element::i64, {}, {offset_elems});
    स्वतः stop = std::make_shared<ov::op::v1::Add>(start, src_len);
    स्वतः step = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    स्वतः indices = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);
    स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    स्वतः updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    स्वतः dst_shape = std::make_shared<ov::op::v3::ShapeOf>(dst, ov::element::i64);
    स्वतः res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, false);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
