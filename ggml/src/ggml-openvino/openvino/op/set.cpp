#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstdint>
#ضمن <openvino/frontend/exception.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/range.hpp>
#ضمن <openvino/op/reduce_prod.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/scatter_update.hpp>
#ضمن <openvino/op/shape_of.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
Output_ويڪٽر translate_سيٽ(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 2, 2);

    خودڪار dst = process_view_داخلا_new(context, 0);
    خودڪار src = process_view_داخلا_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    ثابت خودڪار dst_stride = context.get_داخلا_stride(0);
    FRONT_پڇاڙي_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    ثابت خودڪار * op_params = reinterpret_cast<ثابت uint32_t *>(context.get_output_op_params());
    ثابت size_t وٿي = static_cast<size_t>(op_params[3]);

    ثابت size_t elem_سائيز = dst_stride.back();
    FRONT_پڇاڙي_OP_CONVERSION_CHECK(elem_سائيز != 0 && وٿي % elem_سائيز == 0,
                                  "SET offset must be aligned to destination element size");

    ثابت int64_t وٿي_elems = static_cast<int64_t>(وٿي / elem_سائيز);

    خودڪار dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        خطأ);

    خودڪار src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        خطأ);

    خودڪار src_shape = std::make_shared<ov::op::v3::Shape_جو>(src_flat, ov::element::i64);
    خودڪار src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        خطأ);

    خودڪار start = ov::op::v0::Constant::create(ov::element::i64, {}, {وٿي_elems});
    خودڪار stop = std::make_shared<ov::op::v1::جوڙيو>(start, src_len);
    خودڪار قدم = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    خودڪار indices = std::make_shared<ov::op::v4::Range>(start, stop, قدم, ov::element::i64);
    خودڪار axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    خودڪار updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    خودڪار dst_shape = std::make_shared<ov::op::v3::Shape_جو>(dst, ov::element::i64);
    خودڪار res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, خطأ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
