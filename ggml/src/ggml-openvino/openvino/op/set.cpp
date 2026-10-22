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
Output_ویکټور translate_سېټ(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 2, 2);

    خپلکاری dst = process_view_ننوت_new(context, 0);
    خپلکاری src = process_view_ننوت_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    ثابت خپلکاری dst_stride = context.get_ننوت_stride(0);
    FRONT_پای_ټکی_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    ثابت خپلکاری * op_params = reinterpret_cast<ثابت uint32_t *>(context.get_output_op_params());
    ثابت size_t دننه_واټن = static_cast<size_t>(op_params[3]);

    ثابت size_t elem_اندازه = dst_stride.back();
    FRONT_پای_ټکی_OP_CONVERSION_CHECK(elem_اندازه != 0 && دننه_واټن % elem_اندازه == 0,
                                  "SET offset must be aligned to destination element size");

    ثابت int64_t دننه_واټن_elems = static_cast<int64_t>(دننه_واټن / elem_اندازه);

    خپلکاری dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        خطأ);

    خپلکاری src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        خطأ);

    خپلکاری src_shape = std::make_shared<ov::op::v3::Shape_د>(src_flat, ov::element::i64);
    خپلکاری src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        خطأ);

    خپلکاری start = ov::op::v0::Constant::create(ov::element::i64, {}, {دننه_واټن_elems});
    خپلکاری stop = std::make_shared<ov::op::v1::جمع_کړه>(start, src_len);
    خپلکاری ګام = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    خپلکاری indices = std::make_shared<ov::op::v4::Range>(start, stop, ګام, ov::element::i64);
    خپلکاری axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    خپلکاری updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    خپلکاری dst_shape = std::make_shared<ov::op::v3::Shape_د>(dst, ov::element::i64);
    خپلکاری res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, خطأ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
