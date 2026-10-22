#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/reduce_mean.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mean(const बुहोल_Context & context) {
    नुमेरो_inputs_check(context, 1, 1);

    auto पासोक = process_view_पासोक_new(context, 0);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v1::ReduceMean>(पासोक, axis, true);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
