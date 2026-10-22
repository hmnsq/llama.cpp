#ضمن "fuse_moe_router.h"

#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/clamp.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/softmax.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/tile.hpp>
#ضمن <openvino/op/topk.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

منطقي constant_is(ثابت ov::Output<ov::Node> & output, ثابت std::vector<int64_t> & قيم) {
    auto عقدة = ov::as_type_ptr<ov::op::v0::Constant>(output.get_عقدة_shared_ptr());
    إرجاع عقدة && عقدة->get_element_type().is_integral_العدد() && عقدة->cast_vector<int64_t>() == قيم;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using نطاق ov::pass::pattern;
    using نطاق ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::قمة_K>({softmax, any_إدخال()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_إدخال(), any_إدخال(), any_إدخال(), any_إدخال()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_إدخال()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_إدخال()});
    auto data = wrap_type<v0::Squeeze>({probs, any_إدخال()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_إدخال(), any_إدخال()});
    auto ids_تعداد = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_إدخال(), any_إدخال()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_تعداد});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_إدخال()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_إدخال()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_إدخال()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_إدخال()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_إدخال()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    ثابت auto callback = [=](Matcher & m) {
        ثابت auto & pm = m.get_pattern_قيمة_map();
        ثابت auto عقدة = [&](ثابت std::shared_ptr<ov::Node> & p) { إرجاع pm.at(p).get_عقدة_shared_ptr(); };
        ثابت auto إدخال_is = [&](ثابت std::shared_ptr<ov::Node> & p, size_t i, ثابت std::vector<int64_t> & v) {
            إرجاع constant_is(عقدة(p)->إدخال_قيمة(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(عقدة(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(عقدة(softmax));
        auto tk = ov::as_type_ptr<v11::قمة_K>(عقدة(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(عقدة(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(عقدة(clamp));
        ثابت auto shape = mm->get_output_partial_shape(0);
        إذا (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_إدخال_partial_shape(0).rank() != 4 ||
            mm->get_إدخال_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::قمة_K::Mode::MAX ||
            tk->get_sort_type() != v11::قمة_K::SortType::SORT_قيم || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            إرجاع خطأ;
        }
        ثابت int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(عقدة(ids)->get_إدخال_عقدة_shared_ptr(2));
        إذا (!end || !end->get_element_type().is_integral_العدد() || ov::shape_size(end->get_shape()) != 1) {
            إرجاع خطأ;
        }
        ثابت int64_t k = end->cast_vector<int64_t>()[0];
        ثابت auto sorted_shape = tk->get_output_partial_shape(1);
        إذا (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !إدخال_is(probs, 1, {1, -1, experts, 1}) ||
            !إدخال_is(data, 1, {0}) || !إدخال_is(ids_2d, 1, {0, 1}) ||
            !إدخال_is(weights_4d, 1, {0}) || !إدخال_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(عقدة(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(عقدة(weights))->get_special_zero() ||
            !إدخال_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(عقدة(gather))->get_batch_dims() != 1 ||
            !إدخال_is(data_batch, 1, {0}) || !إدخال_is(data_batch, 2, {0}) ||
            !إدخال_is(ids_تعداد, 1, {1}) || !إدخال_is(ids_تعداد, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(عقدة(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(عقدة(ids_تعداد))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(عقدة(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(عقدة(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!إدخال_is(sum, 1, {-1}) && !إدخال_is(sum, 1, {3})) ||
            !إدخال_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(عقدة(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            إرجاع خطأ;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        إذا (!(limit->get_min() <= 0.5 * مزدوج(k) / experts && limit->get_max() >= 2.0)) {
            إرجاع خطأ;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        لكل (ثابت auto & إدخال : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(إدخال.get_عقدة()->shared_from_this());
            إذا (!slice || إدخال.get_index() != 0 || slice->get_إدخال_size() != 5 ||
                !constant_is(slice->إدخال_قيمة(1), {0}) || !constant_is(slice->إدخال_قيمة(2), {k}) ||
                !constant_is(slice->إدخال_قيمة(3), {1}) ||
                (!constant_is(slice->إدخال_قيمة(4), {3}) && !constant_is(slice->إدخال_قيمة(4), {-1}))) {
                إرجاع خطأ;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->إدخال_قيمة(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->إدخال_قيمة(1), خطأ, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::قمة_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::قمة_K::Mode::MAX, v11::قمة_K::SortType::SORT_قيم, tk->get_index_element_type(), خطأ);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, عقدة(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_عقدة(عقدة(norm), weights_out);
        لكل (ثابت auto & slice : slices) {
            ov::replace_عقدة(slice, ids_out);
        }
        إرجاع true;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
