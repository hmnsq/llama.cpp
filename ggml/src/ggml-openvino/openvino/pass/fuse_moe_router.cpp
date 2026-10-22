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

منطقي constant_is(ثابت ov::Output<ov::Node> & output, ثابت std::vector<int64_t> & اقدار) {
    auto گرہ = ov::as_type_ptr<ov::op::v0::Constant>(output.get_گرہ_shared_ptr());
    أرجع گرہ && گرہ->get_element_type().is_integral_نمبر() && گرہ->cast_vector<int64_t>() == اقدار;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using نطاق ov::pass::pattern;
    using نطاق ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::چوٹی_K>({softmax, any_داخلہ()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_داخلہ(), any_داخلہ(), any_داخلہ(), any_داخلہ()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_داخلہ()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_داخلہ()});
    auto data = wrap_type<v0::Squeeze>({probs, any_داخلہ()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_داخلہ(), any_داخلہ()});
    auto ids_گنتی = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_داخلہ(), any_داخلہ()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_گنتی});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_داخلہ()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_داخلہ()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_داخلہ()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_داخلہ()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_داخلہ()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    ثابت auto callback = [=](Matcher & m) {
        ثابت auto & pm = m.get_pattern_قدر_map();
        ثابت auto گرہ = [&](ثابت std::shared_ptr<ov::Node> & p) { أرجع pm.at(p).get_گرہ_shared_ptr(); };
        ثابت auto داخلہ_is = [&](ثابت std::shared_ptr<ov::Node> & p, size_t i, ثابت std::vector<int64_t> & v) {
            أرجع constant_is(گرہ(p)->داخلہ_قدر(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(گرہ(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(گرہ(softmax));
        auto tk = ov::as_type_ptr<v11::چوٹی_K>(گرہ(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(گرہ(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(گرہ(clamp));
        ثابت auto shape = mm->get_output_partial_shape(0);
        إذا (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_داخلہ_partial_shape(0).rank() != 4 ||
            mm->get_داخلہ_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::چوٹی_K::Mode::MAX ||
            tk->get_sort_type() != v11::چوٹی_K::SortType::SORT_اقدار || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            أرجع خطأ;
        }
        ثابت int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(گرہ(ids)->get_داخلہ_گرہ_shared_ptr(2));
        إذا (!end || !end->get_element_type().is_integral_نمبر() || ov::shape_size(end->get_shape()) != 1) {
            أرجع خطأ;
        }
        ثابت int64_t k = end->cast_vector<int64_t>()[0];
        ثابت auto sorted_shape = tk->get_output_partial_shape(1);
        إذا (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !داخلہ_is(probs, 1, {1, -1, experts, 1}) ||
            !داخلہ_is(data, 1, {0}) || !داخلہ_is(ids_2d, 1, {0, 1}) ||
            !داخلہ_is(weights_4d, 1, {0}) || !داخلہ_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(گرہ(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(گرہ(weights))->get_special_zero() ||
            !داخلہ_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(گرہ(gather))->get_batch_dims() != 1 ||
            !داخلہ_is(data_batch, 1, {0}) || !داخلہ_is(data_batch, 2, {0}) ||
            !داخلہ_is(ids_گنتی, 1, {1}) || !داخلہ_is(ids_گنتی, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(گرہ(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(گرہ(ids_گنتی))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(گرہ(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(گرہ(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!داخلہ_is(sum, 1, {-1}) && !داخلہ_is(sum, 1, {3})) ||
            !داخلہ_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(گرہ(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            أرجع خطأ;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        إذا (!(limit->get_min() <= 0.5 * مزدوج(k) / experts && limit->get_max() >= 2.0)) {
            أرجع خطأ;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        لكل (ثابت auto & داخلہ : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(داخلہ.get_گرہ()->shared_from_this());
            إذا (!slice || داخلہ.get_index() != 0 || slice->get_داخلہ_size() != 5 ||
                !constant_is(slice->داخلہ_قدر(1), {0}) || !constant_is(slice->داخلہ_قدر(2), {k}) ||
                !constant_is(slice->داخلہ_قدر(3), {1}) ||
                (!constant_is(slice->داخلہ_قدر(4), {3}) && !constant_is(slice->داخلہ_قدر(4), {-1}))) {
                أرجع خطأ;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->داخلہ_قدر(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->داخلہ_قدر(1), خطأ, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::چوٹی_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::چوٹی_K::Mode::MAX, v11::چوٹی_K::SortType::SORT_اقدار, tk->get_index_element_type(), خطأ);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, گرہ(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_گرہ(گرہ(norm), weights_out);
        لكل (ثابت auto & slice : slices) {
            ov::replace_گرہ(slice, ids_out);
        }
        أرجع true;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
