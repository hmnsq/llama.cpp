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

منطقي constant_is(ثابت ov::Output<ov::Node> & output, ثابت std::vector<int64_t> & درجوجي) {
    auto كلي = ov::as_type_ptr<ov::op::v0::Constant>(output.get_كلي_shared_ptr());
    أرجع كلي && كلي->get_element_type().is_integral_لمب() && كلي->cast_vector<int64_t>() == درجوجي;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using نطاق ov::pass::pattern;
    using نطاق ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::كولي_K>({softmax, any_شغرو()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_شغرو(), any_شغرو(), any_شغرو(), any_شغرو()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_شغرو()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_شغرو()});
    auto data = wrap_type<v0::Squeeze>({probs, any_شغرو()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_شغرو(), any_شغرو()});
    auto ids_كدي = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_شغرو(), any_شغرو()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_كدي});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_شغرو()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_شغرو()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_شغرو()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_شغرو()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_شغرو()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    ثابت auto callback = [=](Matcher & m) {
        ثابت auto & pm = m.get_pattern_درج_map();
        ثابت auto كلي = [&](ثابت std::shared_ptr<ov::Node> & p) { أرجع pm.at(p).get_كلي_shared_ptr(); };
        ثابت auto شغرو_is = [&](ثابت std::shared_ptr<ov::Node> & p, size_t i, ثابت std::vector<int64_t> & v) {
            أرجع constant_is(كلي(p)->شغرو_درج(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(كلي(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(كلي(softmax));
        auto tk = ov::as_type_ptr<v11::كولي_K>(كلي(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(كلي(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(كلي(clamp));
        ثابت auto shape = mm->get_output_partial_shape(0);
        إذا (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_شغرو_partial_shape(0).rank() != 4 ||
            mm->get_شغرو_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::كولي_K::Mode::MAX ||
            tk->get_sort_type() != v11::كولي_K::SortType::SORT_درجوجي || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            أرجع خطأ;
        }
        ثابت int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(كلي(ids)->get_شغرو_كلي_shared_ptr(2));
        إذا (!end || !end->get_element_type().is_integral_لمب() || ov::shape_size(end->get_shape()) != 1) {
            أرجع خطأ;
        }
        ثابت int64_t k = end->cast_vector<int64_t>()[0];
        ثابت auto sorted_shape = tk->get_output_partial_shape(1);
        إذا (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !شغرو_is(probs, 1, {1, -1, experts, 1}) ||
            !شغرو_is(data, 1, {0}) || !شغرو_is(ids_2d, 1, {0, 1}) ||
            !شغرو_is(weights_4d, 1, {0}) || !شغرو_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(كلي(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(كلي(weights))->get_special_zero() ||
            !شغرو_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(كلي(gather))->get_batch_dims() != 1 ||
            !شغرو_is(data_batch, 1, {0}) || !شغرو_is(data_batch, 2, {0}) ||
            !شغرو_is(ids_كدي, 1, {1}) || !شغرو_is(ids_كدي, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(كلي(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(كلي(ids_كدي))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(كلي(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(كلي(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!شغرو_is(sum, 1, {-1}) && !شغرو_is(sum, 1, {3})) ||
            !شغرو_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(كلي(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            أرجع خطأ;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        إذا (!(limit->get_min() <= 0.5 * مزدوج(k) / experts && limit->get_max() >= 2.0)) {
            أرجع خطأ;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        لكل (ثابت auto & شغرو : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(شغرو.get_كلي()->shared_from_this());
            إذا (!slice || شغرو.get_index() != 0 || slice->get_شغرو_size() != 5 ||
                !constant_is(slice->شغرو_درج(1), {0}) || !constant_is(slice->شغرو_درج(2), {k}) ||
                !constant_is(slice->شغرو_درج(3), {1}) ||
                (!constant_is(slice->شغرو_درج(4), {3}) && !constant_is(slice->شغرو_درج(4), {-1}))) {
                أرجع خطأ;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->شغرو_درج(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->شغرو_درج(1), خطأ, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::كولي_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::كولي_K::Mode::MAX, v11::كولي_K::SortType::SORT_درجوجي, tk->get_index_element_type(), خطأ);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, كلي(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_كلي(كلي(norm), weights_out);
        لكل (ثابت auto & slice : slices) {
            ov::replace_كلي(slice, ids_out);
        }
        أرجع true;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
