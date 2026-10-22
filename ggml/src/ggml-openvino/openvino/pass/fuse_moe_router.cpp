#অন্তর্ভাব "fuse_moe_router.h"

#অন্তর্ভাব <openvino/core/graph_util.hpp>
#অন্তর্ভাব <openvino/core/rt_info.hpp>
#অন্তর্ভাব <openvino/op/broadcast.hpp>
#অন্তর্ভাব <openvino/op/clamp.hpp>
#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/divide.hpp>
#অন্তর্ভাব <openvino/op/gather.hpp>
#অন্তর্ভাব <openvino/op/matmul.hpp>
#অন্তর্ভাব <openvino/op/reduce_sum.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>
#অন্তর্ভাব <openvino/op/softmax.hpp>
#অন্তর্ভাব <openvino/op/squeeze.hpp>
#অন্তর্ভাব <openvino/op/tile.hpp>
#অন্তর্ভাব <openvino/op/topk.hpp>
#অন্তর্ভাব <openvino/op/unsqueeze.hpp>
#অন্তর্ভাব <openvino/pass/pattern/op/wrap_type.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ pass {

নামাকাশ {

তার্কিক constant_is(ধ্রুব ov::Output<ov::Node> & output, ধ্রুব std::vector<int64_t> & মানসমূহ) {
    auto গ্রন্থি = ov::as_type_ptr<ov::op::v0::Constant>(output.get_গ্রন্থি_shared_ptr());
    প্রত্যাগম গ্রন্থি && গ্রন্থি->get_element_type().is_integral_সংখ্যা() && গ্রন্থি->cast_vector<int64_t>() == মানসমূহ;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using নামাকাশ ov::pass::pattern;
    using নামাকাশ ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::শীর্ষ_K>({softmax, any_প্রবেশ()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_প্রবেশ(), any_প্রবেশ(), any_প্রবেশ(), any_প্রবেশ()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_প্রবেশ()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_প্রবেশ()});
    auto data = wrap_type<v0::Squeeze>({probs, any_প্রবেশ()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_প্রবেশ(), any_প্রবেশ()});
    auto ids_গুনতি = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_প্রবেশ(), any_প্রবেশ()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_গুনতি});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_প্রবেশ()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_প্রবেশ()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_প্রবেশ()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_প্রবেশ()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_প্রবেশ()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    ধ্রুব auto callback = [=](Matcher & m) {
        ধ্রুব auto & pm = m.get_pattern_মান_map();
        ধ্রুব auto গ্রন্থি = [&](ধ্রুব std::shared_ptr<ov::Node> & p) { প্রত্যাগম pm.at(p).get_গ্রন্থি_shared_ptr(); };
        ধ্রুব auto প্রবেশ_is = [&](ধ্রুব std::shared_ptr<ov::Node> & p, size_t i, ধ্রুব std::vector<int64_t> & v) {
            প্রত্যাগম constant_is(গ্রন্থি(p)->প্রবেশ_মান(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(গ্রন্থি(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(গ্রন্থি(softmax));
        auto tk = ov::as_type_ptr<v11::শীর্ষ_K>(গ্রন্থি(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(গ্রন্থি(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(গ্রন্থি(clamp));
        ধ্রুব auto shape = mm->get_output_partial_shape(0);
        যদি (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_প্রবেশ_partial_shape(0).rank() != 4 ||
            mm->get_প্রবেশ_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::শীর্ষ_K::Mode::MAX ||
            tk->get_sort_type() != v11::শীর্ষ_K::SortType::SORT_মানসমূহ || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            প্রত্যাগম অসত্য;
        }
        ধ্রুব int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(গ্রন্থি(ids)->get_প্রবেশ_গ্রন্থি_shared_ptr(2));
        যদি (!end || !end->get_element_type().is_integral_সংখ্যা() || ov::shape_size(end->get_shape()) != 1) {
            প্রত্যাগম অসত্য;
        }
        ধ্রুব int64_t k = end->cast_vector<int64_t>()[0];
        ধ্রুব auto sorted_shape = tk->get_output_partial_shape(1);
        যদি (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !প্রবেশ_is(probs, 1, {1, -1, experts, 1}) ||
            !প্রবেশ_is(data, 1, {0}) || !প্রবেশ_is(ids_2d, 1, {0, 1}) ||
            !প্রবেশ_is(weights_4d, 1, {0}) || !প্রবেশ_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(গ্রন্থি(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(গ্রন্থি(weights))->get_special_zero() ||
            !প্রবেশ_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(গ্রন্থি(gather))->get_batch_dims() != 1 ||
            !প্রবেশ_is(data_batch, 1, {0}) || !প্রবেশ_is(data_batch, 2, {0}) ||
            !প্রবেশ_is(ids_গুনতি, 1, {1}) || !প্রবেশ_is(ids_গুনতি, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(গ্রন্থি(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(গ্রন্থি(ids_গুনতি))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(গ্রন্থি(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(গ্রন্থি(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!প্রবেশ_is(sum, 1, {-1}) && !প্রবেশ_is(sum, 1, {3})) ||
            !প্রবেশ_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(গ্রন্থি(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            প্রত্যাগম অসত্য;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        যদি (!(limit->get_min() <= 0.5 * দ্বিগুণ(k) / experts && limit->get_max() >= 2.0)) {
            প্রত্যাগম অসত্য;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        প্রতি (ধ্রুব auto & প্রবেশ : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(প্রবেশ.get_গ্রন্থি()->shared_from_this());
            যদি (!slice || প্রবেশ.get_index() != 0 || slice->get_প্রবেশ_size() != 5 ||
                !constant_is(slice->প্রবেশ_মান(1), {0}) || !constant_is(slice->প্রবেশ_মান(2), {k}) ||
                !constant_is(slice->প্রবেশ_মান(3), {1}) ||
                (!constant_is(slice->প্রবেশ_মান(4), {3}) && !constant_is(slice->প্রবেশ_মান(4), {-1}))) {
                প্রত্যাগম অসত্য;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->প্রবেশ_মান(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->প্রবেশ_মান(1), অসত্য, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::শীর্ষ_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::শীর্ষ_K::Mode::MAX, v11::শীর্ষ_K::SortType::SORT_মানসমূহ, tk->get_index_element_type(), অসত্য);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), সত্য);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, গ্রন্থি(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_গ্রন্থি(গ্রন্থি(norm), weights_out);
        প্রতি (ধ্রুব auto & slice : slices) {
            ov::replace_গ্রন্থি(slice, ids_out);
        }
        প্রত্যাগম সত্য;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
