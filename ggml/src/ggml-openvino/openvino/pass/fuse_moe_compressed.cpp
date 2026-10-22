#ضمن "fuse_moe_compressed.h"

#ضمن <cstring>
#ضمن <limits>
#ضمن <set>
#ضمن <memory>
#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/gelu.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/swish.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/constant_folding.hpp>
#ضمن <openvino/pass/pattern/op/optional.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

#ضمن "../op/gather_matmul.hpp"
#ضمن "../op/moe_compressed.hpp"

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

بنية dequant_inputs {
    ov::Output<ov::Node> weight;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zp;
    منطقي has_zp = خطأ;
    منطقي ok = خطأ;
};

// Peel the chain built by make_int4_weights/make_int8_weights back to its Constant inputs.
// Grouped weights keep the pre-Reshape rank-4 form [n_expert, n, k/group, group] with scale
// and zp at [n_expert, n, k/group, 1], which is the layout MOECompressed expects. Channel-wise
// weights stay rank-3 with a rank-3 scale and carry no zp.
dequant_inputs unwrap_dequant(ثابت ov::Output<ov::Node> & b) {
    dequant_inputs res;

    auto گرہ = b.get_گرہ_shared_ptr();
    طالما (ov::is_type<ov::op::v0::Convert>(گرہ) || ov::is_type<ov::op::v1::Reshape>(گرہ)) {
        گرہ = گرہ->get_داخلہ_گرہ_shared_ptr(0);
    }

    auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(گرہ);
    إذا (!mul) {
        أرجع res;
    }
    res.scale = mul->داخلہ_قدر(1);

    auto lhs = mul->get_داخلہ_گرہ_shared_ptr(0);
    إذا (auto sub = ov::as_type_ptr<ov::op::v1::Subtract>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        auto zp_گرہ = sub->get_داخلہ_گرہ_shared_ptr(1);
        طالما (ov::is_type<ov::op::v0::Convert>(zp_گرہ)) {
            zp_گرہ = zp_گرہ->get_داخلہ_گرہ_shared_ptr(0);
        }
        res.zp = zp_گرہ->output(0);
        res.has_zp = true;
        lhs = sub->get_داخلہ_گرہ_shared_ptr(0);
    }
    طالما (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_داخلہ_گرہ_shared_ptr(0);
    }
    إذا (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        أرجع res;
    }

    res.weight = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.weight.get_partial_shape().is_static();
    أرجع res;
}

size_t logical_k(ثابت ov::Shape & shape) {
    أرجع shape.size() == 4 ? shape[2] * shape[3] : shape.back();
}

// Slice a [n_expert, m, ...] compressed weight/scale/zp Constant along axis 1 (m), [begin, end).
// Built by copying raw bytes directly instead of a graph Slice op: CommonOptimizations rewrites
// v8::Slice into v1::StridedSlice for constant folding regardless of disable_constant_folding
// (the rewrite doesn't carry the marking over), and that reference evaluator crashes on
// sub-byte (u4/i4) element types. Each axis-1 "row" (the trailing dims) is confirmed
// byte-aligned here -- weight/scale/zp trailing sizes are always whole groups -- so a
// byte-range memcpy per row is exact for both regular and sub-byte element types.
ov::Output<ov::Node> slice_experts_dim1(ثابت ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_گرہ_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    ثابت auto & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    ثابت auto & type = constant->get_element_type();

    size_t row_elems = 1;
    لكل (size_t i = 2; i < shape.size(); ++i) {
        row_elems *= shape[i];
    }
    ثابت size_t row_bits = row_elems * type.bitwidth();
    OPENVINO_ASSERT(row_bits % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    ثابت size_t row_bytes = row_bits / 8;

    ov::Shape out_shape = shape;
    out_shape[1] = static_cast<size_t>(end - begin);

    std::vector<uint8_t> out_data(shape[0] * static_cast<size_t>(end - begin) * row_bytes);
    ثابت auto * src = static_cast<ثابت uint8_t *>(constant->get_data_ptr());
    ثابت size_t src_row_bytes = shape[1] * row_bytes;
    لكل (size_t e = 0; e < shape[0]; ++e) {
        std::memcpy(out_data.data() + e * static_cast<size_t>(end - begin) * row_bytes,
                   src + e * src_row_bytes + static_cast<size_t>(begin) * row_bytes,
                   static_cast<size_t>(end - begin) * row_bytes);
    }

    أرجع std::make_shared<ov::op::v0::Constant>(type, out_shape, out_data.data());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    using نطاق ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    auto hidden_gate_m = any_داخلہ();
    auto a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_داخلہ() });
    auto a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_داخلہ() });
    auto hidden_up_m = any_داخلہ();
    auto a_up_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_up_m, any_داخلہ() }) }),
          any_داخلہ() });

    auto gate_w_m = any_داخلہ();
    auto up_w_m = any_داخلہ();
    auto down_w_m = any_داخلہ();
    auto ids_gate_m = any_داخلہ();
    auto ids_up_m = any_داخلہ();
    auto ids_down_m = any_داخلہ();

    auto bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_داخلہ() });
    auto gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_داخلہ() }), any_داخلہ() }) });

    auto silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    auto bgm_up_m = wrap_type<ov::op::internal::GatherMatmul>({ a_up_m, up_w_m, ids_up_m, any_داخلہ() });
    auto up_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_up_m, any_داخلہ() }), any_داخلہ() }) });
    auto swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, up_u_m });

    auto d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_داخلہ() }) }),
          any_داخلہ() });
    auto bgm_down_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, down_w_m, ids_down_m, any_داخلہ() });
    auto down_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_down_m, any_داخلہ() }), any_داخلہ() }) });

    auto routing_m = any_داخلہ();
    auto weighted_m = wrap_type<ov::op::v1::Multiply>({ down_u_m, routing_m });
    auto جڑ_m = wrap_type<ov::op::v1::ReduceSum>({ weighted_m, any_داخلہ() });

    ثابت auto callback = [=](Matcher & m) {
        auto & pm = m.get_pattern_قدر_map();

        ثابت auto gate = unwrap_dequant(pm.at(gate_w_m));
        ثابت auto up = unwrap_dequant(pm.at(up_w_m));
        ثابت auto down = unwrap_dequant(pm.at(down_w_m));
        إذا (!gate.ok || !up.ok || !down.ok) {
            أرجع خطأ;
        }

        ثابت auto gate_shape = gate.weight.get_shape();
        ثابت auto up_shape = up.weight.get_shape();
        ثابت auto down_shape = down.weight.get_shape();
        إذا (gate_shape != up_shape || gate_shape.size() < 3 || down_shape.size() < 3) {
            أرجع خطأ;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        إذا (gate.has_zp != down.has_zp || gate_shape.size() != down_shape.size()) {
            أرجع خطأ;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        إذا (gate.has_zp) {
            ساكن ثابت std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            إذا (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(down.zp.get_element_type()) == 0) {
                أرجع خطأ;
            }
        }

        // Config holds a single group_size for all three projections.
        ثابت auto group_of = [](ثابت dequant_inputs & w) {
            ثابت auto s = w.weight.get_shape();
            أرجع s.size() == 4 ? s[3] : logical_k(s);
        };
        إذا (group_of(gate) != group_of(up) || group_of(gate) != group_of(down)) {
            أرجع خطأ;
        }

        // all three branches must route the same hidden state through the same experts
        إذا (pm.at(hidden_gate_m) != pm.at(hidden_up_m)) {
            أرجع خطأ;
        }

        auto ids = pm.at(ids_down_m);
        ثابت auto ids_pshape = ids.get_partial_shape();
        إذا (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_length() - 1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t چوٹی_k = ids_pshape[ids_pshape.rank().get_length() - 1].get_length();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        auto routing = pm.at(routing_m);
        ثابت auto routing_pshape = routing.get_partial_shape();
        إذا (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_length() != 4 ||
            routing_pshape[3] != 1) {
            أرجع خطأ;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        إذا (ids_pshape.rank().get_length() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        إذا (routing.get_partial_shape() != ids.get_partial_shape()) {
            أرجع خطأ;
        }

        ثابت size_t down_k = logical_k(down_shape);
        ثابت auto down_scale_shape = down.scale.get_shape();
        ثابت size_t down_groups = down_scale_shape.size() >= 3 ? down_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_size = logical_k(gate_shape);
        config.inter_size = gate_shape[1];
        config.عدد_expert = gate_shape[0];
        config.عدد_shared_expert = 0;
        config.چوٹی_k = چوٹی_k;
        config.group_size = down_groups <= 1 ? std::numeric_limits<size_t>::max() : down_k / down_groups;
        config.has_batch_dim = true;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.out_type = ov::element::dynamic;

        auto absent_zp = [] {
            auto zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            أرجع zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        ثابت auto a_transpose = pm.at(a_gate_m).get_گرہ_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->داخلہ_قدر(1));

        ثابت ov::OutputVector args = {
            hidden, routing, ids,
            gate.weight,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            up.weight,    up.scale,     up.has_zp ? up.zp : absent_zp(),
            down.weight,  down.scale,   down.has_zp ? down.zp : absent_zp(),
        };

        auto moe = std::make_shared<ov::op::internal::MOECompressed>(args, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> result = moe->output(0);
        ثابت auto جڑ_type = m.get_match_جڑ()->get_output_element_type(0);
        إذا (result.get_element_type() != جڑ_type) {
            result = std::make_shared<ov::op::v0::Convert>(result, جڑ_type);
        }

        result.get_گرہ_shared_ptr()->set_friendly_name(m.get_match_جڑ()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result.get_گرہ_shared_ptr());
        ov::replace_گرہ(m.get_match_جڑ(), result.get_گرہ_shared_ptr());
        register_new_گرہ(moe);
        أرجع true;
    };

    register_matcher(std::make_shared<Matcher>(جڑ_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

FuseMoeCompressedFusedGateUp::FuseMoeCompressedFusedGateUp() {
    using نطاق ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    auto hidden_m = any_داخلہ();
    auto a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_داخلہ() });
    auto a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_داخلہ() });

    auto gate_up_w_m = any_داخلہ();
    auto ids_gate_up_m = any_داخلہ();
    auto bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_up_w_m, ids_gate_up_m, any_داخلہ() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    auto gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_داخلہ() });
    auto gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_داخلہ() }) });

    auto gate_slice_m = wrap_type<ov::op::v8::Slice>({ gu_u_m, any_داخلہ(), any_داخلہ(), any_داخلہ(), any_داخلہ() });
    auto up_slice_m = wrap_type<ov::op::v8::Slice>({ gu_u_m, any_داخلہ(), any_داخلہ(), any_داخلہ(), any_داخلہ() });
    auto gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_slice_m });
    auto geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, up_slice_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    auto d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_داخلہ() }), any_داخلہ() }) }),
          any_داخلہ() });
    auto down_w_m = any_داخلہ();
    auto ids_down_m = any_داخلہ();
    auto bgm_down_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, down_w_m, ids_down_m, any_داخلہ() });
    // Same two shapes as the gate/up branch above.
    auto down_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_down_m }), any_داخلہ() });
    auto down_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ down_t_m, any_داخلہ() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    auto down_scale_m = any_داخلہ();
    auto down_scaled_m = wrap_type<ov::op::v1::Multiply>({ down_u_m, down_scale_m });

    auto routing_m = any_داخلہ();
    auto weighted_m = wrap_type<ov::op::v1::Multiply>({ down_scaled_m, routing_m });
    auto جڑ_m = wrap_type<ov::op::v1::ReduceSum>({ weighted_m, any_داخلہ() });

    ثابت auto callback = [=](Matcher & m) {
        auto & pm = m.get_pattern_قدر_map();

        ثابت auto gate_up = unwrap_dequant(pm.at(gate_up_w_m));
        ثابت auto down = unwrap_dequant(pm.at(down_w_m));
        إذا (!gate_up.ok || !down.ok) {
            أرجع خطأ;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        إذا (!gate_up.has_zp || !down.has_zp) {
            أرجع خطأ;
        }

        ثابت auto gate_up_shape = gate_up.weight.get_shape();
        ثابت auto down_shape = down.weight.get_shape();
        إذا (gate_up_shape.size() < 3 || down_shape.size() < 3 || gate_up_shape.size() != down_shape.size()) {
            أرجع خطأ;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        ساكن ثابت std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        إذا (int_types.count(gate_up.weight.get_element_type()) == 0 ||
            int_types.count(down.weight.get_element_type()) == 0 ||
            int_types.count(gate_up.zp.get_element_type()) == 0 ||
            int_types.count(down.zp.get_element_type()) == 0) {
            أرجع خطأ;
        }

        ثابت auto group_of = [](ثابت dequant_inputs & w) {
            ثابت auto s = w.weight.get_shape();
            أرجع s.size() == 4 ? s[3] : logical_k(s);
        };
        إذا (group_of(gate_up) != group_of(down)) {
            أرجع خطأ;
        }

        auto gelu_گرہ = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_گرہ_shared_ptr());
        إذا (!gelu_گرہ || gelu_گرہ->get_approximation_mode() != ov::op::GeluApproximationMode::ERF) {
            أرجع خطأ;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        auto gate_slice = ov::as_type_ptr<ov::op::v8::Slice>(pm.at(gate_slice_m).get_گرہ_shared_ptr());
        auto up_slice = ov::as_type_ptr<ov::op::v8::Slice>(pm.at(up_slice_m).get_گرہ_shared_ptr());
        auto gate_end_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_slice->داخلہ_قدر(2).get_گرہ_shared_ptr());
        auto up_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(up_slice->داخلہ_قدر(1).get_گرہ_shared_ptr());
        إذا (!gate_end_c || !up_begin_c) {
            أرجع خطأ;
        }
        ثابت auto gate_end = gate_end_c->cast_vector<int64_t>().at(0);
        ثابت auto up_begin = up_begin_c->cast_vector<int64_t>().at(0);
        إذا (gate_end != up_begin || gate_end <= 0 || gate_end >= static_cast<int64_t>(gate_up_shape[1])) {
            أرجع خطأ;
        }
        ثابت int64_t mid = gate_end;
        ثابت int64_t full = static_cast<int64_t>(gate_up_shape[1]);

        auto ids = pm.at(ids_down_m);
        ثابت auto ids_pshape = ids.get_partial_shape();
        إذا (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_length() - 1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t چوٹی_k = ids_pshape[ids_pshape.rank().get_length() - 1].get_length();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        ثابت auto routing_in = pm.at(routing_m);
        ثابت auto routing_pshape = routing_in.get_partial_shape();
        ثابت auto routing_rank = routing_pshape.rank();
        إذا (routing_rank.is_dynamic() || (routing_rank.get_length() != 4 && routing_rank.get_length() != 3) ||
            routing_pshape[routing_rank.get_length() - 1] != 1) {
            أرجع خطأ;
        }
        ثابت منطقي batchless = routing_rank.get_length() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        auto down_scale = pm.at(down_scale_m);
        إذا (down_scale.get_partial_shape() != routing_pshape) {
            أرجع خطأ;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_in, down_scale);
        إذا (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        إذا (ids_pshape.rank().get_length() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        إذا (routing.get_partial_shape() != ids.get_partial_shape()) {
            أرجع خطأ;
        }

        ثابت size_t down_k = logical_k(down_shape);
        ثابت auto down_scale_shape = down.scale.get_shape();
        ثابت size_t down_groups = down_scale_shape.size() >= 3 ? down_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        ثابت ov::Output<ov::Node> gate_weight = slice_experts_dim1(gate_up.weight, 0, mid);
        ثابت ov::Output<ov::Node> up_weight = slice_experts_dim1(gate_up.weight, mid, full);
        ثابت ov::Output<ov::Node> gate_scale = slice_experts_dim1(gate_up.scale, 0, mid);
        ثابت ov::Output<ov::Node> up_scale = slice_experts_dim1(gate_up.scale, mid, full);
        ثابت ov::Output<ov::Node> gate_zp = slice_experts_dim1(gate_up.zp, 0, mid);
        ثابت ov::Output<ov::Node> up_zp = slice_experts_dim1(gate_up.zp, mid, full);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_size = logical_k(gate_up_shape);
        config.inter_size = static_cast<size_t>(mid);
        config.عدد_expert = gate_up_shape[0];
        config.عدد_shared_expert = 0;
        config.چوٹی_k = چوٹی_k;
        config.group_size = down_groups <= 1 ? std::numeric_limits<size_t>::max() : down_k / down_groups;
        config.has_batch_dim = true;
        config.has_zp = true;
        config.out_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        ثابت auto a_transpose = pm.at(a_m).get_گرہ_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->داخلہ_قدر(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        ثابت ov::OutputVector args = {
            hidden,      routing,     ids,
            gate_weight, gate_scale,  gate_zp,
            up_weight,   up_scale,    up_zp,
            down.weight, down.scale,  down.zp,
        };

        auto moe = std::make_shared<ov::op::internal::MOECompressed>(args, config);

        ov::Output<ov::Node> result = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        إذا (batchless) {
            result = std::make_shared<ov::op::v0::Squeeze>(
                result, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ثابت auto جڑ_type = m.get_match_جڑ()->get_output_element_type(0);
        إذا (result.get_element_type() != جڑ_type) {
            result = std::make_shared<ov::op::v0::Convert>(result, جڑ_type);
        }
        إذا (result.get_partial_shape() != m.get_match_جڑ()->get_output_partial_shape(0)) {
            أرجع خطأ;
        }

        result.get_گرہ_shared_ptr()->set_friendly_name(m.get_match_جڑ()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result.get_گرہ_shared_ptr());
        ov::replace_گرہ(m.get_match_جڑ(), result.get_گرہ_shared_ptr());
        register_new_گرہ(moe);
        أرجع true;
    };

    register_matcher(
        std::make_shared<Matcher>(جڑ_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
