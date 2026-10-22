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
    ov::Output<ov::Node> وزن;
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

    خودڪار ڳنڍ = b.get_ڳنڍ_shared_ptr();
    طالما (ov::is_type<ov::op::v0::Convert>(ڳنڍ) || ov::is_type<ov::op::v1::Reshape>(ڳنڍ)) {
        ڳنڍ = ڳنڍ->get_داخلا_ڳنڍ_shared_ptr(0);
    }

    خودڪار mul = ov::as_type_ptr<ov::op::v1::Multiply>(ڳنڍ);
    إذا (!mul) {
        أرجع res;
    }
    res.scale = mul->داخلا_قدر(1);

    خودڪار lhs = mul->get_داخلا_ڳنڍ_shared_ptr(0);
    إذا (خودڪار sub = ov::as_type_ptr<ov::op::v1::ڪٽيو>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        خودڪار zp_ڳنڍ = sub->get_داخلا_ڳنڍ_shared_ptr(1);
        طالما (ov::is_type<ov::op::v0::Convert>(zp_ڳنڍ)) {
            zp_ڳنڍ = zp_ڳنڍ->get_داخلا_ڳنڍ_shared_ptr(0);
        }
        res.zp = zp_ڳنڍ->output(0);
        res.has_zp = سچ;
        lhs = sub->get_داخلا_ڳنڍ_shared_ptr(0);
    }
    طالما (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_داخلا_ڳنڍ_shared_ptr(0);
    }
    إذا (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        أرجع res;
    }

    res.وزن = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.وزن.get_partial_shape().is_static();
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
ov::Output<ov::Node> ٽڪرو_experts_dim1(ثابت ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    خودڪار constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_ڳنڍ_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    ثابت خودڪار & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    ثابت خودڪار & type = constant->get_element_type();

    size_t صف_elems = 1;
    لكل (size_t i = 2; i < shape.size(); ++i) {
        صف_elems *= shape[i];
    }
    ثابت size_t صف_بٽ = صف_elems * type.bitwidth();
    OPENVINO_ASSERT(صف_بٽ % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    ثابت size_t صف_بائيٽون = صف_بٽ / 8;

    ov::Shape ٻاهريون_shape = shape;
    ٻاهريون_shape[1] = static_cast<size_t>(end - begin);

    std::vector<uint8_t> ٻاهريون_معلومات(shape[0] * static_cast<size_t>(end - begin) * صف_بائيٽون);
    ثابت خودڪار * src = static_cast<ثابت uint8_t *>(constant->get_معلومات_ptr());
    ثابت size_t src_صف_بائيٽون = shape[1] * صف_بائيٽون;
    لكل (size_t e = 0; e < shape[0]; ++e) {
        std::يادگيري_نقل(ٻاهريون_معلومات.معلومات() + e * static_cast<size_t>(end - begin) * صف_بائيٽون,
                   src + e * src_صف_بائيٽون + static_cast<size_t>(begin) * صف_بائيٽون,
                   static_cast<size_t>(end - begin) * صف_بائيٽون);
    }

    أرجع std::make_shared<ov::op::v0::Constant>(type, ٻاهريون_shape, ٻاهريون_معلومات.معلومات());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    استعمال نطاق ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    خودڪار hidden_gate_m = any_داخلا();
    خودڪار a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_داخلا() });
    خودڪار a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_داخلا() });
    خودڪار hidden_مٿي_m = any_داخلا();
    خودڪار a_مٿي_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_مٿي_m, any_داخلا() }) }),
          any_داخلا() });

    خودڪار gate_w_m = any_داخلا();
    خودڪار مٿي_w_m = any_داخلا();
    خودڪار هيٺ_ڏانهن_w_m = any_داخلا();
    خودڪار ids_gate_m = any_داخلا();
    خودڪار ids_مٿي_m = any_داخلا();
    خودڪار ids_هيٺ_ڏانهن_m = any_داخلا();

    خودڪار bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_داخلا() });
    خودڪار gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_داخلا() }), any_داخلا() }) });

    خودڪار silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    خودڪار bgm_مٿي_m = wrap_type<ov::op::internal::GatherMatmul>({ a_مٿي_m, مٿي_w_m, ids_مٿي_m, any_داخلا() });
    خودڪار مٿي_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_مٿي_m, any_داخلا() }), any_داخلا() }) });
    خودڪار swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, مٿي_u_m });

    خودڪار d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_داخلا() }) }),
          any_داخلا() });
    خودڪار bgm_هيٺ_ڏانهن_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, هيٺ_ڏانهن_w_m, ids_هيٺ_ڏانهن_m, any_داخلا() });
    خودڪار هيٺ_ڏانهن_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_هيٺ_ڏانهن_m, any_داخلا() }), any_داخلا() }) });

    خودڪار routing_m = any_داخلا();
    خودڪار weighted_m = wrap_type<ov::op::v1::Multiply>({ هيٺ_ڏانهن_u_m, routing_m });
    خودڪار پاڙ_m = wrap_type<ov::op::v1::Reduce_جوڙ>({ weighted_m, any_داخلا() });

    ثابت خودڪار callback = [=](Matcher & m) {
        خودڪار & pm = m.get_نمونو_قدر_map();

        ثابت خودڪار gate = unwrap_dequant(pm.at(gate_w_m));
        ثابت خودڪار مٿي = unwrap_dequant(pm.at(مٿي_w_m));
        ثابت خودڪار هيٺ_ڏانهن = unwrap_dequant(pm.at(هيٺ_ڏانهن_w_m));
        إذا (!gate.ok || !مٿي.ok || !هيٺ_ڏانهن.ok) {
            أرجع خطأ;
        }

        ثابت خودڪار gate_shape = gate.وزن.get_shape();
        ثابت خودڪار مٿي_shape = مٿي.وزن.get_shape();
        ثابت خودڪار هيٺ_ڏانهن_shape = هيٺ_ڏانهن.وزن.get_shape();
        إذا (gate_shape != مٿي_shape || gate_shape.size() < 3 || هيٺ_ڏانهن_shape.size() < 3) {
            أرجع خطأ;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        إذا (gate.has_zp != هيٺ_ڏانهن.has_zp || gate_shape.size() != هيٺ_ڏانهن_shape.size()) {
            أرجع خطأ;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        إذا (gate.has_zp) {
            ساكن ثابت std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            إذا (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(هيٺ_ڏانهن.zp.get_element_type()) == 0) {
                أرجع خطأ;
            }
        }

        // Config holds a single group_size for all three projections.
        ثابت خودڪار گروهه_جو = [](ثابت dequant_inputs & w) {
            ثابت خودڪار s = w.وزن.get_shape();
            أرجع s.size() == 4 ? s[3] : logical_k(s);
        };
        إذا (گروهه_جو(gate) != گروهه_جو(مٿي) || گروهه_جو(gate) != گروهه_جو(هيٺ_ڏانهن)) {
            أرجع خطأ;
        }

        // all three branches must route the same hidden state through the same experts
        إذا (pm.at(hidden_gate_m) != pm.at(hidden_مٿي_m)) {
            أرجع خطأ;
        }

        خودڪار ids = pm.at(ids_هيٺ_ڏانهن_m);
        ثابت خودڪار ids_pshape = ids.get_partial_shape();
        إذا (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_ڊيگهائي() - 1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t چوٽي_k = ids_pshape[ids_pshape.rank().get_ڊيگهائي() - 1].get_ڊيگهائي();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        خودڪار routing = pm.at(routing_m);
        ثابت خودڪار routing_pshape = routing.get_partial_shape();
        إذا (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_ڊيگهائي() != 4 ||
            routing_pshape[3] != 1) {
            أرجع خطأ;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        إذا (ids_pshape.rank().get_ڊيگهائي() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        إذا (routing.get_partial_shape() != ids.get_partial_shape()) {
            أرجع خطأ;
        }

        ثابت size_t هيٺ_ڏانهن_k = logical_k(هيٺ_ڏانهن_shape);
        ثابت خودڪار هيٺ_ڏانهن_scale_shape = هيٺ_ڏانهن.scale.get_shape();
        ثابت size_t هيٺ_ڏانهن_groups = هيٺ_ڏانهن_scale_shape.size() >= 3 ? هيٺ_ڏانهن_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_سائيز = logical_k(gate_shape);
        config.inter_سائيز = gate_shape[1];
        config.انگ_expert = gate_shape[0];
        config.انگ_shared_expert = 0;
        config.چوٽي_k = چوٽي_k;
        config.گروهه_سائيز = هيٺ_ڏانهن_groups <= 1 ? std::numeric_limits<size_t>::max() : هيٺ_ڏانهن_k / هيٺ_ڏانهن_groups;
        config.has_batch_dim = سچ;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.ٻاهريون_type = ov::element::dynamic;

        خودڪار absent_zp = [] {
            خودڪار zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            أرجع zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        ثابت خودڪار a_transpose = pm.at(a_gate_m).get_ڳنڍ_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->داخلا_قدر(1));

        ثابت ov::Output_ويڪٽر دليلون_2 = {
            hidden, routing, ids,
            gate.وزن,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            مٿي.وزن,    مٿي.scale,     مٿي.has_zp ? مٿي.zp : absent_zp(),
            هيٺ_ڏانهن.وزن,  هيٺ_ڏانهن.scale,   هيٺ_ڏانهن.has_zp ? هيٺ_ڏانهن.zp : absent_zp(),
        };

        خودڪار moe = std::make_shared<ov::op::internal::MOECompressed>(دليلون_2, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> نتيجو = moe->output(0);
        ثابت خودڪار پاڙ_type = m.get_ميلاپ_پاڙ()->get_output_element_type(0);
        إذا (نتيجو.get_element_type() != پاڙ_type) {
            نتيجو = std::make_shared<ov::op::v0::Convert>(نتيجو, پاڙ_type);
        }

        نتيجو.get_ڳنڍ_shared_ptr()->سيٽ_friendly_name(m.get_ميلاپ_پاڙ()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), نتيجو.get_ڳنڍ_shared_ptr());
        ov::replace_ڳنڍ(m.get_ميلاپ_پاڙ(), نتيجو.get_ڳنڍ_shared_ptr());
        رجسٽر_new_ڳنڍ(moe);
        أرجع سچ;
    };

    رجسٽر_matcher(std::make_shared<Matcher>(پاڙ_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

Fuse_Moe_Compressed_Fused_Gate_مٿي::Fuse_Moe_Compressed_Fused_Gate_مٿي() {
    استعمال نطاق ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    خودڪار hidden_m = any_داخلا();
    خودڪار a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_داخلا() });
    خودڪار a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_داخلا() });

    خودڪار gate_مٿي_w_m = any_داخلا();
    خودڪار ids_gate_مٿي_m = any_داخلا();
    خودڪار bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_مٿي_w_m, ids_gate_مٿي_m, any_داخلا() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    خودڪار gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_داخلا() });
    خودڪار gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_داخلا() }) });

    خودڪار gate_ٽڪرو_m = wrap_type<ov::op::v8::ٽڪرو>({ gu_u_m, any_داخلا(), any_داخلا(), any_داخلا(), any_داخلا() });
    خودڪار مٿي_ٽڪرو_m = wrap_type<ov::op::v8::ٽڪرو>({ gu_u_m, any_داخلا(), any_داخلا(), any_داخلا(), any_داخلا() });
    خودڪار gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_ٽڪرو_m });
    خودڪار geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, مٿي_ٽڪرو_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    خودڪار d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_داخلا() }), any_داخلا() }) }),
          any_داخلا() });
    خودڪار هيٺ_ڏانهن_w_m = any_داخلا();
    خودڪار ids_هيٺ_ڏانهن_m = any_داخلا();
    خودڪار bgm_هيٺ_ڏانهن_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, هيٺ_ڏانهن_w_m, ids_هيٺ_ڏانهن_m, any_داخلا() });
    // Same two shapes as the gate/up branch above.
    خودڪار هيٺ_ڏانهن_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_هيٺ_ڏانهن_m }), any_داخلا() });
    خودڪار هيٺ_ڏانهن_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ هيٺ_ڏانهن_t_m, any_داخلا() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    خودڪار هيٺ_ڏانهن_scale_m = any_داخلا();
    خودڪار هيٺ_ڏانهن_scaled_m = wrap_type<ov::op::v1::Multiply>({ هيٺ_ڏانهن_u_m, هيٺ_ڏانهن_scale_m });

    خودڪار routing_m = any_داخلا();
    خودڪار weighted_m = wrap_type<ov::op::v1::Multiply>({ هيٺ_ڏانهن_scaled_m, routing_m });
    خودڪار پاڙ_m = wrap_type<ov::op::v1::Reduce_جوڙ>({ weighted_m, any_داخلا() });

    ثابت خودڪار callback = [=](Matcher & m) {
        خودڪار & pm = m.get_نمونو_قدر_map();

        ثابت خودڪار gate_مٿي = unwrap_dequant(pm.at(gate_مٿي_w_m));
        ثابت خودڪار هيٺ_ڏانهن = unwrap_dequant(pm.at(هيٺ_ڏانهن_w_m));
        إذا (!gate_مٿي.ok || !هيٺ_ڏانهن.ok) {
            أرجع خطأ;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        إذا (!gate_مٿي.has_zp || !هيٺ_ڏانهن.has_zp) {
            أرجع خطأ;
        }

        ثابت خودڪار gate_مٿي_shape = gate_مٿي.وزن.get_shape();
        ثابت خودڪار هيٺ_ڏانهن_shape = هيٺ_ڏانهن.وزن.get_shape();
        إذا (gate_مٿي_shape.size() < 3 || هيٺ_ڏانهن_shape.size() < 3 || gate_مٿي_shape.size() != هيٺ_ڏانهن_shape.size()) {
            أرجع خطأ;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        ساكن ثابت std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        إذا (int_types.count(gate_مٿي.وزن.get_element_type()) == 0 ||
            int_types.count(هيٺ_ڏانهن.وزن.get_element_type()) == 0 ||
            int_types.count(gate_مٿي.zp.get_element_type()) == 0 ||
            int_types.count(هيٺ_ڏانهن.zp.get_element_type()) == 0) {
            أرجع خطأ;
        }

        ثابت خودڪار گروهه_جو = [](ثابت dequant_inputs & w) {
            ثابت خودڪار s = w.وزن.get_shape();
            أرجع s.size() == 4 ? s[3] : logical_k(s);
        };
        إذا (گروهه_جو(gate_مٿي) != گروهه_جو(هيٺ_ڏانهن)) {
            أرجع خطأ;
        }

        خودڪار gelu_ڳنڍ = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_ڳنڍ_shared_ptr());
        إذا (!gelu_ڳنڍ || gelu_ڳنڍ->get_approximation_گهڻائي() != ov::op::Gelu_Approximation_گهڻائي::ERF) {
            أرجع خطأ;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        خودڪار gate_ٽڪرو = ov::as_type_ptr<ov::op::v8::ٽڪرو>(pm.at(gate_ٽڪرو_m).get_ڳنڍ_shared_ptr());
        خودڪار مٿي_ٽڪرو = ov::as_type_ptr<ov::op::v8::ٽڪرو>(pm.at(مٿي_ٽڪرو_m).get_ڳنڍ_shared_ptr());
        خودڪار gate_پڇاڙي_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_ٽڪرو->داخلا_قدر(2).get_ڳنڍ_shared_ptr());
        خودڪار مٿي_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(مٿي_ٽڪرو->داخلا_قدر(1).get_ڳنڍ_shared_ptr());
        إذا (!gate_پڇاڙي_c || !مٿي_begin_c) {
            أرجع خطأ;
        }
        ثابت خودڪار gate_پڇاڙي = gate_پڇاڙي_c->cast_ويڪٽر<int64_t>().at(0);
        ثابت خودڪار مٿي_begin = مٿي_begin_c->cast_ويڪٽر<int64_t>().at(0);
        إذا (gate_پڇاڙي != مٿي_begin || gate_پڇاڙي <= 0 || gate_پڇاڙي >= static_cast<int64_t>(gate_مٿي_shape[1])) {
            أرجع خطأ;
        }
        ثابت int64_t mid = gate_پڇاڙي;
        ثابت int64_t ڀريل = static_cast<int64_t>(gate_مٿي_shape[1]);

        خودڪار ids = pm.at(ids_هيٺ_ڏانهن_m);
        ثابت خودڪار ids_pshape = ids.get_partial_shape();
        إذا (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_ڊيگهائي() - 1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t چوٽي_k = ids_pshape[ids_pshape.rank().get_ڊيگهائي() - 1].get_ڊيگهائي();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        ثابت خودڪار routing_اندريون = pm.at(routing_m);
        ثابت خودڪار routing_pshape = routing_اندريون.get_partial_shape();
        ثابت خودڪار routing_rank = routing_pshape.rank();
        إذا (routing_rank.is_dynamic() || (routing_rank.get_ڊيگهائي() != 4 && routing_rank.get_ڊيگهائي() != 3) ||
            routing_pshape[routing_rank.get_ڊيگهائي() - 1] != 1) {
            أرجع خطأ;
        }
        ثابت منطقي batchless = routing_rank.get_ڊيگهائي() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        خودڪار هيٺ_ڏانهن_scale = pm.at(هيٺ_ڏانهن_scale_m);
        إذا (هيٺ_ڏانهن_scale.get_partial_shape() != routing_pshape) {
            أرجع خطأ;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_اندريون, هيٺ_ڏانهن_scale);
        إذا (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        إذا (ids_pshape.rank().get_ڊيگهائي() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        إذا (routing.get_partial_shape() != ids.get_partial_shape()) {
            أرجع خطأ;
        }

        ثابت size_t هيٺ_ڏانهن_k = logical_k(هيٺ_ڏانهن_shape);
        ثابت خودڪار هيٺ_ڏانهن_scale_shape = هيٺ_ڏانهن.scale.get_shape();
        ثابت size_t هيٺ_ڏانهن_groups = هيٺ_ڏانهن_scale_shape.size() >= 3 ? هيٺ_ڏانهن_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        ثابت ov::Output<ov::Node> gate_وزن = ٽڪرو_experts_dim1(gate_مٿي.وزن, 0, mid);
        ثابت ov::Output<ov::Node> مٿي_وزن = ٽڪرو_experts_dim1(gate_مٿي.وزن, mid, ڀريل);
        ثابت ov::Output<ov::Node> gate_scale = ٽڪرو_experts_dim1(gate_مٿي.scale, 0, mid);
        ثابت ov::Output<ov::Node> مٿي_scale = ٽڪرو_experts_dim1(gate_مٿي.scale, mid, ڀريل);
        ثابت ov::Output<ov::Node> gate_zp = ٽڪرو_experts_dim1(gate_مٿي.zp, 0, mid);
        ثابت ov::Output<ov::Node> مٿي_zp = ٽڪرو_experts_dim1(gate_مٿي.zp, mid, ڀريل);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_سائيز = logical_k(gate_مٿي_shape);
        config.inter_سائيز = static_cast<size_t>(mid);
        config.انگ_expert = gate_مٿي_shape[0];
        config.انگ_shared_expert = 0;
        config.چوٽي_k = چوٽي_k;
        config.گروهه_سائيز = هيٺ_ڏانهن_groups <= 1 ? std::numeric_limits<size_t>::max() : هيٺ_ڏانهن_k / هيٺ_ڏانهن_groups;
        config.has_batch_dim = سچ;
        config.has_zp = سچ;
        config.ٻاهريون_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        ثابت خودڪار a_transpose = pm.at(a_m).get_ڳنڍ_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->داخلا_قدر(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        ثابت ov::Output_ويڪٽر دليلون_2 = {
            hidden,      routing,     ids,
            gate_وزن, gate_scale,  gate_zp,
            مٿي_وزن,   مٿي_scale,    مٿي_zp,
            هيٺ_ڏانهن.وزن, هيٺ_ڏانهن.scale,  هيٺ_ڏانهن.zp,
        };

        خودڪار moe = std::make_shared<ov::op::internal::MOECompressed>(دليلون_2, config);

        ov::Output<ov::Node> نتيجو = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        إذا (batchless) {
            نتيجو = std::make_shared<ov::op::v0::Squeeze>(
                نتيجو, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ثابت خودڪار پاڙ_type = m.get_ميلاپ_پاڙ()->get_output_element_type(0);
        إذا (نتيجو.get_element_type() != پاڙ_type) {
            نتيجو = std::make_shared<ov::op::v0::Convert>(نتيجو, پاڙ_type);
        }
        إذا (نتيجو.get_partial_shape() != m.get_ميلاپ_پاڙ()->get_output_partial_shape(0)) {
            أرجع خطأ;
        }

        نتيجو.get_ڳنڍ_shared_ptr()->سيٽ_friendly_name(m.get_ميلاپ_پاڙ()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), نتيجو.get_ڳنڍ_shared_ptr());
        ov::replace_ڳنڍ(m.get_ميلاپ_پاڙ(), نتيجو.get_ڳنڍ_shared_ptr());
        رجسٽر_new_ڳنڍ(moe);
        أرجع سچ;
    };

    رجسٽر_matcher(
        std::make_shared<Matcher>(پاڙ_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
