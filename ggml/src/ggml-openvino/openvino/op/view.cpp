#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/gather.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_view(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    যদি (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        প্রত্যাগম {context.get_প্রবেশ(0)};
    }

    যদি (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        ধ্রুব std::string & vname = context.get_name();
        যদি (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_প্রবেশ_shape(0);
            auto dst_ps = context.get_output_shape();
            যদি (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                auto sst = context.get_প্রবেশ_stride(0);
                auto dst = context.get_output_stride();
                size_t voff = context.get_output_op_offset();
                auto ss = src_ps.to_shape();
                auto dd = dst_ps.to_shape();
                ধ্রুব size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                ধ্রুব auto in_rank = context.get_প্রবেশ(0).get_partial_shape().rank();
                ধ্রুব পূর্ণাঙ্ক axis_shift =
                    in_rank.is_static() ? (পূর্ণাঙ্ক) nd - (পূর্ণাঙ্ক) in_rank.get_length() : 0;
                তার্কিক shift_ok = axis_shift >= 0 && (size_t) axis_shift < nd;
                প্রতি (পূর্ণাঙ্ক a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                যদি (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<তার্কিক> used(nd, অসত্য);
                    std::vector<পূর্ণাঙ্ক> dst_to_src(nd, -1);
                    তার্কিক ok = সত্য;
                    প্রতি (size_t d = 0; d < nd; ++d) {
                        যদি (dd[d] == 1) {
                            অনুবর্ত;
                        }
                        পূর্ণাঙ্ক found = -1;
                        প্রতি (size_t s = 0; s < nd; ++s) {
                            যদি (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (পূর্ণাঙ্ক) s;
                                ভঙ্গ;
                            }
                        }
                        যদি (found < 0) {
                            ok = অসত্য;
                            ভঙ্গ;
                        }
                        used[found] = সত্য;
                        dst_to_src[d] = found;
                    }
                    পূর্ণাঙ্ক dropped = -1;
                    যদি (ok) {
                        প্রতি (size_t s = 0; s < nd; ++s) {
                            যদি (!used[s] && ss[s] > 1) {
                                যদি (dropped >= 0) {
                                    ok = অসত্য;
                                    ভঙ্গ;
                                }
                                dropped = (পূর্ণাঙ্ক) s;
                            }
                        }
                    }
                    যদি (ok && dropped >= 0) {
                        ধ্রুব size_t dstr = sst[dropped];
                        ধ্রুব int64_t dsz = (int64_t) ss[dropped];
                        যদি (dstr > 0 && voff % dstr == 0) {
                            ধ্রুব int64_t sel = (int64_t) (voff / dstr);
                            যদি (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::Slice>(
                                    context.get_প্রবেশ(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_shift}));
                                // Build the reshape target from the (concrete) dst shape, but
                                // keep the dynamic token axis dynamic instead of freezing it
                                // to the captured n_tokens. Without this the constant dst
                                // shape bakes in the prefill token count and the static value
                                // flows downstream, turning every later decoder layer static
                                // (the GPU in-place-concat KV-cache bug). The token axis is
                                // PERMUTED between the sliced input and the dst (e.g. input
                                // [1,tok,expert,emb] -> dst [1,1,tok,emb]), so special_zero
                                // (which copies the same-position dim) is not enough: pull the
                                // dynamic dim from the correct SOURCE axis via ShapeOf+Gather
                                // and place it at the dst token position.
                                ধ্রুব int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                পূর্ণাঙ্ক dst_ov_axis = (dyn != -1) ? ((পূর্ণাঙ্ক) nd - 1 - (পূর্ণাঙ্ক) dyn) : -1;
                                পূর্ণাঙ্ক src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (পূর্ণাঙ্ক) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                যদি (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    auto sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    auto tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    প্রতি (পূর্ণাঙ্ক a = axis_shift; a < (পূর্ণাঙ্ক) nd; ++a) {
                                        যদি (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } অন্যথা {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    auto dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, অসত্য);
                                    প্রত্যাগম rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                auto dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, অসত্য);
                                প্রত্যাগম rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        প্রত্যাগম {context.get_প্রবেশ(0)};
    }

    auto প্রবেশ = context.get_প্রবেশ(0);
    auto src_shape = context.get_প্রবেশ_shape(0);
    auto dst_shape = context.get_output_shape();

    যদি (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        প্রত্যাগম {প্রবেশ};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    প্রতি (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        যদি (src_shape[i].is_dynamic()) {
            প্রত্যাগম {প্রবেশ};
        }
        src_elems *= src_shape[i].get_length();
    }
    প্রতি (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        যদি (dst_shape[i].is_dynamic()) {
            প্রত্যাগম {প্রবেশ};
        }
        dst_elems *= dst_shape[i].get_length();
    }

    যদি (dst_elems >= src_elems) {
        প্রত্যাগম {প্রবেশ};
    }

    auto src_stride = context.get_প্রবেশ_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    তার্কিক same_stride = (src_stride.size() == dst_stride.size());
    যদি (same_stride) {
        প্রতি (size_t i = 0; i < src_stride.size(); ++i) {
            যদি (src_stride[i] != dst_stride[i]) {
                same_stride = অসত্য;
                ভঙ্গ;
            }
        }
    }

    যদি (!same_stride) {
        প্রত্যাগম {প্রবেশ};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    যদি (dst_ov_shape.size() != ndims) {
        প্রত্যাগম {প্রবেশ};
    }

    std::vector<পূর্ণাঙ্ক> diff_dims;
    প্রতি (size_t i = 0; i < ndims; ++i) {
        যদি (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<পূর্ণাঙ্ক>(i));
        }
    }

    যদি (diff_dims.size() != 1) {
        প্রত্যাগম {প্রবেশ};
    }

    পূর্ণাঙ্ক slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    প্রতি (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    যদি (elem_size == 0) {
        elem_size = 1;
    }

    int64_t begin_val = 0;
    যদি (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    যদি (begin_val < 0 || end_val > dim_size) {
        প্রত্যাগম {প্রবেশ};
    }

    auto sliced =
        std::make_shared<ov::op::v8::Slice>(প্রবেশ, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    প্রত্যাগম {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
