#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/gather.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    अगर (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        वापस {context.get_निवेश(0)};
    }

    अगर (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        स्थिर std::string & vname = context.get_name();
        अगर (vname.find("ffn_moe_weighted") != std::string::npos) {
            स्वतः src_ps = context.get_निवेश_shape(0);
            स्वतः dst_ps = context.get_output_shape();
            अगर (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                स्वतः sst = context.get_निवेश_stride(0);
                स्वतः dst = context.get_output_stride();
                माप_प्रकार voff = context.get_output_op_offset();
                स्वतः ss = src_ps.to_shape();
                स्वतः dd = dst_ps.to_shape();
                स्थिर माप_प्रकार nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                स्थिर स्वतः in_rank = context.get_निवेश(0).get_partial_shape().rank();
                स्थिर पूर्णांक axis_shift =
                    in_rank.is_static() ? (पूर्णांक) nd - (पूर्णांक) in_rank.get_लंबाई() : 0;
                bool shift_ok = axis_shift >= 0 && (माप_प्रकार) axis_shift < nd;
                क्रम (पूर्णांक a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                अगर (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<bool> used(nd, false);
                    std::vector<पूर्णांक> dst_to_src(nd, -1);
                    bool ok = true;
                    क्रम (माप_प्रकार d = 0; d < nd; ++d) {
                        अगर (dd[d] == 1) {
                            जारी;
                        }
                        पूर्णांक found = -1;
                        क्रम (माप_प्रकार s = 0; s < nd; ++s) {
                            अगर (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (पूर्णांक) s;
                                अवरोध;
                            }
                        }
                        अगर (found < 0) {
                            ok = false;
                            अवरोध;
                        }
                        used[found] = true;
                        dst_to_src[d] = found;
                    }
                    पूर्णांक dropped = -1;
                    अगर (ok) {
                        क्रम (माप_प्रकार s = 0; s < nd; ++s) {
                            अगर (!used[s] && ss[s] > 1) {
                                अगर (dropped >= 0) {
                                    ok = false;
                                    अवरोध;
                                }
                                dropped = (पूर्णांक) s;
                            }
                        }
                    }
                    अगर (ok && dropped >= 0) {
                        स्थिर माप_प्रकार dstr = sst[dropped];
                        स्थिर int64_t dsz = (int64_t) ss[dropped];
                        अगर (dstr > 0 && voff % dstr == 0) {
                            स्थिर int64_t sel = (int64_t) (voff / dstr);
                            अगर (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::Slice>(
                                    context.get_निवेश(0),
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
                                स्थिर int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                पूर्णांक dst_ov_axis = (dyn != -1) ? ((पूर्णांक) nd - 1 - (पूर्णांक) dyn) : -1;
                                पूर्णांक src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (पूर्णांक) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                अगर (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    स्वतः sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    स्वतः tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    क्रम (पूर्णांक a = axis_shift; a < (पूर्णांक) nd; ++a) {
                                        अगर (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } अन्यथा {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    स्वतः dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    स्वतः rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, false);
                                    वापस rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                स्वतः dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                स्वतः rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, false);
                                वापस rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        वापस {context.get_निवेश(0)};
    }

    स्वतः निवेश = context.get_निवेश(0);
    स्वतः src_shape = context.get_निवेश_shape(0);
    स्वतः dst_shape = context.get_output_shape();

    अगर (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        वापस {निवेश};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    क्रम (int64_t i = 0; i < src_shape.rank().get_लंबाई(); ++i) {
        अगर (src_shape[i].is_dynamic()) {
            वापस {निवेश};
        }
        src_elems *= src_shape[i].get_लंबाई();
    }
    क्रम (int64_t i = 0; i < dst_shape.rank().get_लंबाई(); ++i) {
        अगर (dst_shape[i].is_dynamic()) {
            वापस {निवेश};
        }
        dst_elems *= dst_shape[i].get_लंबाई();
    }

    अगर (dst_elems >= src_elems) {
        वापस {निवेश};
    }

    स्वतः src_stride = context.get_निवेश_stride(0);
    स्वतः dst_stride = context.get_output_stride();
    माप_प्रकार view_offset = context.get_output_op_offset();

    bool same_stride = (src_stride.size() == dst_stride.size());
    अगर (same_stride) {
        क्रम (माप_प्रकार i = 0; i < src_stride.size(); ++i) {
            अगर (src_stride[i] != dst_stride[i]) {
                same_stride = false;
                अवरोध;
            }
        }
    }

    अगर (!same_stride) {
        वापस {निवेश};
    }

    स्वतः src_ov_shape = src_shape.to_shape();
    स्वतः dst_ov_shape = dst_shape.to_shape();
    माप_प्रकार ndims = src_ov_shape.size();
    अगर (dst_ov_shape.size() != ndims) {
        वापस {निवेश};
    }

    std::vector<पूर्णांक> diff_dims;
    क्रम (माप_प्रकार i = 0; i < ndims; ++i) {
        अगर (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<पूर्णांक>(i));
        }
    }

    अगर (diff_dims.size() != 1) {
        वापस {निवेश};
    }

    पूर्णांक slice_dim = diff_dims[0];
    int64_t dim_आकार = static_cast<int64_t>(src_ov_shape[slice_dim]);

    माप_प्रकार ov_stride_for_dim = 1;
    क्रम (माप_प्रकार i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    माप_प्रकार elem_आकार = src_stride.back();
    अगर (elem_आकार == 0) {
        elem_आकार = 1;
    }

    int64_t begin_val = 0;
    अगर (ov_stride_for_dim > 0 && elem_आकार > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_आकार) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    अगर (begin_val < 0 || end_val > dim_आकार) {
        वापस {निवेश};
    }

    स्वतः sliced =
        std::make_shared<ov::op::v8::Slice>(निवेश, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    वापस {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
