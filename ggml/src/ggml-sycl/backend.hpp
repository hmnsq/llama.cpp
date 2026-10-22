//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_BACKEND_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_BACKEND_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "binbcast.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "col2im-1d.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "concat.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "conv.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "conv3d.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "convert.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "count-equal.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "cpy.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "dequantize.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "dmmv.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "element_wise.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fusion.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "gated_delta_net.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "gla.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "im2col.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "mmq.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "mmvq.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "norm.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "outprod.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "pad.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "pad_reflect_1d.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "quantize.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "roll.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "rope.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "set_rows.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ssm_conv.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "softmax.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "topk-moe.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "topk-radix.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "tsembd.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "upscale.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "wkv.hpp"


#ਜੇ_ਅੰਤ  // GGML_SYCL_BACKEND_HPP
