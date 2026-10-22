// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#অন্তর্ভাৱ <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd {
সার্ৱজনিক:
    FrontEnd();

    স্থির std::shared_ptr<Model> convert(ধ্রুৱ প্ৰবেশ_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
