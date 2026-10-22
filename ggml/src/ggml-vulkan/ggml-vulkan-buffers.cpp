#ضمن "ggml-vulkan-common.h"

#ضمن <cstdint>
#ضمن <cstdlib>
#ضمن <mutex>
#ضمن <shared_mutex>

ساكن فراغ * portable_aligned_malloc(size_t align, size_t size) {
#جے_تعریف_شدہ _WIN32
    أرجع _aligned_malloc(size, align);
#وإلا
    فراغ * ptr = خالی_سوچک;
    إذا (posix_memalign(&ptr, align, size) != 0) {
        أرجع خالی_سوچک;
    }
    أرجع ptr;
#جے_انت
}

ساكن فراغ portable_aligned_free(فراغ * ptr) {
#جے_تعریف_شدہ _WIN32
    _aligned_free(ptr);
#وإلا
    std::حرر(ptr);
#جے_انت
}

ggml_backend_بفر_type_i ggml_backend_vk_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_vk_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_vk_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_بفر_type_get_max_ناپ,
    /* .get_alloc_size      = */ ggml_backend_vk_بفر_type_get_alloc_ناپ,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};

ساكن std::vector<uint32_t> ggml_vk_لبھو_حافظہ_properties(ثابت vk::Physical_Device_حافظہ_Properties* mem_props, vk::حافظہ_Requirements* mem_req, vk::حافظہ_Property_Flags flags) {
    std::vector<uint32_t> indices;

    لكل (uint32_t i = 0; i < mem_props->حافظہ_Type_گنتی; ++i) {
        vk::حافظہ_Type حافظہ_type = mem_props->حافظہ_Types[i];
        إذا ((mem_req->حافظہ_Type_بٹ & ((uint64_t)1 << i)) &&
            (flags & حافظہ_type.propertyFlags) == flags &&
            mem_props->حافظہ_Heaps[حافظہ_type.ڈھیر_اشاریہ].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    أرجع indices;
}

ساكن vk_بفر ggml_vk_create_بفر(vk_device& device, size_t size, ثابت std::initializer_list<vk::حافظہ_Property_Flags> & req_flags_list,
                                       فراغ *import_ptr = خالی_سوچک) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    إذا (size > device->max_بفر_ناپ) {
        ارم vk::باہرلا_دا_Device_حافظہ_Error("Requested buffer size exceeds device buffer size limit");
    }

    vk_بفر buf = std::make_shared<vk_بفر_struct>();

    إذا (size == 0) {
        buf->size = 0;
        أرجع buf;
    }

    vk::بفر_Usage_Flags usage_flags = vk::بفر_Usage_Flag_بٹ::e_Storage_بفر | vk::بفر_Usage_Flag_بٹ::eTransferSrc | vk::بفر_Usage_Flag_بٹ::eTransferDst;
    vk::حافظہ_Allocate_Flags mem_flags {};
    إذا (device->بفر_device_پتہ) {
        usage_flags |= vk::بفر_Usage_Flag_بٹ::e_Shader_Device_پتہ;
        mem_flags |= vk::حافظہ_Allocate_Flag_بٹ::e_Device_پتہ;
    }

    vk::بفر_Create_Info بفر_create_info{
        vk::بفر_Create_Flags(),
        size,
        usage_flags,
        vk::Sharing_منوال::eExclusive,
        0,
        خالی_سوچک,
    };

    vk::External_حافظہ_بفر_Create_Info external_حافظہ_bci;
    إذا (import_ptr) {
        external_حافظہ_bci.handleTypes = vk::External_حافظہ_Handle_Type_Flag_بٹ::e_Host_ونڈ_EXT;
        بفر_create_info.سیٹ_PNext(&external_حافظہ_bci);
    }

    buf->بفر = device->device.create_بفر(بفر_create_info);

    vk::حافظہ_Requirements mem_req = device->device.get_بفر_حافظہ_Requirements(buf->بفر);

    vk::Physical_Device_حافظہ_Properties mem_props = device->physical_device.get_حافظہ_Properties();

    ثابت vk::حافظہ_ترجیح_Allocate_Info_EXT mem_ترجیح_info { 1.0f };

    vk::حافظہ_Allocate_Flags_Info mem_flags_info { mem_flags };

    إذا (device->حافظہ_ترجیح) {
        mem_flags_info.سیٹ_PNext(&mem_ترجیح_info);
    }

    إذا (import_ptr) {
        vk::حافظہ_Host_اشارہ_گر_Properties_EXT host_اشارہ_گر_props;
        حاول {
            host_اشارہ_گر_props = device->device.get_حافظہ_Host_اشارہ_گر_Properties_EXT(vk::External_حافظہ_Handle_Type_Flag_بٹ::e_Host_ونڈ_EXT, import_ptr);
        } امسك (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_بفر(buf->بفر);
            أرجع {};
        }
        vk::Physical_Device_حافظہ_Properties mem_props = device->physical_device.get_حافظہ_Properties();

        uint32_t حافظہ_type_idx;
        vk::حافظہ_Property_Flags property_flags = *req_flags_list.begin();
        لكل (حافظہ_type_idx = 0; حافظہ_type_idx < 32; ++حافظہ_type_idx) {
            إذا (!(host_اشارہ_گر_props.حافظہ_Type_بٹ & (1u << حافظہ_type_idx))) {
                استمر;
            }
            إذا (!(mem_req.حافظہ_Type_بٹ & (1u << حافظہ_type_idx))) {
                استمر;
            }

            vk::حافظہ_Type حافظہ_type = mem_props.حافظہ_Types[حافظہ_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            إذا ((حافظہ_type.propertyFlags & property_flags) == property_flags) {
                property_flags = حافظہ_type.propertyFlags;
                اكسر;
            }
        }
        إذا (حافظہ_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_بفر(buf->بفر);
            أرجع {};
        }

        buf->حافظہ_property_flags = mem_props.حافظہ_Types[حافظہ_type_idx].propertyFlags;
        حاول {
            vk::Import_حافظہ_Host_اشارہ_گر_Info_EXT import_info;
            import_info.handleType = vk::External_حافظہ_Handle_Type_Flag_بٹ::e_Host_ونڈ_EXT;
            import_info.p_Host_اشارہ_گر = import_ptr;
            import_info.سیٹ_PNext(&mem_flags_info);
            buf->device_حافظہ = device->device.allocate_حافظہ({ size, حافظہ_type_idx, &import_info });
        } امسك (ثابت vk::SystemError& e) {
        }
    } وإلا {
        لكل (خودکار it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            ثابت خودکار & req_flags = *it;

            ثابت std::vector<uint32_t> حافظہ_type_indices = ggml_vk_لبھو_حافظہ_properties(&mem_props, &mem_req, req_flags);

            إذا (حافظہ_type_indices.خالی_تھاواں()) {
                استمر;
            }

            منطقي پورا = خطأ;

            لكل (خودکار mtype_it = حافظہ_type_indices.begin(); mtype_it != حافظہ_type_indices.end(); mtype_it++) {
                حاول {
                    buf->device_حافظہ = device->device.allocate_حافظہ({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->حافظہ_property_flags = mem_props.حافظہ_Types[*mtype_it].propertyFlags;
                    پورا = سچ;
                    اكسر;
                } امسك (ثابت vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    إذا (it + 1 == req_flags_list.end() && mtype_it + 1 == حافظہ_type_indices.end()) {
                        device->device.destroy_بفر(buf->بفر);
                        ارم e;
                    }
                }
            }

            إذا (پورا) {
                اكسر;
            }
        }
    }

    إذا (!buf->device_حافظہ) {
        device->device.destroy_بفر(buf->بفر);
        ارم vk::باہرلا_دا_Device_حافظہ_Error("No suitable memory type found");
    }

    buf->ptr = خالی_سوچک;

    إذا (import_ptr) {
        buf->ptr = import_ptr;
    } وإلا {
        إذا (buf->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible) {
            buf->ptr = device->device.map_حافظہ(buf->device_حافظہ, 0, VK_سارا_ناپ);
        }
    }

    device->device.bind_بفر_حافظہ(buf->بفر, buf->device_حافظہ, 0);

    buf->device = device;
    buf->size = size;

    إذا (device->بفر_device_پتہ) {
        ثابت vk::بفر_Device_پتہ_Info پتہ_Info(buf->بفر);
        buf->bda_addr = device->device.get_بفر_پتہ(پتہ_Info);
    }

    device->حافظہ_logger->log_ونڈ(buf, size);

    أرجع buf;
}

vk_بفر ggml_vk_create_بفر_check(vk_device& device, size_t size, vk::حافظہ_Property_Flags req_flags, vk::حافظہ_Property_Flags fallback_flags) {
    حاول {
        أرجع ggml_vk_create_بفر(device, size, {req_flags, fallback_flags});
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::سطر_انت;
        std::cerr << "ggml_vulkan: " << e.what() << std::سطر_انت;
        ارم e;
    }
}

vk_بفر ggml_vk_create_بفر_device(vk_device& device, size_t size) {
    vk_بفر buf;
    حاول {
        إذا (device->prefer_host_حافظہ) {
            buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent,
                                                       vk::حافظہ_Property_Flag_بٹ::eDeviceLocal});
        } وإلا إذا (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eDeviceLocal | vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent,
                                                       vk::حافظہ_Property_Flag_بٹ::eDeviceLocal,
                                                       vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent});
        } وإلا إذا (device->disable_host_visible_vidmem) {
            إذا (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eDeviceLocal,
                                                           vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent});
            } وإلا {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eDeviceLocal});
            }
        } وإلا {
            // use rebar if available, otherwise fallback to device only visible memory
            إذا (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eDeviceLocal | vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent,
                                                           vk::حافظہ_Property_Flag_بٹ::eDeviceLocal,
                                                           vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent});
            } وإلا {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eDeviceLocal | vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent,
                                                           vk::حافظہ_Property_Flag_بٹ::eDeviceLocal});
            }
        }
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::سطر_انت;
        std::cerr << "ggml_vulkan: " << e.what() << std::سطر_انت;
        ارم e;
    }

    أرجع buf;
}

فراغ ggml_vk_destroy_بفر(vk_بفر& buf) {
    إذا (buf == خالی_سوچک) {
        أرجع;
    }

    إذا (buf->device != خالی_سوچک) {
        buf->device->حافظہ_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
ساكن size_t ggml_vk_host_ونڈ_لبھو(ثابت فراغ * ptr) {
    لكل (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        ثابت uint8_t * ادھار = (ثابت uint8_t *) vk_instance.host_allocations[i].ادھار;
        إذا (ptr >= ادھار && ptr < ادھار + vk_instance.host_allocations[i].size) {
            أرجع i;
        }
    }
    أرجع SIZE_MAX;
}

فراغ * ggml_vk_host_malloc(size_t size) {
    VK_LOG_حافظہ("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    إذا (vk_instance.multi_device) {
        ثابت size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        ثابت size_t alloc_ناپ = (size + align - 1) / align * align;

        فراغ * ptr = portable_aligned_malloc(align, alloc_ناپ);
        إذا (ptr == خالی_سوچک) {
            أرجع خالی_سوچک;
        }
        vk_host_ونڈ اندراج;
        اندراج.ادھار = ptr;
        اندراج.size = alloc_ناپ;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(اندراج));
        أرجع ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_بفر buf;
    حاول {
        buf = ggml_vk_create_بفر(dev0, size,
            {vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent | vk::حافظہ_Property_Flag_بٹ::eHostCached,
             vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent});
    } امسك (vk::SystemError&) {
        أرجع خالی_سوچک;
    }

    إذا (!(buf->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_بفر(buf);
        أرجع خالی_سوچک;
    }

    vk_host_ونڈ اندراج;
    اندراج.ادھار             = buf->ptr;
    اندراج.size             = size;
    اندراج.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(اندراج));
    أرجع buf->ptr;
}

فراغ ggml_vk_host_free(فراغ * ptr) {
    إذا (ptr == خالی_سوچک) {
        أرجع;
    }
    VK_LOG_حافظہ("ggml_vk_host_free(" << ptr << ")");

    vk_host_ونڈ اندراج;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        ثابت size_t اشاریہ = ggml_vk_host_ونڈ_لبھو(ptr);
        إذا (اشاریہ == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            أرجع;
        }
        اندراج = std::move(vk_instance.host_allocations[اشاریہ]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + اشاریہ);
    }

    لكل (خودکار & view : اندراج.views) {
        ggml_vk_destroy_بفر(view);
    }
    إذا (vk_instance.multi_device) {
        portable_aligned_free(اندراج.ادھار);
    }
}

فراغ ggml_vk_host_get(ثابت vk_device& device, ثابت فراغ * ptr, vk_بفر& buf, size_t& buf_وتھ) {
    buf = خالی_سوچک;
    buf_وتھ = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    ثابت size_t اشاریہ = ggml_vk_host_ونڈ_لبھو(ptr);
    إذا (اشاریہ == SIZE_MAX) {
        أرجع;
    }
    vk_host_ونڈ & اندراج = vk_instance.host_allocations[اشاریہ];

    إذا (اندراج.views[device->idx] == خالی_سوچک) {
        ثابت uint32_t ثنائی_ہندسہ = 1u << device->idx;
        إذا (!vk_instance.multi_device || !vk_instance.device_host_بفر_import[device->idx] || (vk_instance.host_import_failed & ثنائی_ہندسہ)) {
            أرجع;
        }
        GGML_ASSERT(device->external_حافظہ_host);
        // const only propagates from the const context; the pointee is mutable
        اندراج.views[device->idx] = ggml_vk_بفر_from_host_ptr(const_cast<vk_device&>(device), اندراج.ادھار, اندراج.size);
        إذا (اندراج.views[device->idx] == خالی_سوچک) {
            vk_instance.host_import_failed |= ثنائی_ہندسہ;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            أرجع;
        }
    }

    buf = اندراج.views[device->idx];
    buf_وتھ = (ثابت uint8_t *) ptr - (ثابت uint8_t *) اندراج.ادھار;
}

فراغ ggml_vk_ensure_sync_staging_بفر(vk_device& device, size_t size) {
    إذا (device->sync_staging == خالی_سوچک || device->sync_staging->size < size) {
        VK_LOG_حافظہ("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_بفر(device->sync_staging);
        device->sync_staging = ggml_vk_create_بفر_check(device, size,
            vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent | vk::حافظہ_Property_Flag_بٹ::eHostCached,
            vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent);
    }
}

فراغ ggml_vk_ensure_sync_staging_بفر(ggml_backend_vk_context * ctx, size_t size) {
    إذا (ctx->sync_staging == خالی_سوچک || ctx->sync_staging->size < size) {
        VK_LOG_حافظہ("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_بفر(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_بفر_check(ctx->device, size,
            vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent | vk::حافظہ_Property_Flag_بٹ::eHostCached,
            vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent);
    }
}

ساكن فراغ ggml_vk_بفر_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_بفر& dst, size_t وتھ, ثابت ggml_tensor * tensor, منطقي sync_staging = خطأ) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    إذا(dst->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::سطر_انت;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_بفر buf = خالی_سوچک;
    size_t buf_وتھ = 0;
    ggml_vk_host_get(ctx->device, tensor->انکڑے, buf, buf_وتھ);

    ثابت uint64_t ne0 = tensor->ne[0];
    ثابت uint64_t ne1 = tensor->ne[1];
    ثابت uint64_t ne2 = tensor->ne[2];
    ثابت uint64_t ne3 = tensor->ne[3];
    ثابت uint64_t nb0 = tensor->nb[0];
    ثابت uint64_t nb1 = tensor->nb[1];
    ثابت uint64_t nb2 = tensor->nb[2];
    ثابت uint64_t nb3 = tensor->nb[3];
    ثابت ggml_type type = tensor->type;
    ثابت uint64_t ts = ggml_type_ناپ(type);
    ثابت uint64_t bs = ggml_blck_ناپ(type);

    ثابت uint64_t dstnb0 = ts;
    ثابت uint64_t dstnb1 = dstnb0*(ne0/bs);
    ثابت uint64_t dstnb2 = dstnb1*ne1;
    ثابت uint64_t dstnb3 = dstnb2*ne2;

    ثابت uint64_t ne = ggml_nelements(tensor);

    إذا (buf != خالی_سوچک) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::بفر_Copy> slices;

        لكل (uint64_t i3 = 0; i3 < ne3; i3++) {
            لكل (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                إذا (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_وتھ + i3*nb3 + i2*nb2, وتھ + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } وإلا {
                    لكل (uint64_t i1 = 0; i1 < ne1; i1++) {
                        إذا (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_وتھ + i3*nb3 + i2*nb2 + i1*nb1, وتھ + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } وإلا {
                            ثابت uint64_t s_off = buf_وتھ + i3*nb3 + i2*nb2 + i1*nb1;
                            ثابت uint64_t d_off = وتھ + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            لكل (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->بفر->buf.copy_بفر(buf->بفر, dst->بفر, slices);
        أرجع;
    }

    إذا (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_بفر& staging = ctx->device->sync_staging;
    ثابت uint64_t copy_ناپ = ts*ne/bs;
    ggml_vk_ensure_sync_staging_بفر(ctx->device, copy_ناپ);
    Vk_بفر_Copy buf_copy{ 0, وتھ, copy_ناپ };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_بفر(subctx->s->بفر->buf, (Vk_بفر)staging->بفر, (Vk_بفر)dst->بفر, 1, &buf_copy);

    لكل (uint64_t i3 = 0; i3 < ne3; i3++) {
        لكل (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            إذا (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (ثابت uint8_t *) tensor->انکڑے + buf_وتھ + i3*nb3 + i2*nb2, dstnb2, &subctx->وچ_memcpys);
            } وإلا {
                لكل (uint64_t i1 = 0; i1 < ne1; i1++) {
                    إذا (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (ثابت uint8_t *) tensor->انکڑے + buf_وتھ + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->وچ_memcpys);
                    } وإلا {
                        ثابت uint64_t s_off = buf_وتھ + i3*nb3 + i2*nb2 + i1*nb1;
                        ثابت uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        لكل (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (ثابت uint8_t *) tensor->انکڑے + s_off + i0*nb0, dstnb0, &subctx->وچ_memcpys);
                        }
                    }
                }
            }
        }
    }
}

منطقي ggml_vk_بفر_write_2d_async(vk_context subctx, vk_بفر& dst, size_t وتھ, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اچائی, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << چوڑائی << ", " << اچائی << ")");
    // Check if src is pinned memory
    vk_بفر buf = خالی_سوچک;
    size_t buf_وتھ = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_وتھ);

    إذا (buf != خالی_سوچک) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::بفر_Copy> slices(1);
        إذا (چوڑائی == spitch && چوڑائی == dpitch) {
            // Only do single write if stride is equal
            slices[0].src_وتھ_2 = buf_وتھ;
            slices[0].dst_وتھ_2 = وتھ;
            slices[0].size = چوڑائی * اچائی;
        } وإلا {
            slices.resize(اچائی);
            لكل (size_t i = 0; i < اچائی; i++) {
                slices[i].src_وتھ_2 = buf_وتھ + i * spitch;
                slices[i].dst_وتھ_2 = وتھ + i * dpitch;
                slices[i].size = چوڑائی;
            }
        }

        ggml_vk_sync_buffers(خالی_سوچک, subctx);
        subctx->s->بفر->buf.copy_بفر(buf->بفر, dst->بفر, slices);
        أرجع سچ;
    }
    VK_LOG_DEBUG("STAGING");

    إذا (!sync_staging) {
        // copy was not handled caller needs to fall back
        أرجع خطأ;
    }

    // Staging buffer required
    ثابت size_t staging_ناپ = چوڑائی * اچائی;
    ggml_vk_ensure_sync_staging_بفر(dst->device, staging_ناپ);

    vk_بفر& staging_بفر = dst->device->sync_staging;

    std::vector<vk::بفر_Copy> slices(1);
    إذا (چوڑائی == dpitch) {
        slices[0].src_وتھ_2 = 0;
        slices[0].dst_وتھ_2 = وتھ;
        slices[0].size = staging_ناپ;
    } وإلا {
        slices.resize(اچائی);
        لكل (size_t i = 0; i < اچائی; i++) {
            slices[i].src_وتھ_2 = i * چوڑائی;
            slices[i].dst_وتھ_2 = وتھ + i * dpitch;
            slices[i].size = چوڑائی;
        }
    }

    ggml_vk_sync_buffers(خالی_سوچک, subctx);
    subctx->s->بفر->buf.copy_بفر(staging_بفر->بفر, dst->بفر, slices);

    إذا (چوڑائی == spitch) {
        deferred_memcpy((uint8_t *)staging_بفر->ptr, src, staging_ناپ, &subctx->وچ_memcpys);
    } وإلا {
        لكل (size_t i = 0; i < اچائی; i++) {
            deferred_memcpy((uint8_t *)staging_بفر->ptr + i * چوڑائی, (ثابت uint8_t *) src + i * spitch, چوڑائی, &subctx->وچ_memcpys);
        }
    }
    أرجع سچ;
}

منطقي ggml_vk_بفر_write_async(vk_context subctx, vk_بفر& dst, size_t وتھ, ثابت فراغ * src, size_t size, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    أرجع ggml_vk_بفر_write_2d_async(subctx, dst, وتھ, src, size, size, size, 1, sync_staging);
}

فراغ ggml_vk_بفر_write_2d(vk_بفر& dst, size_t وتھ, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اچائی) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << چوڑائی << ", " << اچائی << ")");
    // Buffer is already mapped
    إذا(dst->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible) {
        GGML_ASSERT(dst->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostCoherent);

        إذا (چوڑائی == spitch && چوڑائی == dpitch) {
            حافظہ_نقل((uint8_t *)dst->ptr + وتھ, src, چوڑائی * اچائی);
        } وإلا {
            لكل (size_t i = 0; i < اچائی; i++) {
                حافظہ_نقل((uint8_t *)dst->ptr + وتھ + i * dpitch, (ثابت uint8_t *) src + i * spitch, چوڑائی);
            }
        }
    } وإلا {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_قطار->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        منطقي ret = ggml_vk_بفر_write_2d_async(subctx, dst, وتھ, src, spitch, dpitch, چوڑائی, اچائی, سچ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_سرا(subctx);

        لكل (خودکار& cpy : subctx->وچ_memcpys) {
            حافظہ_نقل(cpy.dst, cpy.src, cpy.n);
        }

        لكل (خودکار& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, سچ, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_قطار_command_pools_cleanup(dst->device);
    }
}

فراغ ggml_vk_بفر_write(vk_بفر& dst, size_t وتھ, ثابت فراغ * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_بفر_write_2d(dst, وتھ, src, size, size, size, 1);
}

منطقي ggml_vk_بفر_read_2d_async(vk_context subctx, vk_بفر& src, size_t وتھ, فراغ * dst, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اچائی, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << وتھ << ", width=" << چوڑائی << ", height=" << اچائی << ")");
    GGML_ASSERT(چوڑائی > 0);
    GGML_ASSERT(اچائی > 0);
    GGML_ASSERT(src != خالی_سوچک);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_بفر buf = خالی_سوچک;
    size_t buf_وتھ = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_وتھ);

    std::vector<vk::بفر_Copy> slices(1);
    إذا (چوڑائی == spitch && چوڑائی == dpitch) {
        // Only do single write if stride is equal
        slices[0].src_وتھ_2 = وتھ;
        slices[0].dst_وتھ_2 = buf_وتھ;
        slices[0].size = چوڑائی * اچائی;
    } وإلا {
        slices.resize(اچائی);
        لكل (size_t i = 0; i < اچائی; i++) {
            slices[i].src_وتھ_2 = وتھ + i * spitch;
            slices[i].dst_وتھ_2 = buf_وتھ + i * dpitch;
            slices[i].size = چوڑائی;
        }
    }

    إذا (buf != خالی_سوچک) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(خالی_سوچک, subctx);
        subctx->s->بفر->buf.copy_بفر(src->بفر, buf->بفر, slices);

        أرجع سچ;
    }
    VK_LOG_DEBUG("STAGING");

    إذا (!sync_staging) {
        // copy was not handled caller needs to fall back
        أرجع خطأ;
    }

    // Fall back to staging buffer
    ثابت size_t staging_ناپ = چوڑائی * اچائی;
    ggml_vk_ensure_sync_staging_بفر(src->device, staging_ناپ);

    vk_بفر& staging_بفر = src->device->sync_staging;

    std::vector<vk::بفر_Copy> staging_slices(1);
    إذا (چوڑائی == spitch) {
        staging_slices[0].src_وتھ_2 = وتھ;
        staging_slices[0].dst_وتھ_2 = 0;
        staging_slices[0].size = staging_ناپ;
    } وإلا {
        staging_slices.resize(اچائی);
        لكل (size_t i = 0; i < اچائی; i++) {
            staging_slices[i].src_وتھ_2 = وتھ + i * spitch;
            staging_slices[i].dst_وتھ_2 = i * چوڑائی;
            staging_slices[i].size = چوڑائی;
        }
    }

    ggml_vk_sync_buffers(خالی_سوچک, subctx);
    subctx->s->بفر->buf.copy_بفر(src->بفر, staging_بفر->بفر, staging_slices);

    إذا (چوڑائی == dpitch) {
        deferred_memcpy(dst, staging_بفر->ptr, staging_ناپ, &subctx->باہرلا_memcpys);
    } وإلا {
        لكل (size_t i = 0; i < اچائی; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (ثابت uint8_t *) staging_بفر->ptr + i * چوڑائی, چوڑائی, &subctx->باہرلا_memcpys);
        }
    }
    أرجع سچ;
}

ساكن منطقي ggml_vk_بفر_read_async(vk_context subctx, vk_بفر& src, size_t وتھ, فراغ * dst, size_t size, منطقي sync_staging = خطأ) {
    أرجع ggml_vk_بفر_read_2d_async(subctx, src, وتھ, dst, size, size, size, 1, sync_staging);
}

فراغ ggml_vk_بفر_read_2d(vk_بفر& src, size_t وتھ, فراغ * dst, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اچائی) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->بفر << ", " << وتھ << ", " << چوڑائی << ", " << اچائی << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    ثابت منطقي slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostCached);
    إذا(src->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_قطار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->بفر->buf.pipelineBarrier(
            vk::Pipeline_Stage_Flag_بٹ::eComputeShader | vk::Pipeline_Stage_Flag_بٹ::eTransfer,
            vk::Pipeline_Stage_Flag_بٹ::eHost,
            {},
            { { vk::Access_Flag_بٹ::eShaderWrite | vk::Access_Flag_بٹ::eTransferWrite,
                vk::Access_Flag_بٹ::eHostRead } },
            {}, {});
        ggml_vk_ctx_سرا(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, سچ, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_قطار_command_pools_cleanup(src->device);

        إذا (چوڑائی == spitch && چوڑائی == dpitch) {
            حافظہ_نقل(dst, (ثابت uint8_t *) src->ptr + وتھ, چوڑائی * اچائی);
        } وإلا {
            لكل (size_t i = 0; i < اچائی; i++) {
                حافظہ_نقل((uint8_t *) dst + i * dpitch, (ثابت uint8_t *) src->ptr + وتھ + i * spitch, چوڑائی);
            }
        }
    } وإلا {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_قطار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        منطقي ret = ggml_vk_بفر_read_2d_async(subctx, src, وتھ, dst, spitch, dpitch, چوڑائی, اچائی, سچ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_سرا(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, سچ, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_قطار_command_pools_cleanup(src->device);

        لكل (خودکار& cpy : subctx->باہرلا_memcpys) {
            حافظہ_نقل(cpy.dst, cpy.src, cpy.n);
        }
    }
}

فراغ ggml_vk_بفر_read(vk_بفر& src, size_t وتھ, فراغ * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->بفر << ", " << وتھ << ", " << size << ")");
    ggml_vk_بفر_read_2d(src, وتھ, dst, size, size, size, 1);
}

فراغ ggml_vk_بفر_copy_async(vk_context& ctx, vk_بفر& dst, size_t dst_وتھ, vk_بفر& src, size_t src_وتھ, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_بفر_Copy bc{ src_وتھ, dst_وتھ, size };

    vk_Cmd_Copy_بفر(ctx->s->بفر->buf, (Vk_بفر)src->بفر, (Vk_بفر)dst->بفر, 1, &bc);
}

فراغ ggml_vk_بفر_copy(vk_بفر& dst, size_t dst_وتھ, vk_بفر& src, size_t src_وتھ, size_t size) {
    إذا (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_قطار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_بفر_copy_async(subctx, dst, dst_وتھ, src, src_وتھ, size);
        ggml_vk_ctx_سرا(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, سچ, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_قطار_command_pools_cleanup(src->device);
    } وإلا {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_بفر(src->device, size);

        // Copy to src staging buffer
        ggml_vk_بفر_copy(src->device->sync_staging, 0, src, src_وتھ, size);
        // Copy to dst buffer
        ggml_vk_بفر_write(dst, dst_وتھ, src->device->sync_staging->ptr, size);
    }
}

فراغ ggml_vk_بفر_memset_async(vk_context& ctx, vk_بفر& dst, size_t وتھ, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << وتھ << ", " << c << ", " << size << ")");

    إذا (dst->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + وتھ, c, size, &ctx->memsets);
        أرجع;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->بفر->buf.fill_بفر(dst->بفر, وتھ, size, c);
}

فراغ ggml_vk_بفر_memset(vk_بفر& dst, size_t وتھ, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << وتھ << ", " << c << ", " << size << ")");

    إذا (dst->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + وتھ, c, size);
        أرجع;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_قطار->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->بفر->buf.fill_بفر(dst->بفر, وتھ, size, c);
    ggml_vk_ctx_سرا(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, سچ, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_قطار_command_pools_cleanup(dst->device);
}

ggml_backend_بفر_i ggml_backend_vk_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_vk_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_vk_بفر_get_ادھار,
    /* .init_tensor     = */ ggml_backend_vk_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_vk_بفر_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_بفر_سیٹ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_بفر_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_بفر_clear,
    /* .reset           = */ عدم,
};

vk_بفر ggml_vk_بفر_from_host_ptr(vk_device & device, فراغ * ptr, size_t size) {
    إذا (!device->external_حافظہ_host) {
        أرجع {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    إذا (uptr & (device->min_imported_host_اشارہ_گر_alignment - 1)) {
        أرجع {};
    }
    إذا (size & (device->min_imported_host_اشارہ_گر_alignment - 1)) {
        أرجع {};
    }

    ثابت vk::حافظہ_Property_Flags property_flags = vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent | vk::حافظہ_Property_Flag_بٹ::eHostCached;

    vk_بفر buf {};
    حاول {
        buf = ggml_vk_create_بفر(device, size, { property_flags }, ptr);
    } امسك (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    أرجع buf;
}

