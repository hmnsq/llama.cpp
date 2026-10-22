#شامل "ggml-vulkan-common.h"

#شامل <cstdint>
#شامل <cstdlib>
#شامل <mutex>
#شامل <shared_mutex>

ساکن خالی * portable_aligned_malloc(size_t align, size_t size) {
#اگر_تعریف_شدہ _WIN32
    واپس _aligned_malloc(size, align);
#ورنہ
    خالی * ptr = خالی_اشاریہ;
    اگر (posix_memalign(&ptr, align, size) != 0) {
        واپس خالی_اشاریہ;
    }
    واپس ptr;
#اگر_ختم
}

ساکن خالی portable_aligned_free(خالی * ptr) {
#اگر_تعریف_شدہ _WIN32
    _aligned_free(ptr);
#ورنہ
    std::آزاد(ptr);
#اگر_ختم
}

ggml_backend_بفر_type_i ggml_backend_vk_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_vk_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ تہی,
    /* .get_alignment       = */ ggml_backend_vk_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_بفر_type_get_max_سائز,
    /* .get_alloc_size      = */ ggml_backend_vk_بفر_type_get_alloc_سائز,
    /* .get_alloc_size_n    = */ تہی,
    /* .is_host             = */ تہی,
};

ساکن std::vector<uint32_t> ggml_vk_ڈھونڈو_حافظہ_properties(ثابت vk::Physical_Device_حافظہ_Properties* mem_props, vk::حافظہ_Requirements* mem_req, vk::حافظہ_Property_Flags flags) {
    std::vector<uint32_t> indices;

    برائے (uint32_t i = 0; i < mem_props->حافظہ_Type_گنتی; ++i) {
        vk::حافظہ_Type حافظہ_type = mem_props->حافظہ_Types[i];
        اگر ((mem_req->حافظہ_Type_بٹ & ((uint64_t)1 << i)) &&
            (flags & حافظہ_type.propertyFlags) == flags &&
            mem_props->حافظہ_Heaps[حافظہ_type.ڈھیر_اشاریہ].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    واپس indices;
}

ساکن vk_بفر ggml_vk_create_بفر(vk_device& device, size_t size, ثابت std::initializer_list<vk::حافظہ_Property_Flags> & req_flags_list,
                                       خالی *import_ptr = خالی_اشاریہ) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    اگر (size > device->max_بفر_سائز) {
        پھینکو vk::برآمد_کا_Device_حافظہ_Error("Requested buffer size exceeds device buffer size limit");
    }

    vk_بفر buf = std::make_shared<vk_بفر_struct>();

    اگر (size == 0) {
        buf->size = 0;
        واپس buf;
    }

    vk::بفر_Usage_Flags usage_flags = vk::بفر_Usage_Flag_بٹ::e_Storage_بفر | vk::بفر_Usage_Flag_بٹ::eTransferSrc | vk::بفر_Usage_Flag_بٹ::eTransferDst;
    vk::حافظہ_Allocate_Flags mem_flags {};
    اگر (device->بفر_device_پتہ) {
        usage_flags |= vk::بفر_Usage_Flag_بٹ::e_Shader_Device_پتہ;
        mem_flags |= vk::حافظہ_Allocate_Flag_بٹ::e_Device_پتہ;
    }

    vk::بفر_Create_Info بفر_create_info{
        vk::بفر_Create_Flags(),
        size,
        usage_flags,
        vk::Sharing_منوال::eExclusive,
        0,
        خالی_اشاریہ,
    };

    vk::External_حافظہ_بفر_Create_Info external_حافظہ_bci;
    اگر (import_ptr) {
        external_حافظہ_bci.handleTypes = vk::External_حافظہ_Handle_Type_Flag_بٹ::e_Host_تخصیص_EXT;
        بفر_create_info.سیٹ_PNext(&external_حافظہ_bci);
    }

    buf->بفر = device->device.create_بفر(بفر_create_info);

    vk::حافظہ_Requirements mem_req = device->device.get_بفر_حافظہ_Requirements(buf->بفر);

    vk::Physical_Device_حافظہ_Properties mem_props = device->physical_device.get_حافظہ_Properties();

    ثابت vk::حافظہ_ترجیح_Allocate_Info_EXT mem_ترجیح_info { 1.0f };

    vk::حافظہ_Allocate_Flags_Info mem_flags_info { mem_flags };

    اگر (device->حافظہ_ترجیح) {
        mem_flags_info.سیٹ_PNext(&mem_ترجیح_info);
    }

    اگر (import_ptr) {
        vk::حافظہ_Host_اشارہ_گر_Properties_EXT host_اشارہ_گر_props;
        کوشش {
            host_اشارہ_گر_props = device->device.get_حافظہ_Host_اشارہ_گر_Properties_EXT(vk::External_حافظہ_Handle_Type_Flag_بٹ::e_Host_تخصیص_EXT, import_ptr);
        } پکڑو (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_بفر(buf->بفر);
            واپس {};
        }
        vk::Physical_Device_حافظہ_Properties mem_props = device->physical_device.get_حافظہ_Properties();

        uint32_t حافظہ_type_idx;
        vk::حافظہ_Property_Flags property_flags = *req_flags_list.begin();
        برائے (حافظہ_type_idx = 0; حافظہ_type_idx < 32; ++حافظہ_type_idx) {
            اگر (!(host_اشارہ_گر_props.حافظہ_Type_بٹ & (1u << حافظہ_type_idx))) {
                جاری;
            }
            اگر (!(mem_req.حافظہ_Type_بٹ & (1u << حافظہ_type_idx))) {
                جاری;
            }

            vk::حافظہ_Type حافظہ_type = mem_props.حافظہ_Types[حافظہ_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            اگر ((حافظہ_type.propertyFlags & property_flags) == property_flags) {
                property_flags = حافظہ_type.propertyFlags;
                توڑو;
            }
        }
        اگر (حافظہ_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_بفر(buf->بفر);
            واپس {};
        }

        buf->حافظہ_property_flags = mem_props.حافظہ_Types[حافظہ_type_idx].propertyFlags;
        کوشش {
            vk::Import_حافظہ_Host_اشارہ_گر_Info_EXT import_info;
            import_info.handleType = vk::External_حافظہ_Handle_Type_Flag_بٹ::e_Host_تخصیص_EXT;
            import_info.p_Host_اشارہ_گر = import_ptr;
            import_info.سیٹ_PNext(&mem_flags_info);
            buf->device_حافظہ = device->device.allocate_حافظہ({ size, حافظہ_type_idx, &import_info });
        } پکڑو (ثابت vk::SystemError& e) {
        }
    } ورنہ {
        برائے (خودکار it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            ثابت خودکار & req_flags = *it;

            ثابت std::vector<uint32_t> حافظہ_type_indices = ggml_vk_ڈھونڈو_حافظہ_properties(&mem_props, &mem_req, req_flags);

            اگر (حافظہ_type_indices.خالی_خانے()) {
                جاری;
            }

            منطقی مکمل = جھوٹ;

            برائے (خودکار mtype_it = حافظہ_type_indices.begin(); mtype_it != حافظہ_type_indices.end(); mtype_it++) {
                کوشش {
                    buf->device_حافظہ = device->device.allocate_حافظہ({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->حافظہ_property_flags = mem_props.حافظہ_Types[*mtype_it].propertyFlags;
                    مکمل = سچ;
                    توڑو;
                } پکڑو (ثابت vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    اگر (it + 1 == req_flags_list.end() && mtype_it + 1 == حافظہ_type_indices.end()) {
                        device->device.destroy_بفر(buf->بفر);
                        پھینکو e;
                    }
                }
            }

            اگر (مکمل) {
                توڑو;
            }
        }
    }

    اگر (!buf->device_حافظہ) {
        device->device.destroy_بفر(buf->بفر);
        پھینکو vk::برآمد_کا_Device_حافظہ_Error("No suitable memory type found");
    }

    buf->ptr = خالی_اشاریہ;

    اگر (import_ptr) {
        buf->ptr = import_ptr;
    } ورنہ {
        اگر (buf->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible) {
            buf->ptr = device->device.map_حافظہ(buf->device_حافظہ, 0, VK_پورا_سائز);
        }
    }

    device->device.bind_بفر_حافظہ(buf->بفر, buf->device_حافظہ, 0);

    buf->device = device;
    buf->size = size;

    اگر (device->بفر_device_پتہ) {
        ثابت vk::بفر_Device_پتہ_Info پتہ_Info(buf->بفر);
        buf->bda_addr = device->device.get_بفر_پتہ(پتہ_Info);
    }

    device->حافظہ_logger->log_تخصیص(buf, size);

    واپس buf;
}

vk_بفر ggml_vk_create_بفر_check(vk_device& device, size_t size, vk::حافظہ_Property_Flags req_flags, vk::حافظہ_Property_Flags fallback_flags) {
    کوشش {
        واپس ggml_vk_create_بفر(device, size, {req_flags, fallback_flags});
    } پکڑو (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::سطر_ختم;
        std::cerr << "ggml_vulkan: " << e.what() << std::سطر_ختم;
        پھینکو e;
    }
}

vk_بفر ggml_vk_create_بفر_device(vk_device& device, size_t size) {
    vk_بفر buf;
    کوشش {
        اگر (device->prefer_host_حافظہ) {
            buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent,
                                                       vk::حافظہ_Property_Flag_بٹ::eDeviceLocal});
        } ورنہ اگر (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eDeviceLocal | vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent,
                                                       vk::حافظہ_Property_Flag_بٹ::eDeviceLocal,
                                                       vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent});
        } ورنہ اگر (device->disable_host_visible_vidmem) {
            اگر (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eDeviceLocal,
                                                           vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent});
            } ورنہ {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eDeviceLocal});
            }
        } ورنہ {
            // use rebar if available, otherwise fallback to device only visible memory
            اگر (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eDeviceLocal | vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent,
                                                           vk::حافظہ_Property_Flag_بٹ::eDeviceLocal,
                                                           vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent});
            } ورنہ {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظہ_Property_Flag_بٹ::eDeviceLocal | vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent,
                                                           vk::حافظہ_Property_Flag_بٹ::eDeviceLocal});
            }
        }
    } پکڑو (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::سطر_ختم;
        std::cerr << "ggml_vulkan: " << e.what() << std::سطر_ختم;
        پھینکو e;
    }

    واپس buf;
}

خالی ggml_vk_destroy_بفر(vk_بفر& buf) {
    اگر (buf == خالی_اشاریہ) {
        واپس;
    }

    اگر (buf->device != خالی_اشاریہ) {
        buf->device->حافظہ_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
ساکن size_t ggml_vk_host_تخصیص_ڈھونڈو(ثابت خالی * ptr) {
    برائے (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        ثابت uint8_t * بنیاد = (ثابت uint8_t *) vk_instance.host_allocations[i].بنیاد;
        اگر (ptr >= بنیاد && ptr < بنیاد + vk_instance.host_allocations[i].size) {
            واپس i;
        }
    }
    واپس SIZE_MAX;
}

خالی * ggml_vk_host_malloc(size_t size) {
    VK_LOG_حافظہ("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    اگر (vk_instance.multi_device) {
        ثابت size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        ثابت size_t alloc_سائز = (size + align - 1) / align * align;

        خالی * ptr = portable_aligned_malloc(align, alloc_سائز);
        اگر (ptr == خالی_اشاریہ) {
            واپس خالی_اشاریہ;
        }
        vk_host_تخصیص اندراج_2;
        اندراج_2.بنیاد = ptr;
        اندراج_2.size = alloc_سائز;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(اندراج_2));
        واپس ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_بفر buf;
    کوشش {
        buf = ggml_vk_create_بفر(dev0, size,
            {vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent | vk::حافظہ_Property_Flag_بٹ::eHostCached,
             vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent});
    } پکڑو (vk::SystemError&) {
        واپس خالی_اشاریہ;
    }

    اگر (!(buf->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_بفر(buf);
        واپس خالی_اشاریہ;
    }

    vk_host_تخصیص اندراج_2;
    اندراج_2.بنیاد             = buf->ptr;
    اندراج_2.size             = size;
    اندراج_2.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(اندراج_2));
    واپس buf->ptr;
}

خالی ggml_vk_host_free(خالی * ptr) {
    اگر (ptr == خالی_اشاریہ) {
        واپس;
    }
    VK_LOG_حافظہ("ggml_vk_host_free(" << ptr << ")");

    vk_host_تخصیص اندراج_2;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        ثابت size_t اشاریہ = ggml_vk_host_تخصیص_ڈھونڈو(ptr);
        اگر (اشاریہ == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            واپس;
        }
        اندراج_2 = std::move(vk_instance.host_allocations[اشاریہ]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + اشاریہ);
    }

    برائے (خودکار & view : اندراج_2.views) {
        ggml_vk_destroy_بفر(view);
    }
    اگر (vk_instance.multi_device) {
        portable_aligned_free(اندراج_2.بنیاد);
    }
}

خالی ggml_vk_host_get(ثابت vk_device& device, ثابت خالی * ptr, vk_بفر& buf, size_t& buf_انحراف) {
    buf = خالی_اشاریہ;
    buf_انحراف = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    ثابت size_t اشاریہ = ggml_vk_host_تخصیص_ڈھونڈو(ptr);
    اگر (اشاریہ == SIZE_MAX) {
        واپس;
    }
    vk_host_تخصیص & اندراج_2 = vk_instance.host_allocations[اشاریہ];

    اگر (اندراج_2.views[device->idx] == خالی_اشاریہ) {
        ثابت uint32_t ثنائی_ہندسہ = 1u << device->idx;
        اگر (!vk_instance.multi_device || !vk_instance.device_host_بفر_import[device->idx] || (vk_instance.host_import_failed & ثنائی_ہندسہ)) {
            واپس;
        }
        GGML_ASSERT(device->external_حافظہ_host);
        // const only propagates from the const context; the pointee is mutable
        اندراج_2.views[device->idx] = ggml_vk_بفر_from_host_ptr(const_cast<vk_device&>(device), اندراج_2.بنیاد, اندراج_2.size);
        اگر (اندراج_2.views[device->idx] == خالی_اشاریہ) {
            vk_instance.host_import_failed |= ثنائی_ہندسہ;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            واپس;
        }
    }

    buf = اندراج_2.views[device->idx];
    buf_انحراف = (ثابت uint8_t *) ptr - (ثابت uint8_t *) اندراج_2.بنیاد;
}

خالی ggml_vk_ensure_sync_staging_بفر(vk_device& device, size_t size) {
    اگر (device->sync_staging == خالی_اشاریہ || device->sync_staging->size < size) {
        VK_LOG_حافظہ("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_بفر(device->sync_staging);
        device->sync_staging = ggml_vk_create_بفر_check(device, size,
            vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent | vk::حافظہ_Property_Flag_بٹ::eHostCached,
            vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent);
    }
}

خالی ggml_vk_ensure_sync_staging_بفر(ggml_backend_vk_context * ctx, size_t size) {
    اگر (ctx->sync_staging == خالی_اشاریہ || ctx->sync_staging->size < size) {
        VK_LOG_حافظہ("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_بفر(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_بفر_check(ctx->device, size,
            vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent | vk::حافظہ_Property_Flag_بٹ::eHostCached,
            vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent);
    }
}

ساکن خالی ggml_vk_بفر_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_بفر& dst, size_t انحراف, ثابت ggml_tensor * tensor, منطقی sync_staging = جھوٹ) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    اگر(dst->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::سطر_ختم;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_بفر buf = خالی_اشاریہ;
    size_t buf_انحراف = 0;
    ggml_vk_host_get(ctx->device, tensor->ڈیٹا, buf, buf_انحراف);

    ثابت uint64_t ne0 = tensor->ne[0];
    ثابت uint64_t ne1 = tensor->ne[1];
    ثابت uint64_t ne2 = tensor->ne[2];
    ثابت uint64_t ne3 = tensor->ne[3];
    ثابت uint64_t nb0 = tensor->nb[0];
    ثابت uint64_t nb1 = tensor->nb[1];
    ثابت uint64_t nb2 = tensor->nb[2];
    ثابت uint64_t nb3 = tensor->nb[3];
    ثابت ggml_type type = tensor->type;
    ثابت uint64_t ts = ggml_type_سائز(type);
    ثابت uint64_t bs = ggml_blck_سائز(type);

    ثابت uint64_t dstnb0 = ts;
    ثابت uint64_t dstnb1 = dstnb0*(ne0/bs);
    ثابت uint64_t dstnb2 = dstnb1*ne1;
    ثابت uint64_t dstnb3 = dstnb2*ne2;

    ثابت uint64_t ne = ggml_nelements(tensor);

    اگر (buf != خالی_اشاریہ) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::بفر_Copy> slices;

        برائے (uint64_t i3 = 0; i3 < ne3; i3++) {
            برائے (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                اگر (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_انحراف + i3*nb3 + i2*nb2, انحراف + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } ورنہ {
                    برائے (uint64_t i1 = 0; i1 < ne1; i1++) {
                        اگر (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_انحراف + i3*nb3 + i2*nb2 + i1*nb1, انحراف + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } ورنہ {
                            ثابت uint64_t s_off = buf_انحراف + i3*nb3 + i2*nb2 + i1*nb1;
                            ثابت uint64_t d_off = انحراف + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            برائے (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->بفر->buf.copy_بفر(buf->بفر, dst->بفر, slices);
        واپس;
    }

    اگر (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_بفر& staging = ctx->device->sync_staging;
    ثابت uint64_t copy_سائز = ts*ne/bs;
    ggml_vk_ensure_sync_staging_بفر(ctx->device, copy_سائز);
    Vk_بفر_Copy buf_copy{ 0, انحراف, copy_سائز };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_بفر(subctx->s->بفر->buf, (Vk_بفر)staging->بفر, (Vk_بفر)dst->بفر, 1, &buf_copy);

    برائے (uint64_t i3 = 0; i3 < ne3; i3++) {
        برائے (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            اگر (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (ثابت uint8_t *) tensor->ڈیٹا + buf_انحراف + i3*nb3 + i2*nb2, dstnb2, &subctx->اندر_memcpys);
            } ورنہ {
                برائے (uint64_t i1 = 0; i1 < ne1; i1++) {
                    اگر (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (ثابت uint8_t *) tensor->ڈیٹا + buf_انحراف + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->اندر_memcpys);
                    } ورنہ {
                        ثابت uint64_t s_off = buf_انحراف + i3*nb3 + i2*nb2 + i1*nb1;
                        ثابت uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        برائے (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (ثابت uint8_t *) tensor->ڈیٹا + s_off + i0*nb0, dstnb0, &subctx->اندر_memcpys);
                        }
                    }
                }
            }
        }
    }
}

منطقی ggml_vk_بفر_write_2d_async(vk_context subctx, vk_بفر& dst, size_t انحراف, ثابت خالی * src, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اونچائی, منطقی sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << چوڑائی << ", " << اونچائی << ")");
    // Check if src is pinned memory
    vk_بفر buf = خالی_اشاریہ;
    size_t buf_انحراف = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_انحراف);

    اگر (buf != خالی_اشاریہ) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::بفر_Copy> slices(1);
        اگر (چوڑائی == spitch && چوڑائی == dpitch) {
            // Only do single write if stride is equal
            slices[0].src_انحراف_2 = buf_انحراف;
            slices[0].dst_انحراف_2 = انحراف;
            slices[0].size = چوڑائی * اونچائی;
        } ورنہ {
            slices.resize(اونچائی);
            برائے (size_t i = 0; i < اونچائی; i++) {
                slices[i].src_انحراف_2 = buf_انحراف + i * spitch;
                slices[i].dst_انحراف_2 = انحراف + i * dpitch;
                slices[i].size = چوڑائی;
            }
        }

        ggml_vk_sync_buffers(خالی_اشاریہ, subctx);
        subctx->s->بفر->buf.copy_بفر(buf->بفر, dst->بفر, slices);
        واپس سچ;
    }
    VK_LOG_DEBUG("STAGING");

    اگر (!sync_staging) {
        // copy was not handled caller needs to fall back
        واپس جھوٹ;
    }

    // Staging buffer required
    ثابت size_t staging_سائز = چوڑائی * اونچائی;
    ggml_vk_ensure_sync_staging_بفر(dst->device, staging_سائز);

    vk_بفر& staging_بفر = dst->device->sync_staging;

    std::vector<vk::بفر_Copy> slices(1);
    اگر (چوڑائی == dpitch) {
        slices[0].src_انحراف_2 = 0;
        slices[0].dst_انحراف_2 = انحراف;
        slices[0].size = staging_سائز;
    } ورنہ {
        slices.resize(اونچائی);
        برائے (size_t i = 0; i < اونچائی; i++) {
            slices[i].src_انحراف_2 = i * چوڑائی;
            slices[i].dst_انحراف_2 = انحراف + i * dpitch;
            slices[i].size = چوڑائی;
        }
    }

    ggml_vk_sync_buffers(خالی_اشاریہ, subctx);
    subctx->s->بفر->buf.copy_بفر(staging_بفر->بفر, dst->بفر, slices);

    اگر (چوڑائی == spitch) {
        deferred_memcpy((uint8_t *)staging_بفر->ptr, src, staging_سائز, &subctx->اندر_memcpys);
    } ورنہ {
        برائے (size_t i = 0; i < اونچائی; i++) {
            deferred_memcpy((uint8_t *)staging_بفر->ptr + i * چوڑائی, (ثابت uint8_t *) src + i * spitch, چوڑائی, &subctx->اندر_memcpys);
        }
    }
    واپس سچ;
}

منطقی ggml_vk_بفر_write_async(vk_context subctx, vk_بفر& dst, size_t انحراف, ثابت خالی * src, size_t size, منطقی sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    واپس ggml_vk_بفر_write_2d_async(subctx, dst, انحراف, src, size, size, size, 1, sync_staging);
}

خالی ggml_vk_بفر_write_2d(vk_بفر& dst, size_t انحراف, ثابت خالی * src, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اونچائی) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << چوڑائی << ", " << اونچائی << ")");
    // Buffer is already mapped
    اگر(dst->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible) {
        GGML_ASSERT(dst->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostCoherent);

        اگر (چوڑائی == spitch && چوڑائی == dpitch) {
            حافظہ_نقل((uint8_t *)dst->ptr + انحراف, src, چوڑائی * اونچائی);
        } ورنہ {
            برائے (size_t i = 0; i < اونچائی; i++) {
                حافظہ_نقل((uint8_t *)dst->ptr + انحراف + i * dpitch, (ثابت uint8_t *) src + i * spitch, چوڑائی);
            }
        }
    } ورنہ {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_قطار->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        منطقی ret = ggml_vk_بفر_write_2d_async(subctx, dst, انحراف, src, spitch, dpitch, چوڑائی, اونچائی, سچ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_انتہا(subctx);

        برائے (خودکار& cpy : subctx->اندر_memcpys) {
            حافظہ_نقل(cpy.dst, cpy.src, cpy.n);
        }

        برائے (خودکار& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, سچ, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_قطار_command_pools_cleanup(dst->device);
    }
}

خالی ggml_vk_بفر_write(vk_بفر& dst, size_t انحراف, ثابت خالی * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_بفر_write_2d(dst, انحراف, src, size, size, size, 1);
}

منطقی ggml_vk_بفر_read_2d_async(vk_context subctx, vk_بفر& src, size_t انحراف, خالی * dst, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اونچائی, منطقی sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << انحراف << ", width=" << چوڑائی << ", height=" << اونچائی << ")");
    GGML_ASSERT(چوڑائی > 0);
    GGML_ASSERT(اونچائی > 0);
    GGML_ASSERT(src != خالی_اشاریہ);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_بفر buf = خالی_اشاریہ;
    size_t buf_انحراف = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_انحراف);

    std::vector<vk::بفر_Copy> slices(1);
    اگر (چوڑائی == spitch && چوڑائی == dpitch) {
        // Only do single write if stride is equal
        slices[0].src_انحراف_2 = انحراف;
        slices[0].dst_انحراف_2 = buf_انحراف;
        slices[0].size = چوڑائی * اونچائی;
    } ورنہ {
        slices.resize(اونچائی);
        برائے (size_t i = 0; i < اونچائی; i++) {
            slices[i].src_انحراف_2 = انحراف + i * spitch;
            slices[i].dst_انحراف_2 = buf_انحراف + i * dpitch;
            slices[i].size = چوڑائی;
        }
    }

    اگر (buf != خالی_اشاریہ) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(خالی_اشاریہ, subctx);
        subctx->s->بفر->buf.copy_بفر(src->بفر, buf->بفر, slices);

        واپس سچ;
    }
    VK_LOG_DEBUG("STAGING");

    اگر (!sync_staging) {
        // copy was not handled caller needs to fall back
        واپس جھوٹ;
    }

    // Fall back to staging buffer
    ثابت size_t staging_سائز = چوڑائی * اونچائی;
    ggml_vk_ensure_sync_staging_بفر(src->device, staging_سائز);

    vk_بفر& staging_بفر = src->device->sync_staging;

    std::vector<vk::بفر_Copy> staging_slices(1);
    اگر (چوڑائی == spitch) {
        staging_slices[0].src_انحراف_2 = انحراف;
        staging_slices[0].dst_انحراف_2 = 0;
        staging_slices[0].size = staging_سائز;
    } ورنہ {
        staging_slices.resize(اونچائی);
        برائے (size_t i = 0; i < اونچائی; i++) {
            staging_slices[i].src_انحراف_2 = انحراف + i * spitch;
            staging_slices[i].dst_انحراف_2 = i * چوڑائی;
            staging_slices[i].size = چوڑائی;
        }
    }

    ggml_vk_sync_buffers(خالی_اشاریہ, subctx);
    subctx->s->بفر->buf.copy_بفر(src->بفر, staging_بفر->بفر, staging_slices);

    اگر (چوڑائی == dpitch) {
        deferred_memcpy(dst, staging_بفر->ptr, staging_سائز, &subctx->برآمد_memcpys);
    } ورنہ {
        برائے (size_t i = 0; i < اونچائی; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (ثابت uint8_t *) staging_بفر->ptr + i * چوڑائی, چوڑائی, &subctx->برآمد_memcpys);
        }
    }
    واپس سچ;
}

ساکن منطقی ggml_vk_بفر_read_async(vk_context subctx, vk_بفر& src, size_t انحراف, خالی * dst, size_t size, منطقی sync_staging = جھوٹ) {
    واپس ggml_vk_بفر_read_2d_async(subctx, src, انحراف, dst, size, size, size, 1, sync_staging);
}

خالی ggml_vk_بفر_read_2d(vk_بفر& src, size_t انحراف, خالی * dst, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اونچائی) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->بفر << ", " << انحراف << ", " << چوڑائی << ", " << اونچائی << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    ثابت منطقی slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostCached);
    اگر(src->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible && src->device->uma && !slow_host_read) {
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
        ggml_vk_ctx_انتہا(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, سچ, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_قطار_command_pools_cleanup(src->device);

        اگر (چوڑائی == spitch && چوڑائی == dpitch) {
            حافظہ_نقل(dst, (ثابت uint8_t *) src->ptr + انحراف, چوڑائی * اونچائی);
        } ورنہ {
            برائے (size_t i = 0; i < اونچائی; i++) {
                حافظہ_نقل((uint8_t *) dst + i * dpitch, (ثابت uint8_t *) src->ptr + انحراف + i * spitch, چوڑائی);
            }
        }
    } ورنہ {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_قطار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        منطقی ret = ggml_vk_بفر_read_2d_async(subctx, src, انحراف, dst, spitch, dpitch, چوڑائی, اونچائی, سچ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_انتہا(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, سچ, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_قطار_command_pools_cleanup(src->device);

        برائے (خودکار& cpy : subctx->برآمد_memcpys) {
            حافظہ_نقل(cpy.dst, cpy.src, cpy.n);
        }
    }
}

خالی ggml_vk_بفر_read(vk_بفر& src, size_t انحراف, خالی * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->بفر << ", " << انحراف << ", " << size << ")");
    ggml_vk_بفر_read_2d(src, انحراف, dst, size, size, size, 1);
}

خالی ggml_vk_بفر_copy_async(vk_context& ctx, vk_بفر& dst, size_t dst_انحراف, vk_بفر& src, size_t src_انحراف, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_بفر_Copy bc{ src_انحراف, dst_انحراف, size };

    vk_Cmd_Copy_بفر(ctx->s->بفر->buf, (Vk_بفر)src->بفر, (Vk_بفر)dst->بفر, 1, &bc);
}

خالی ggml_vk_بفر_copy(vk_بفر& dst, size_t dst_انحراف, vk_بفر& src, size_t src_انحراف, size_t size) {
    اگر (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_قطار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_بفر_copy_async(subctx, dst, dst_انحراف, src, src_انحراف, size);
        ggml_vk_ctx_انتہا(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, سچ, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_قطار_command_pools_cleanup(src->device);
    } ورنہ {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_بفر(src->device, size);

        // Copy to src staging buffer
        ggml_vk_بفر_copy(src->device->sync_staging, 0, src, src_انحراف, size);
        // Copy to dst buffer
        ggml_vk_بفر_write(dst, dst_انحراف, src->device->sync_staging->ptr, size);
    }
}

خالی ggml_vk_بفر_memset_async(vk_context& ctx, vk_بفر& dst, size_t انحراف, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << انحراف << ", " << c << ", " << size << ")");

    اگر (dst->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + انحراف, c, size, &ctx->memsets);
        واپس;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->بفر->buf.fill_بفر(dst->بفر, انحراف, size, c);
}

خالی ggml_vk_بفر_memset(vk_بفر& dst, size_t انحراف, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << انحراف << ", " << c << ", " << size << ")");

    اگر (dst->حافظہ_property_flags & vk::حافظہ_Property_Flag_بٹ::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + انحراف, c, size);
        واپس;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_قطار->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->بفر->buf.fill_بفر(dst->بفر, انحراف, size, c);
    ggml_vk_ctx_انتہا(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, سچ, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_قطار_command_pools_cleanup(dst->device);
}

ggml_backend_بفر_i ggml_backend_vk_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_vk_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_vk_بفر_get_بنیاد,
    /* .init_tensor     = */ ggml_backend_vk_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_vk_بفر_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_بفر_سیٹ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_بفر_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_بفر_clear,
    /* .reset           = */ تہی,
};

vk_بفر ggml_vk_بفر_from_host_ptr(vk_device & device, خالی * ptr, size_t size) {
    اگر (!device->external_حافظہ_host) {
        واپس {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    اگر (uptr & (device->min_imported_host_اشارہ_گر_alignment - 1)) {
        واپس {};
    }
    اگر (size & (device->min_imported_host_اشارہ_گر_alignment - 1)) {
        واپس {};
    }

    ثابت vk::حافظہ_Property_Flags property_flags = vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent | vk::حافظہ_Property_Flag_بٹ::eHostCached;

    vk_بفر buf {};
    کوشش {
        buf = ggml_vk_create_بفر(device, size, { property_flags }, ptr);
    } پکڑو (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    واپس buf;
}

