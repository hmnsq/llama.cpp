#شامل "transport.h"
#شامل "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#وگرنه
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#شامل <cstdlib>
#شامل <cstring>
#شامل <mutex>
#شامل <optional>

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <array>
#  include <cerrno>
#  include <chrono>
#  include <time.h>
#  ifndef _WIN32
#    include <poll.h>
#  endif
#  ifdef GGML_RPC_RDMA_APPLE
#    include "transport-apple.h"
#  endif
#endif // GGML_RPC_RDMA

#ifdef _WIN32
تعریفنوع SOCKET sockfd_t;
using ssize_t = __int64;
#وگرنه
تعریفنوع صحیح sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
ایستا constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#اگر defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
ایستا constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
ایستا constexpr صحیح    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
ایستا constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

ساختار rdma_conn {
    ساختار ibv_context * ctx = nullptr;
    ساختار ibv_pd * pd  = nullptr;
    ساختار ibv_cq * scq = nullptr;   // send completions
    ساختار ibv_cq * rcq = nullptr;   // recv completions
    ساختار ibv_qp * qp  = nullptr;
    ساختار ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    تهی          * tx_buf = nullptr;
    ساختار ibv_mr * tx_mr  = nullptr;

    تهی          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    ساختار ibv_mr * rx_mr  = nullptr;
    صحیح             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(صحیح i) ثابت {
        بازگشت static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    منطقی post_rx(صحیح i) {
        ساختار ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        ساختار ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.رقم_sge = 1;
        بازگشت ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        اگر (tx_mr) ibv_dereg_mr(tx_mr);
        اگر (rx_mr) ibv_dereg_mr(rx_mr);
        آزاد(tx_buf);
        آزاد(rx_buf);
        اگر (qp)  ibv_destroy_qp(qp);
        اگر (scq) ibv_destroy_cq(scq);
        اگر (rcq) ibv_destroy_cq(rcq);
        اگر (ch)  ibv_destroy_comp_channel(ch);
        اگر (pd)  ibv_dealloc_pd(pd);
        اگر (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
ساختار rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    صحیح      gid_idx = 0;
    شمارش ibv_mtu path_mtu = IBV_MTU_1024;
};

ساختار rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(اندازه(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

ساختار socket_t::impl {
    impl(sockfd_t fd) : use_rdma(نادرست), fd(fd) {}
    ~impl();
    منطقی send_data(ثابت تهی * data, size_t size);
    منطقی recv_data(تهی * data, size_t size);
    منطقی flush();
    تهی get_caps(uint8_t * local_caps);
    تهی update_caps(ثابت uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  وگرنه
    منطقی rdma_probe();
    منطقی rdma_send(ثابت تهی * data, size_t size);
    منطقی rdma_recv(تهی * data, size_t size);
    منطقی tcp_peer_closed();
    منطقی rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ثابت uint8_t * remote_gid);
    منطقی rdma_poll(ساختار ibv_cq * cq, ساختار ibv_wc * wc);
    منطقی rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    منطقی     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, این->fd);
#ifdef _WIN32
    اگر (fd != INVALID_SOCKET) closesocket(این->fd);
#وگرنه
    اگر (fd >= 0) close(این->fd);
#endif
}

#ifdef GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = اندازه(addr);
    اگر (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        بازگشت std::nullopt;
    }
    rdma_gid_t target = {};
    اگر (addr.ss_family == AF_INET) {
        ثابت auto * a = reinterpret_cast<ثابت sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        بازگشت target;
    }
    اگر (addr.ss_family == AF_INET6) {
        ثابت auto * a = reinterpret_cast<ثابت sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        بازگشت target;
    }
    بازگشت std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

منطقی socket_t::impl::tcp_peer_closed() {
    اگر (fd < 0) بازگشت نادرست;
#ifndef _WIN32
    ساختار pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    صحیح r = poll(&pfd, 1, 0);
    بازگشت r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#وگرنه
    بازگشت نادرست;
#endif
}

منطقی socket_t::impl::rdma_probe() {
    ثابت نویسه * dev_env = std::getenv("GGML_RDMA_DEV");
    ثابت نویسه * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    اگر (!target_gid) {
        بازگشت نادرست;
    }

    ثابت uint8_t ib_port = 1;
    صحیح رقم_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&رقم_devs);
    اگر (!devs || رقم_devs == 0) بازگشت نادرست;

    ibv_context * ibctx = nullptr;
    ثابت نویسه * matched_dev = nullptr;
    صحیح gid_idx = gid_env ? atoi(gid_env) : -1;
    صحیح gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    برای (صحیح d = 0; d < رقم_devs; d++) {
        ثابت نویسه * dn = ibv_get_device_name(devs[d]);
        اگر (dev_env && strcmp(dev_env, dn) != 0) ادامه;

        ibv_context * ctx = ibv_open_device(devs[d]);
        اگر (!ctx) ادامه;

        ibv_port_attr pa;
        اگر (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); ادامه; }

        صحیح found_gid = gid_idx;
        صحیح found_version = IBV_GID_TYPE_IB;
        اگر (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            صحیح v2_idx = -1;
            صحیح v1_idx = -1;
            برای (صحیح i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                اگر (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) ادامه;
                اگر (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) ادامه;
                اگر (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } وگرنه اگر (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            اگر (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } وگرنه اگر (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } وگرنه {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            اگر (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        اگر (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            بشکن;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    اگر (!ibctx) بازگشت نادرست;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    اگر (!rdma->pd) بازگشت نادرست;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    اگر (!rdma->scq || !rdma->rcq) بازگشت نادرست;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    اگر (!rdma->qp) بازگشت نادرست;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    اگر (!rdma->tx_buf || !rdma->rx_buf) بازگشت نادرست;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    اگر (!rdma->tx_mr || !rdma->rx_mr) بازگشت نادرست;

    ibv_gid local_gid;
    اگر (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) بازگشت نادرست;

    rdma_local.qpn = rdma->qp->qp_رقم;
    rdma_local.psn = rdma->qp->qp_رقم & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    ثابت نویسه * ver_str = "";
    اگر (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } وگرنه اگر (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    بازگشت درست;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
منطقی socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ثابت uint8_t * remote_gid) {
    // RESET -> INIT
    {
        ساختار ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_رقم        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        اگر (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            بازگشت نادرست;
        }
    }

    برای (صحیح i = 0; i < RDMA_RX_DEPTH; i++) {
        اگر (!rdma->post_rx(i)) بازگشت نادرست;
    }

    // INIT -> RTR
    {
        ساختار ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_رقم        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_رقم   = rdma_local.ib_port;
        اگر (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            بازگشت نادرست;
        }
    }

    // RTR -> RTS
    {
        ساختار ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        اگر (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            بازگشت نادرست;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    بازگشت درست;
}

// Sleep until the completion channel has an event or the TCP peer closes.
منطقی socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    ساختار pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    اگر (poll(pfds, 2, -1) < 0) {
        بازگشت errno == EINTR;
    }
    اگر (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        بازگشت نادرست;
    }
    اگر (pfds[0].revents & POLLIN) {
        ساختار ibv_cq * ev_cq  = nullptr;
        تهی          * ev_ctx = nullptr;
        اگر (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            بازگشت نادرست;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    بازگشت درست;
}

منطقی socket_t::impl::rdma_poll(ساختار ibv_cq * cq, ساختار ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    منطقی armed = نادرست;
    برای (uint64_t s = 1; ; s++) {
        صحیح n = ibv_poll_cq(cq, 1, wc);
        اگر (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            اگر (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            بازگشت wc->status == IBV_WC_SUCCESS;
        }
        اگر (n < 0) بازگشت نادرست;
        اگر (armed) {
            // armed and still empty: sleep until the next completion
            اگر (!rdma_wait_event()) {
                بازگشت نادرست;
            }
            armed = نادرست;
            ادامه;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        اگر (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            اگر (ibv_req_notify_cq(cq, 0) != 0) {
                بازگشت نادرست;
            }
            armed = درست;
            ادامه;
        }
        اگر ((s & 0xFFFFF) == 0) {
            اگر (tcp_peer_closed()) {
                بازگشت نادرست;
            }
        }
    }
}

منطقی socket_t::impl::rdma_send(ثابت تهی * data, size_t size) {
    rdma_conn * c = rdma.get();
    ثابت uint8_t * src = (ثابت uint8_t *)data;
    size_t rem = size;
    وقتی (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        ساختار ibv_sge sge = {};
        ساختار ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.رقم_sge = 1;

        اگر (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } وگرنه {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        اگر (ibv_post_send(c->qp, &wr, &bad) != 0) بازگشت نادرست;
        c->last_active = std::chrono::steady_clock::now();
        ساختار ibv_wc wc;
        اگر (!rdma_poll(c->scq, &wc)) بازگشت نادرست;

        src += chunk;
        rem -= chunk;
    }
    بازگشت درست;
}

منطقی socket_t::impl::rdma_recv(تهی * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    وقتی (rem > 0) {
        ساختار ibv_wc wc;
        اگر (!rdma_poll(c->rcq, &wc)) بازگشت نادرست;

        صحیح slot = (صحیح)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        اگر (!c->post_rx(slot)) بازگشت نادرست;

        dst += got;
        rem -= got;
    }
    بازگشت درست;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

منطقی socket_t::impl::send_data(ثابت تهی * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    اگر (use_rdma) {
        بازگشت rdma->send(data, size);
    }
#وگرنهاگر defined(GGML_RPC_RDMA)
    اگر (use_rdma) {
        بازگشت rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    وقتی (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (ثابت نویسه *)data + bytes_sent, size_to_send, 0);
        اگر (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            بازگشت نادرست;
        }
        bytes_sent += (size_t)n;
    }
    بازگشت درست;
}

منطقی socket_t::impl::recv_data(تهی * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    اگر (use_rdma) {
        بازگشت rdma->recv(data, size);
    }
#وگرنهاگر defined(GGML_RPC_RDMA)
    اگر (use_rdma) {
        بازگشت rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    وقتی (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (نویسه *)data + bytes_recv, size_to_recv, 0);
        اگر (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            بازگشت نادرست;
        }
        اگر (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            بازگشت نادرست;
        }
        bytes_recv += (size_t)n;
    }
    بازگشت درست;
}

تهی socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    اگر (std::getenv("GGML_RPC_NO_RDMA")) {
        بازگشت;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    اگر (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    اگر (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  وگرنه
    rdma_local = {};
    اگر (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, اندازه(rc));
    } وگرنه {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

تهی socket_t::impl::update_caps(ثابت uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    منطقی remote_rdma = نادرست;
    برای (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    اگر (!rdma || !remote_rdma) {
        اگر (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        بازگشت;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    منطقی activated = rdma->activate(remote_caps);
#  وگرنه
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, اندازه(rc));
    منطقی activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    اگر (activated) {
        use_rdma = درست;
    } وگرنه {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#وگرنه
    (تهی)remote_caps;
#endif // GGML_RPC_RDMA
}

منطقی socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    اگر (use_rdma) {
        بازگشت rdma->flush();
    }
#endif
    بازگشت درست;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = پیشفرض;

منطقی socket_t::send_data(ثابت تهی * data, size_t size) {
    بازگشت pimpl->send_data(data, size);
}

منطقی socket_t::recv_data(تهی * data, size_t size) {
    بازگشت pimpl->recv_data(data, size);
}

منطقی socket_t::flush() {
    بازگشت pimpl->flush();
}

تهی socket_t::get_caps(uint8_t * local_caps) {
    بازگشت pimpl->get_caps(local_caps);
}

تهی socket_t::update_caps(ثابت uint8_t * remote_caps) {
    بازگشت pimpl->update_caps(remote_caps);
}

ثابت نویسه * socket_t::transport_name() ثابت {
    بازگشت pimpl->use_rdma ? "RDMA" : "TCP";
}

ایستا منطقی is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    بازگشت sockfd != INVALID_SOCKET;
#وگرنه
    بازگشت sockfd >= 0;
#endif
}

ایستا منطقی set_no_delay(sockfd_t sockfd) {
    صحیح flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    صحیح ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (نویسه *)&flag, اندازه(صحیح));
    بازگشت ret == 0;
}

ایستا منطقی set_reuse_addr(sockfd_t sockfd) {
    صحیح flag = 1;
    صحیح ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (نویسه *)&flag, اندازه(صحیح));
    بازگشت ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, پوچ, پوچ);
    اگر (!is_valid_fd(client_socket_fd)) {
        بازگشت nullptr;
    }
    اگر (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        بازگشت nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    بازگشت socket_ptr(جدید socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ثابت نویسه * host, صحیح port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    اگر (!is_valid_fd(sockfd)) {
        بازگشت nullptr;
    }
    اگر (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        بازگشت nullptr;
    }
    اگر (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        بازگشت nullptr;
    }
    ساختار sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    اگر (bind(sockfd, (ساختار sockaddr *) &serv_addr, اندازه(serv_addr)) < 0) {
        بازگشت nullptr;
    }
    اگر (listen(sockfd, 1) < 0) {
        بازگشت nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    بازگشت socket_ptr(جدید socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ثابت نویسه * host, صحیح port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    اگر (!is_valid_fd(sockfd)) {
        بازگشت nullptr;
    }
    اگر (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        بازگشت nullptr;
    }
    ساختار sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    ساختار hostent * server = gethostbyname(host);
    اگر (server == پوچ) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        بازگشت nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    اگر (::connect(sockfd, (ساختار sockaddr *)&addr, اندازه(addr)) < 0) {
        بازگشت nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    بازگشت socket_ptr(جدید socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
ایستا std::mutex g_rpc_transport_mu;
ایستا منطقی g_rpc_transport_wsa_started = نادرست;
#endif

منطقی rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    اگر (g_rpc_transport_wsa_started) {
        بازگشت درست;
    }
    WSADATA wsaData;
    صحیح res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    اگر (res != 0) {
        بازگشت نادرست;
    }
    g_rpc_transport_wsa_started = درست;
    بازگشت درست;
#وگرنه
    بازگشت درست;
#endif
}

تهی rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    اگر (!g_rpc_transport_wsa_started) {
        بازگشت;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = نادرست;
#endif
}
