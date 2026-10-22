.section .text.init, "ax", @progbits
.عام _start
_start:
    # initialize عام pointer
.option push
.option norelax
    la    gp, __global_pointer$
.option pop
    # Firmware sets stack pointer before launch
    # bss not allowed, no init
    فراخوان  entry_point
    li    a2, 0  /* KERNEL_RETURN_SUCCESS (0) */
    mv    a1, a0
    li    a0, 8  /* SYSCALL_RETURN_FROM_KERNEL (8) */
    ecall
