#!/usr/bin/env python3

درآمد argparse
درآمد logging
درآمد os
درآمد shutil
درآمد struct
درآمد sys
درآمد tempfile
درآمد zlib

ELF_MAGIC = b'\x7fELF'
ELF_CLASS_اشاریہ = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_نسخہ_اشاریہ = 6
EV_CURRENT = 1
MAX_PHDR_گنتی = 100

بھرو_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_سائز_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


جماعت Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    تعریف __init__(self, ڈیٹا):
        unpacked_ڈیٹا = (Elf_Ehdr_common.s).unpack(ڈیٹا)
        self.e_ident = unpacked_ڈیٹا[0]
        self.e_type = unpacked_ڈیٹا[1]
        self.e_machine = unpacked_ڈیٹا[2]
        self.e_نسخہ = unpacked_ڈیٹا[3]


جماعت Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    تعریف __init__(self, ڈیٹا):
        unpacked_ڈیٹا = (Elf32_Ehdr.s).unpack(ڈیٹا)
        self.e_ident = unpacked_ڈیٹا[0]
        self.e_type = unpacked_ڈیٹا[1]
        self.e_machine = unpacked_ڈیٹا[2]
        self.e_نسخہ = unpacked_ڈیٹا[3]
        self.e_اندراج = unpacked_ڈیٹا[4]
        self.e_phoff = unpacked_ڈیٹا[5]
        self.e_shoff = unpacked_ڈیٹا[6]
        self.e_flags = unpacked_ڈیٹا[7]
        self.e_ehsize = unpacked_ڈیٹا[8]
        self.e_phentsize = unpacked_ڈیٹا[9]
        self.e_phnum = unpacked_ڈیٹا[10]
        self.e_shentsize = unpacked_ڈیٹا[11]
        self.e_shnum = unpacked_ڈیٹا[12]
        self.e_shstrndx = unpacked_ڈیٹا[13]

    تعریف get_packed_ڈیٹا(self):
        اقدار = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_نسخہ,
                  self.e_اندراج,
                  self.e_phoff,
                  self.e_shoff,
                  self.e_flags,
                  self.e_ehsize,
                  self.e_phentsize,
                  self.e_phnum,
                  self.e_shentsize,
                  self.e_shnum,
                  self.e_shstrndx
                  ]

        واپس (Elf32_Ehdr.s).pack(*اقدار)


جماعت Elf32_Phdr:
    s = struct.Struct('I' * 8)

    تعریف __init__(self, ڈیٹا):
        unpacked_ڈیٹا = (Elf32_Phdr.s).unpack(ڈیٹا)
        self.p_type = unpacked_ڈیٹا[0]
        self.p_انحراف = unpacked_ڈیٹا[1]
        self.p_vaddr = unpacked_ڈیٹا[2]
        self.p_paddr = unpacked_ڈیٹا[3]
        self.p_filesz = unpacked_ڈیٹا[4]
        self.p_memsz = unpacked_ڈیٹا[5]
        self.p_flags = unpacked_ڈیٹا[6]
        self.p_align = unpacked_ڈیٹا[7]

    تعریف get_packed_ڈیٹا(self):
        اقدار = [self.p_type,
                  self.p_انحراف,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        واپس (Elf32_Phdr.s).pack(*اقدار)


جماعت Elf32_Shdr:
    s = struct.Struct('I' * 10)

    تعریف __init__(self, ڈیٹا):
        unpacked_ڈیٹا = (Elf32_Shdr.s).unpack(ڈیٹا)
        self.sh_name = unpacked_ڈیٹا[0]
        self.sh_type = unpacked_ڈیٹا[1]
        self.sh_flags = unpacked_ڈیٹا[2]
        self.sh_addr = unpacked_ڈیٹا[3]
        self.sh_انحراف = unpacked_ڈیٹا[4]
        self.sh_سائز = unpacked_ڈیٹا[5]
        self.sh_link = unpacked_ڈیٹا[6]
        self.sh_info = unpacked_ڈیٹا[7]
        self.sh_addralign = unpacked_ڈیٹا[8]
        self.sh_entsize = unpacked_ڈیٹا[9]
        self.sh_name_str = ""

    تعریف get_packed_ڈیٹا(self):
        اقدار = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_انحراف,
                  self.sh_سائز,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        واپس (Elf32_Shdr.s).pack(*اقدار)


جماعت Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    تعریف __init__(self, ڈیٹا):
        unpacked_ڈیٹا = (Elf64_Ehdr.s).unpack(ڈیٹا)
        self.e_ident = unpacked_ڈیٹا[0]
        self.e_type = unpacked_ڈیٹا[1]
        self.e_machine = unpacked_ڈیٹا[2]
        self.e_نسخہ = unpacked_ڈیٹا[3]
        self.e_اندراج = unpacked_ڈیٹا[4]
        self.e_phoff = unpacked_ڈیٹا[5]
        self.e_shoff = unpacked_ڈیٹا[6]
        self.e_flags = unpacked_ڈیٹا[7]
        self.e_ehsize = unpacked_ڈیٹا[8]
        self.e_phentsize = unpacked_ڈیٹا[9]
        self.e_phnum = unpacked_ڈیٹا[10]
        self.e_shentsize = unpacked_ڈیٹا[11]
        self.e_shnum = unpacked_ڈیٹا[12]
        self.e_shstrndx = unpacked_ڈیٹا[13]

    تعریف get_packed_ڈیٹا(self):
        اقدار = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_نسخہ,
                  self.e_اندراج,
                  self.e_phoff,
                  self.e_shoff,
                  self.e_flags,
                  self.e_ehsize,
                  self.e_phentsize,
                  self.e_phnum,
                  self.e_shentsize,
                  self.e_shnum,
                  self.e_shstrndx
                  ]

        واپس (Elf64_Ehdr.s).pack(*اقدار)


جماعت Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    تعریف __init__(self, ڈیٹا):
        unpacked_ڈیٹا = (Elf64_Phdr.s).unpack(ڈیٹا)
        self.p_type = unpacked_ڈیٹا[0]
        self.p_flags = unpacked_ڈیٹا[1]
        self.p_انحراف = unpacked_ڈیٹا[2]
        self.p_vaddr = unpacked_ڈیٹا[3]
        self.p_paddr = unpacked_ڈیٹا[4]
        self.p_filesz = unpacked_ڈیٹا[5]
        self.p_memsz = unpacked_ڈیٹا[6]
        self.p_align = unpacked_ڈیٹا[7]

    تعریف get_packed_ڈیٹا(self):
        اقدار = [self.p_type,
                  self.p_flags,
                  self.p_انحراف,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        واپس (Elf64_Phdr.s).pack(*اقدار)


جماعت Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    تعریف __init__(self, ڈیٹا):
        unpacked_ڈیٹا = (Elf64_Shdr.s).unpack(ڈیٹا)
        self.sh_name = unpacked_ڈیٹا[0]
        self.sh_type = unpacked_ڈیٹا[1]
        self.sh_flags = unpacked_ڈیٹا[2]
        self.sh_addr = unpacked_ڈیٹا[3]
        self.sh_انحراف = unpacked_ڈیٹا[4]
        self.sh_سائز = unpacked_ڈیٹا[5]
        self.sh_link = unpacked_ڈیٹا[6]
        self.sh_info = unpacked_ڈیٹا[7]
        self.sh_addralign = unpacked_ڈیٹا[8]
        self.sh_entsize = unpacked_ڈیٹا[9]
        self.sh_name_str = ""

    تعریف get_packed_ڈیٹا(self):
        اقدار = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_انحراف,
                  self.sh_سائز,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        واپس (Elf64_Shdr.s).pack(*اقدار)


جماعت SWIV_Seg:
    s = struct.Struct('4sIQ')

    تعریف __init__(self):
        self.magic = SWIV_MAGIC
        self.چکری_جانچ = 0
        self.padding = 0

    تعریف get_packed_ڈیٹا(self):
        اقدار = [self.magic,
                  self.چکری_جانچ,
                  self.padding
                  ]

        واپس (SWIV_Seg.s).pack(*اقدار)


تعریف is_elf64(elf_سرخی):
    واپس elf_سرخی.e_ident[ELF_CLASS_اشاریہ] == ELFCLASS64


تعریف verify_elf_سرخی(elf_سرخی):
    ident = elf_سرخی.e_ident
    واپس (ident.startswith(ELF_MAGIC)
            اور ident[ELF_CLASS_اشاریہ] میں (ELFCLASS32, ELFCLASS64)
            اور ident[ELF_نسخہ_اشاریہ] == EV_CURRENT)


تعریف file_copy_انحراف(اندر_fp, اندر_off, برآمد_fp, برآمد_off, عدد_بائٹیں):
    اندر_fp.seek(اندر_off)
    read_اندر = اندر_fp.اقرأ(عدد_بائٹیں)
    برآمد_fp.seek(برآمد_off)
    برآمد_fp.اكتب(read_اندر)

    واپس عدد_بائٹیں


تعریف preprocess_elf_file(elf_file_name):
    ساتھ open(elf_file_name, 'rb') بطور elf_fp:
        elf_سرخی = Elf_Ehdr_common(elf_fp.اقرأ(Elf_Ehdr_common.s.سائز))

        اگر verify_elf_سرخی(elf_سرخی) ہے جھوٹ:
            اٹھاؤ RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        اگر is_elf64(elf_سرخی):
            elf_سرخی = Elf64_Ehdr(elf_fp.اقرأ(Elf64_Ehdr.s.سائز))
        ورنہ:
            elf_سرخی = Elf32_Ehdr(elf_fp.اقرأ(Elf32_Ehdr.s.سائز))

        phdr_جدول = []

        phdr_سائز = elf_سرخی.e_phentsize

        file_انحراف = elf_سرخی.e_phoff
        elf_fp.seek(file_انحراف)

        برائے _ میں سلسلہ(elf_سرخی.e_phnum):
            اگر is_elf64(elf_سرخی):
                phdr_جدول.append(Elf64_Phdr(elf_fp.اقرأ(phdr_سائز)))
            ورنہ:
                phdr_جدول.append(Elf32_Phdr(elf_fp.اقرأ(phdr_سائز)))

    واپس [elf_سرخی, phdr_جدول]


تعریف process_section_سرخی(elf_file_name):
    ساتھ open(elf_file_name, 'rb') بطور elf_fp:
        elf_سرخی = Elf_Ehdr_common(elf_fp.اقرأ(Elf_Ehdr_common.s.سائز))

        اگر verify_elf_سرخی(elf_سرخی) ہے جھوٹ:
            اٹھاؤ RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        اگر is_elf64(elf_سرخی):
            elf_سرخی = Elf64_Ehdr(elf_fp.اقرأ(Elf64_Ehdr.s.سائز))
        ورنہ:
            elf_سرخی = Elf32_Ehdr(elf_fp.اقرأ(Elf32_Ehdr.s.سائز))

        shdr_جدول = []

        shdr_سائز = elf_سرخی.e_shentsize
        elf_fp.seek(elf_سرخی.e_shoff)

        برائے _ میں سلسلہ(elf_سرخی.e_shnum):
            اگر is_elf64(elf_سرخی):
                shdr_جدول.append(Elf64_Shdr(elf_fp.اقرأ(shdr_سائز)))
            ورنہ:
                shdr_جدول.append(Elf32_Shdr(elf_fp.اقرأ(shdr_سائز)))

    واپس shdr_جدول


تعریف چکری_جانچ_32_update_file(elf_fp, انحراف, سائز, چکری_جانچ):
    elf_fp.seek(انحراف)
    بقایا = سائز
    جب بقایا > 0:
        chunk = elf_fp.اقرأ(min(1024 * 1024, بقایا))
        اگر نہیں chunk:
            اٹھاؤ RuntimeError("Unexpected EOF while calculating CRC")
        چکری_جانچ = zlib.چکری_جانچ_32(chunk, چکری_جانچ)
        بقایا -= لمبائی(chunk)
    واپس چکری_جانچ


تعریف چکری_جانچ_32_update_zeros(سائز, چکری_جانچ):
    zero_chunk = b'\x00' * min(1024 * 1024, سائز)
    بقایا = سائز
    جب بقایا > 0:
        chunk = zero_chunk[:min(لمبائی(zero_chunk), بقایا)]
        چکری_جانچ = zlib.چکری_جانچ_32(chunk, چکری_جانچ)
        بقایا -= لمبائی(chunk)
    واپس چکری_جانچ


تعریف چکری_جانچ_32_calculate(elf_fp, phdr_جدول):
    چکری_جانچ = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    برائے curr_phdr میں phdr_جدول:
        اگر curr_phdr.p_type != بھرو_TYPE:
            جاری

        چکری_جانچ = چکری_جانچ_32_update_file(elf_fp, curr_phdr.p_انحراف, curr_phdr.p_filesz, چکری_جانچ)

        zero_سائز = curr_phdr.p_memsz - curr_phdr.p_filesz
        اگر zero_سائز > 0:
            چکری_جانچ = چکری_جانچ_32_update_zeros(zero_سائز, چکری_جانچ)

    چکری_جانچ &= 0xffffffff
    logger.info("calculated CRC: %#x", چکری_جانچ)

    واپس چکری_جانچ


تعریف verify_swiv_seg(elf_اندر_fp, انحراف):

    elf_اندر_fp.seek(انحراف)
    swiv_seg = SWIV_Seg()
    unpacked_ڈیٹا = (SWIV_Seg.s).unpack(elf_اندر_fp.اقرأ(SWIV_SEG_سائز_MIN))
    swiv_seg.magic = unpacked_ڈیٹا[0]

    واپس swiv_seg.magic == SWIV_MAGIC


تعریف same_output_file(داخلہ_elf, output_elf):
    کوشش:
        واپس os.راستہ.exists(output_elf) اور os.راستہ.samefile(داخلہ_elf, output_elf)
    استثنا File_Not_ملا_Error:
        واپس جھوٹ


تعریف make_output_ہدف(داخلہ_elf, output_elf):
    اگر نہیں same_output_file(داخلہ_elf, output_elf):
        واپس output_elf, کچھ_نہیں

    output_dir = os.راستہ.dirname(os.راستہ.abspath(output_elf))
    output_بنیاد = os.راستہ.basename(output_elf)
    fd, عارضی_output = tempfile.mkstemp(سابقہ=output_بنیاد + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(داخلہ_elf, عارضی_output)
    واپس عارضی_output, عارضی_output


تعریف swiv_generate(داخلہ_elf, output_elf):
    output_ہدف, عارضی_output = make_output_ہدف(داخلہ_elf, output_elf)

    کوشش:
        ساتھ open(داخلہ_elf, "rb") بطور elf_اندر_fp, open(output_ہدف, "wb+") بطور elf_برآمد_fp:
            [elf_سرخی, phdr_جدول] = preprocess_elf_file(داخلہ_elf)

            عدد_phdrs = elf_سرخی.e_phnum
            عدد_shdrs = elf_سرخی.e_shnum
            phdr_سائز = elf_سرخی.e_phentsize

            اگر عدد_phdrs == 0 یا عدد_phdrs > MAX_PHDR_گنتی:
                اٹھاؤ RuntimeError("Input ELF has invalid number of program headers.")

            اگر elf_سرخی.e_shoff == 0 یا عدد_shdrs == 0:
                اٹھاؤ RuntimeError("Input ELF does not contain a section header table.")

            shdr_جدول_داخلہ = process_section_سرخی(داخلہ_elf)
            shdr_جدول_output = process_section_سرخی(داخلہ_elf)

            آخری_phdr = phdr_جدول[عدد_phdrs - 1]
            اگر آخری_phdr.p_memsz == SWIV_SEG_سائز_MIN اور verify_swiv_seg(elf_اندر_fp, آخری_phdr.p_انحراف):
                اٹھاؤ RuntimeError("A SWIV segment is detected in the input ELF.")

            آخری_shdr = shdr_جدول_داخلہ[elf_سرخی.e_shnum - 1]
            اگر آخری_shdr.sh_سائز == SWIV_SEG_سائز_MIN اور verify_swiv_seg(elf_اندر_fp, آخری_shdr.sh_انحراف):
                اٹھاؤ RuntimeError("A SWIV section is detected in the input ELF.")

            اگر is_elf64(elf_سرخی):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.سائز)
            ورنہ:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.سائز)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_سائز = SWIV_SEG_سائز_MIN
            swiv_shdr.sh_انحراف = elf_سرخی.e_shoff + (elf_سرخی.e_shnum + 1) * elf_سرخی.e_shentsize
            shdr_جدول_output.append(swiv_shdr)
            elf_سرخی.e_shnum += 1

            اگر elf_سرخی.e_phoff != 0:
                برائے phdr میں phdr_جدول:
                    file_copy_انحراف(elf_اندر_fp, phdr.p_انحراف, elf_برآمد_fp, phdr.p_انحراف, phdr.p_filesz)

            elf_برآمد_fp.seek(0)
            elf_برآمد_fp.اكتب(elf_سرخی.get_packed_ڈیٹا())

            phdr_آغاز = elf_سرخی.e_phoff
            برائے phdr میں phdr_جدول:
                elf_برآمد_fp.seek(phdr_آغاز)
                elf_برآمد_fp.اكتب(phdr.get_packed_ڈیٹا())
                phdr_آغاز += phdr_سائز

            برائے shdr میں shdr_جدول_داخلہ[:عدد_shdrs]:
                اگر shdr.sh_type == SHT_NOBITS یا shdr.sh_سائز == 0:
                    جاری
                file_copy_انحراف(elf_اندر_fp, shdr.sh_انحراف, elf_برآمد_fp, shdr.sh_انحراف, shdr.sh_سائز)

            shdr_آغاز = elf_سرخی.e_shoff
            برائے shdr میں shdr_جدول_output:
                elf_برآمد_fp.seek(shdr_آغاز)
                elf_برآمد_fp.اكتب(shdr.get_packed_ڈیٹا())
                shdr_آغاز += elf_سرخی.e_shentsize

            elf_برآمد_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.چکری_جانچ = چکری_جانچ_32_calculate(elf_برآمد_fp, phdr_جدول)
            elf_برآمد_fp.seek(swiv_shdr.sh_انحراف)
            elf_برآمد_fp.اكتب(swiv_seg.get_packed_ڈیٹا())

        اگر عارضی_output ہے نہیں کچھ_نہیں:
            os.replace(عارضی_output, output_elf)
            عارضی_output = کچھ_نہیں

        واپس 0
    آخرکار:
        اگر عارضی_output ہے نہیں کچھ_نہیں اور os.راستہ.exists(عارضی_output):
            os.unlink(عارضی_output)


تعریف مین(argv):
    parser = argparse.دلیل_Parser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.جوڑو_دلیل("-i", "--input", dest="input_file", metavar="INPUT", required=سچ)
    parser.جوڑو_دلیل("-o", "--output", dest="output_file", metavar="OUTPUT", required=سچ)
    parser.جوڑو_دلیل("-v", "--verbose", action="store_true", help="show calculated CRC")
    دلائل_2 = parser.parse_دلائل(argv)

    logging.basicConfig(
        level=logging.INFO اگر دلائل_2.verbose ورنہ logging.WARNING,
        format="%(message)s",
    )

    کوشش:
        واپس swiv_generate(دلائل_2.داخلہ_file, دلائل_2.output_file)
    استثنا RuntimeError بطور exc:
        logger.error("ERROR: %s", exc)
        واپس 2


اگر __name__ == "__main__":
    sys.exit(مین(sys.argv[1:]))
