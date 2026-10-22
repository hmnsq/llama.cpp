#!/usr/bin/env python3

استورد argparse
استورد logging
استورد os
استورد shutil
استورد struct
استورد sys
استورد tempfile
استورد zlib

ELF_MAGIC = b'\x7fELF'
ELF_CLASS_اشاريو = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_نسخو_اشاريو = 6
EV_CURRENT = 1
MAX_PHDR_ڳڻپ = 100

ڀريو_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_سائيز_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


صنف Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    وصف __init__(self, معلومات):
        unpacked_معلومات = (Elf_Ehdr_common.s).unpack(معلومات)
        self.e_ident = unpacked_معلومات[0]
        self.e_type = unpacked_معلومات[1]
        self.e_machine = unpacked_معلومات[2]
        self.e_نسخو = unpacked_معلومات[3]


صنف Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    وصف __init__(self, معلومات):
        unpacked_معلومات = (Elf32_Ehdr.s).unpack(معلومات)
        self.e_ident = unpacked_معلومات[0]
        self.e_type = unpacked_معلومات[1]
        self.e_machine = unpacked_معلومات[2]
        self.e_نسخو = unpacked_معلومات[3]
        self.e_داخلا = unpacked_معلومات[4]
        self.e_phoff = unpacked_معلومات[5]
        self.e_shoff = unpacked_معلومات[6]
        self.e_flags = unpacked_معلومات[7]
        self.e_ehsize = unpacked_معلومات[8]
        self.e_phentsize = unpacked_معلومات[9]
        self.e_phnum = unpacked_معلومات[10]
        self.e_shentsize = unpacked_معلومات[11]
        self.e_shnum = unpacked_معلومات[12]
        self.e_shstrndx = unpacked_معلومات[13]

    وصف get_packed_معلومات(self):
        قدرون = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_نسخو,
                  self.e_داخلا,
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

        أرجع (Elf32_Ehdr.s).pack(*قدرون)


صنف Elf32_Phdr:
    s = struct.Struct('I' * 8)

    وصف __init__(self, معلومات):
        unpacked_معلومات = (Elf32_Phdr.s).unpack(معلومات)
        self.p_type = unpacked_معلومات[0]
        self.p_وٿي = unpacked_معلومات[1]
        self.p_vaddr = unpacked_معلومات[2]
        self.p_paddr = unpacked_معلومات[3]
        self.p_filesz = unpacked_معلومات[4]
        self.p_memsz = unpacked_معلومات[5]
        self.p_flags = unpacked_معلومات[6]
        self.p_align = unpacked_معلومات[7]

    وصف get_packed_معلومات(self):
        قدرون = [self.p_type,
                  self.p_وٿي,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        أرجع (Elf32_Phdr.s).pack(*قدرون)


صنف Elf32_Shdr:
    s = struct.Struct('I' * 10)

    وصف __init__(self, معلومات):
        unpacked_معلومات = (Elf32_Shdr.s).unpack(معلومات)
        self.sh_name = unpacked_معلومات[0]
        self.sh_type = unpacked_معلومات[1]
        self.sh_flags = unpacked_معلومات[2]
        self.sh_addr = unpacked_معلومات[3]
        self.sh_وٿي = unpacked_معلومات[4]
        self.sh_سائيز = unpacked_معلومات[5]
        self.sh_link = unpacked_معلومات[6]
        self.sh_info = unpacked_معلومات[7]
        self.sh_addralign = unpacked_معلومات[8]
        self.sh_entsize = unpacked_معلومات[9]
        self.sh_name_str = ""

    وصف get_packed_معلومات(self):
        قدرون = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_وٿي,
                  self.sh_سائيز,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        أرجع (Elf32_Shdr.s).pack(*قدرون)


صنف Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    وصف __init__(self, معلومات):
        unpacked_معلومات = (Elf64_Ehdr.s).unpack(معلومات)
        self.e_ident = unpacked_معلومات[0]
        self.e_type = unpacked_معلومات[1]
        self.e_machine = unpacked_معلومات[2]
        self.e_نسخو = unpacked_معلومات[3]
        self.e_داخلا = unpacked_معلومات[4]
        self.e_phoff = unpacked_معلومات[5]
        self.e_shoff = unpacked_معلومات[6]
        self.e_flags = unpacked_معلومات[7]
        self.e_ehsize = unpacked_معلومات[8]
        self.e_phentsize = unpacked_معلومات[9]
        self.e_phnum = unpacked_معلومات[10]
        self.e_shentsize = unpacked_معلومات[11]
        self.e_shnum = unpacked_معلومات[12]
        self.e_shstrndx = unpacked_معلومات[13]

    وصف get_packed_معلومات(self):
        قدرون = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_نسخو,
                  self.e_داخلا,
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

        أرجع (Elf64_Ehdr.s).pack(*قدرون)


صنف Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    وصف __init__(self, معلومات):
        unpacked_معلومات = (Elf64_Phdr.s).unpack(معلومات)
        self.p_type = unpacked_معلومات[0]
        self.p_flags = unpacked_معلومات[1]
        self.p_وٿي = unpacked_معلومات[2]
        self.p_vaddr = unpacked_معلومات[3]
        self.p_paddr = unpacked_معلومات[4]
        self.p_filesz = unpacked_معلومات[5]
        self.p_memsz = unpacked_معلومات[6]
        self.p_align = unpacked_معلومات[7]

    وصف get_packed_معلومات(self):
        قدرون = [self.p_type,
                  self.p_flags,
                  self.p_وٿي,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        أرجع (Elf64_Phdr.s).pack(*قدرون)


صنف Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    وصف __init__(self, معلومات):
        unpacked_معلومات = (Elf64_Shdr.s).unpack(معلومات)
        self.sh_name = unpacked_معلومات[0]
        self.sh_type = unpacked_معلومات[1]
        self.sh_flags = unpacked_معلومات[2]
        self.sh_addr = unpacked_معلومات[3]
        self.sh_وٿي = unpacked_معلومات[4]
        self.sh_سائيز = unpacked_معلومات[5]
        self.sh_link = unpacked_معلومات[6]
        self.sh_info = unpacked_معلومات[7]
        self.sh_addralign = unpacked_معلومات[8]
        self.sh_entsize = unpacked_معلومات[9]
        self.sh_name_str = ""

    وصف get_packed_معلومات(self):
        قدرون = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_وٿي,
                  self.sh_سائيز,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        أرجع (Elf64_Shdr.s).pack(*قدرون)


صنف SWIV_Seg:
    s = struct.Struct('4sIQ')

    وصف __init__(self):
        self.magic = SWIV_MAGIC
        self.چڪري_جاچ = 0
        self.padding = 0

    وصف get_packed_معلومات(self):
        قدرون = [self.magic,
                  self.چڪري_جاچ,
                  self.padding
                  ]

        أرجع (SWIV_Seg.s).pack(*قدرون)


وصف is_elf64(elf_سرو):
    أرجع elf_سرو.e_ident[ELF_CLASS_اشاريو] == ELFCLASS64


وصف verify_elf_سرو(elf_سرو):
    ident = elf_سرو.e_ident
    أرجع (ident.startswith(ELF_MAGIC)
            و ident[ELF_CLASS_اشاريو] في (ELFCLASS32, ELFCLASS64)
            و ident[ELF_نسخو_اشاريو] == EV_CURRENT)


وصف file_copy_وٿي(اندريون_fp, اندريون_off, ٻاهريون_fp, ٻاهريون_off, انگ_بائيٽون):
    اندريون_fp.seek(اندريون_off)
    read_اندريون = اندريون_fp.اقرأ(انگ_بائيٽون)
    ٻاهريون_fp.seek(ٻاهريون_off)
    ٻاهريون_fp.اكتب(read_اندريون)

    أرجع انگ_بائيٽون


وصف preprocess_elf_file(elf_file_name):
    سان open(elf_file_name, 'rb') كـ elf_fp:
        elf_سرو = Elf_Ehdr_common(elf_fp.اقرأ(Elf_Ehdr_common.s.سائيز))

        إذا verify_elf_سرو(elf_سرو) آهي خطأ:
            ارفع RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        إذا is_elf64(elf_سرو):
            elf_سرو = Elf64_Ehdr(elf_fp.اقرأ(Elf64_Ehdr.s.سائيز))
        وإلا:
            elf_سرو = Elf32_Ehdr(elf_fp.اقرأ(Elf32_Ehdr.s.سائيز))

        phdr_جدول = []

        phdr_سائيز = elf_سرو.e_phentsize

        file_وٿي = elf_سرو.e_phoff
        elf_fp.seek(file_وٿي)

        لكل _ في سلسلو(elf_سرو.e_phnum):
            إذا is_elf64(elf_سرو):
                phdr_جدول.append(Elf64_Phdr(elf_fp.اقرأ(phdr_سائيز)))
            وإلا:
                phdr_جدول.append(Elf32_Phdr(elf_fp.اقرأ(phdr_سائيز)))

    أرجع [elf_سرو, phdr_جدول]


وصف process_section_سرو(elf_file_name):
    سان open(elf_file_name, 'rb') كـ elf_fp:
        elf_سرو = Elf_Ehdr_common(elf_fp.اقرأ(Elf_Ehdr_common.s.سائيز))

        إذا verify_elf_سرو(elf_سرو) آهي خطأ:
            ارفع RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        إذا is_elf64(elf_سرو):
            elf_سرو = Elf64_Ehdr(elf_fp.اقرأ(Elf64_Ehdr.s.سائيز))
        وإلا:
            elf_سرو = Elf32_Ehdr(elf_fp.اقرأ(Elf32_Ehdr.s.سائيز))

        shdr_جدول = []

        shdr_سائيز = elf_سرو.e_shentsize
        elf_fp.seek(elf_سرو.e_shoff)

        لكل _ في سلسلو(elf_سرو.e_shnum):
            إذا is_elf64(elf_سرو):
                shdr_جدول.append(Elf64_Shdr(elf_fp.اقرأ(shdr_سائيز)))
            وإلا:
                shdr_جدول.append(Elf32_Shdr(elf_fp.اقرأ(shdr_سائيز)))

    أرجع shdr_جدول


وصف چڪري_جاچ_32_update_file(elf_fp, وٿي, سائيز, چڪري_جاچ):
    elf_fp.seek(وٿي)
    بچيل = سائيز
    طالما بچيل > 0:
        chunk = elf_fp.اقرأ(min(1024 * 1024, بچيل))
        إذا ليس chunk:
            ارفع RuntimeError("Unexpected EOF while calculating CRC")
        چڪري_جاچ = zlib.چڪري_جاچ_32(chunk, چڪري_جاچ)
        بچيل -= ڊيگهه(chunk)
    أرجع چڪري_جاچ


وصف چڪري_جاچ_32_update_zeros(سائيز, چڪري_جاچ):
    zero_chunk = b'\x00' * min(1024 * 1024, سائيز)
    بچيل = سائيز
    طالما بچيل > 0:
        chunk = zero_chunk[:min(ڊيگهه(zero_chunk), بچيل)]
        چڪري_جاچ = zlib.چڪري_جاچ_32(chunk, چڪري_جاچ)
        بچيل -= ڊيگهه(chunk)
    أرجع چڪري_جاچ


وصف چڪري_جاچ_32_calculate(elf_fp, phdr_جدول):
    چڪري_جاچ = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    لكل curr_phdr في phdr_جدول:
        إذا curr_phdr.p_type != ڀريو_TYPE:
            استمر

        چڪري_جاچ = چڪري_جاچ_32_update_file(elf_fp, curr_phdr.p_وٿي, curr_phdr.p_filesz, چڪري_جاچ)

        zero_سائيز = curr_phdr.p_memsz - curr_phdr.p_filesz
        إذا zero_سائيز > 0:
            چڪري_جاچ = چڪري_جاچ_32_update_zeros(zero_سائيز, چڪري_جاچ)

    چڪري_جاچ &= 0xffffffff
    logger.info("calculated CRC: %#x", چڪري_جاچ)

    أرجع چڪري_جاچ


وصف verify_swiv_seg(elf_اندريون_fp, وٿي):

    elf_اندريون_fp.seek(وٿي)
    swiv_seg = SWIV_Seg()
    unpacked_معلومات = (SWIV_Seg.s).unpack(elf_اندريون_fp.اقرأ(SWIV_SEG_سائيز_MIN))
    swiv_seg.magic = unpacked_معلومات[0]

    أرجع swiv_seg.magic == SWIV_MAGIC


وصف same_output_file(داخلا_elf, output_elf):
    حاول:
        أرجع os.رستو.exists(output_elf) و os.رستو.samefile(داخلا_elf, output_elf)
    استثناء File_Not_مليو_Error:
        أرجع خطأ


وصف make_output_هدف(داخلا_elf, output_elf):
    إذا ليس same_output_file(داخلا_elf, output_elf):
        أرجع output_elf, ڪجهه_نه

    output_dir = os.رستو.dirname(os.رستو.abspath(output_elf))
    output_بنياد = os.رستو.basename(output_elf)
    fd, عارضي_output = tempfile.mkstemp(اڳياڙي=output_بنياد + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(داخلا_elf, عارضي_output)
    أرجع عارضي_output, عارضي_output


وصف swiv_generate(داخلا_elf, output_elf):
    output_هدف, عارضي_output = make_output_هدف(داخلا_elf, output_elf)

    حاول:
        سان open(داخلا_elf, "rb") كـ elf_اندريون_fp, open(output_هدف, "wb+") كـ elf_ٻاهريون_fp:
            [elf_سرو, phdr_جدول] = preprocess_elf_file(داخلا_elf)

            انگ_phdrs = elf_سرو.e_phnum
            انگ_shdrs = elf_سرو.e_shnum
            phdr_سائيز = elf_سرو.e_phentsize

            إذا انگ_phdrs == 0 أو انگ_phdrs > MAX_PHDR_ڳڻپ:
                ارفع RuntimeError("Input ELF has invalid number of program headers.")

            إذا elf_سرو.e_shoff == 0 أو انگ_shdrs == 0:
                ارفع RuntimeError("Input ELF does not contain a section header table.")

            shdr_جدول_داخلا = process_section_سرو(داخلا_elf)
            shdr_جدول_output = process_section_سرو(داخلا_elf)

            آخري_phdr = phdr_جدول[انگ_phdrs - 1]
            إذا آخري_phdr.p_memsz == SWIV_SEG_سائيز_MIN و verify_swiv_seg(elf_اندريون_fp, آخري_phdr.p_وٿي):
                ارفع RuntimeError("A SWIV segment is detected in the input ELF.")

            آخري_shdr = shdr_جدول_داخلا[elf_سرو.e_shnum - 1]
            إذا آخري_shdr.sh_سائيز == SWIV_SEG_سائيز_MIN و verify_swiv_seg(elf_اندريون_fp, آخري_shdr.sh_وٿي):
                ارفع RuntimeError("A SWIV section is detected in the input ELF.")

            إذا is_elf64(elf_سرو):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.سائيز)
            وإلا:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.سائيز)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_سائيز = SWIV_SEG_سائيز_MIN
            swiv_shdr.sh_وٿي = elf_سرو.e_shoff + (elf_سرو.e_shnum + 1) * elf_سرو.e_shentsize
            shdr_جدول_output.append(swiv_shdr)
            elf_سرو.e_shnum += 1

            إذا elf_سرو.e_phoff != 0:
                لكل phdr في phdr_جدول:
                    file_copy_وٿي(elf_اندريون_fp, phdr.p_وٿي, elf_ٻاهريون_fp, phdr.p_وٿي, phdr.p_filesz)

            elf_ٻاهريون_fp.seek(0)
            elf_ٻاهريون_fp.اكتب(elf_سرو.get_packed_معلومات())

            phdr_شروعات = elf_سرو.e_phoff
            لكل phdr في phdr_جدول:
                elf_ٻاهريون_fp.seek(phdr_شروعات)
                elf_ٻاهريون_fp.اكتب(phdr.get_packed_معلومات())
                phdr_شروعات += phdr_سائيز

            لكل shdr في shdr_جدول_داخلا[:انگ_shdrs]:
                إذا shdr.sh_type == SHT_NOBITS أو shdr.sh_سائيز == 0:
                    استمر
                file_copy_وٿي(elf_اندريون_fp, shdr.sh_وٿي, elf_ٻاهريون_fp, shdr.sh_وٿي, shdr.sh_سائيز)

            shdr_شروعات = elf_سرو.e_shoff
            لكل shdr في shdr_جدول_output:
                elf_ٻاهريون_fp.seek(shdr_شروعات)
                elf_ٻاهريون_fp.اكتب(shdr.get_packed_معلومات())
                shdr_شروعات += elf_سرو.e_shentsize

            elf_ٻاهريون_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.چڪري_جاچ = چڪري_جاچ_32_calculate(elf_ٻاهريون_fp, phdr_جدول)
            elf_ٻاهريون_fp.seek(swiv_shdr.sh_وٿي)
            elf_ٻاهريون_fp.اكتب(swiv_seg.get_packed_معلومات())

        إذا عارضي_output آهي ليس ڪجهه_نه:
            os.replace(عارضي_output, output_elf)
            عارضي_output = ڪجهه_نه

        أرجع 0
    أخيرا:
        إذا عارضي_output آهي ليس ڪجهه_نه و os.رستو.exists(عارضي_output):
            os.unlink(عارضي_output)


وصف مکيه(argv):
    parser = argparse.دليل_Parser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.جوڙيو_دليل("-i", "--input", dest="input_file", metavar="INPUT", required=سچ)
    parser.جوڙيو_دليل("-o", "--output", dest="output_file", metavar="OUTPUT", required=سچ)
    parser.جوڙيو_دليل("-v", "--verbose", action="store_true", help="show calculated CRC")
    دليلون_2 = parser.parse_دليلون(argv)

    logging.basicConfig(
        level=logging.INFO إذا دليلون_2.verbose وإلا logging.WARNING,
        format="%(message)s",
    )

    حاول:
        أرجع swiv_generate(دليلون_2.داخلا_file, دليلون_2.output_file)
    استثناء RuntimeError كـ exc:
        logger.error("ERROR: %s", exc)
        أرجع 2


إذا __name__ == "__main__":
    sys.exit(مکيه(sys.argv[1:]))
