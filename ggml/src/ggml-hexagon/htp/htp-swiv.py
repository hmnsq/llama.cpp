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
ELF_CLASS_شاخص = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_نسخه_شاخص = 6
EV_CURRENT = 1
MAX_PHDR_شمېر = 100

بارول_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_اندازه_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


صنف Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    تعریف __init__(self, معلومات):
        unpacked_معلومات = (Elf_Ehdr_common.s).unpack(معلومات)
        self.e_ident = unpacked_معلومات[0]
        self.e_type = unpacked_معلومات[1]
        self.e_machine = unpacked_معلومات[2]
        self.e_نسخه = unpacked_معلومات[3]


صنف Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    تعریف __init__(self, معلومات):
        unpacked_معلومات = (Elf32_Ehdr.s).unpack(معلومات)
        self.e_ident = unpacked_معلومات[0]
        self.e_type = unpacked_معلومات[1]
        self.e_machine = unpacked_معلومات[2]
        self.e_نسخه = unpacked_معلومات[3]
        self.e_ثبت = unpacked_معلومات[4]
        self.e_phoff = unpacked_معلومات[5]
        self.e_shoff = unpacked_معلومات[6]
        self.e_flags = unpacked_معلومات[7]
        self.e_ehsize = unpacked_معلومات[8]
        self.e_phentsize = unpacked_معلومات[9]
        self.e_phnum = unpacked_معلومات[10]
        self.e_shentsize = unpacked_معلومات[11]
        self.e_shnum = unpacked_معلومات[12]
        self.e_shstrndx = unpacked_معلومات[13]

    تعریف get_packed_معلومات(self):
        ارزښتونه = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_نسخه,
                  self.e_ثبت,
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

        أرجع (Elf32_Ehdr.s).pack(*ارزښتونه)


صنف Elf32_Phdr:
    s = struct.Struct('I' * 8)

    تعریف __init__(self, معلومات):
        unpacked_معلومات = (Elf32_Phdr.s).unpack(معلومات)
        self.p_type = unpacked_معلومات[0]
        self.p_دننه_واټن = unpacked_معلومات[1]
        self.p_vaddr = unpacked_معلومات[2]
        self.p_paddr = unpacked_معلومات[3]
        self.p_filesz = unpacked_معلومات[4]
        self.p_memsz = unpacked_معلومات[5]
        self.p_flags = unpacked_معلومات[6]
        self.p_align = unpacked_معلومات[7]

    تعریف get_packed_معلومات(self):
        ارزښتونه = [self.p_type,
                  self.p_دننه_واټن,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        أرجع (Elf32_Phdr.s).pack(*ارزښتونه)


صنف Elf32_Shdr:
    s = struct.Struct('I' * 10)

    تعریف __init__(self, معلومات):
        unpacked_معلومات = (Elf32_Shdr.s).unpack(معلومات)
        self.sh_name = unpacked_معلومات[0]
        self.sh_type = unpacked_معلومات[1]
        self.sh_flags = unpacked_معلومات[2]
        self.sh_addr = unpacked_معلومات[3]
        self.sh_دننه_واټن = unpacked_معلومات[4]
        self.sh_اندازه = unpacked_معلومات[5]
        self.sh_link = unpacked_معلومات[6]
        self.sh_info = unpacked_معلومات[7]
        self.sh_addralign = unpacked_معلومات[8]
        self.sh_entsize = unpacked_معلومات[9]
        self.sh_name_str = ""

    تعریف get_packed_معلومات(self):
        ارزښتونه = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_دننه_واټن,
                  self.sh_اندازه,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        أرجع (Elf32_Shdr.s).pack(*ارزښتونه)


صنف Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    تعریف __init__(self, معلومات):
        unpacked_معلومات = (Elf64_Ehdr.s).unpack(معلومات)
        self.e_ident = unpacked_معلومات[0]
        self.e_type = unpacked_معلومات[1]
        self.e_machine = unpacked_معلومات[2]
        self.e_نسخه = unpacked_معلومات[3]
        self.e_ثبت = unpacked_معلومات[4]
        self.e_phoff = unpacked_معلومات[5]
        self.e_shoff = unpacked_معلومات[6]
        self.e_flags = unpacked_معلومات[7]
        self.e_ehsize = unpacked_معلومات[8]
        self.e_phentsize = unpacked_معلومات[9]
        self.e_phnum = unpacked_معلومات[10]
        self.e_shentsize = unpacked_معلومات[11]
        self.e_shnum = unpacked_معلومات[12]
        self.e_shstrndx = unpacked_معلومات[13]

    تعریف get_packed_معلومات(self):
        ارزښتونه = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_نسخه,
                  self.e_ثبت,
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

        أرجع (Elf64_Ehdr.s).pack(*ارزښتونه)


صنف Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    تعریف __init__(self, معلومات):
        unpacked_معلومات = (Elf64_Phdr.s).unpack(معلومات)
        self.p_type = unpacked_معلومات[0]
        self.p_flags = unpacked_معلومات[1]
        self.p_دننه_واټن = unpacked_معلومات[2]
        self.p_vaddr = unpacked_معلومات[3]
        self.p_paddr = unpacked_معلومات[4]
        self.p_filesz = unpacked_معلومات[5]
        self.p_memsz = unpacked_معلومات[6]
        self.p_align = unpacked_معلومات[7]

    تعریف get_packed_معلومات(self):
        ارزښتونه = [self.p_type,
                  self.p_flags,
                  self.p_دننه_واټن,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        أرجع (Elf64_Phdr.s).pack(*ارزښتونه)


صنف Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    تعریف __init__(self, معلومات):
        unpacked_معلومات = (Elf64_Shdr.s).unpack(معلومات)
        self.sh_name = unpacked_معلومات[0]
        self.sh_type = unpacked_معلومات[1]
        self.sh_flags = unpacked_معلومات[2]
        self.sh_addr = unpacked_معلومات[3]
        self.sh_دننه_واټن = unpacked_معلومات[4]
        self.sh_اندازه = unpacked_معلومات[5]
        self.sh_link = unpacked_معلومات[6]
        self.sh_info = unpacked_معلومات[7]
        self.sh_addralign = unpacked_معلومات[8]
        self.sh_entsize = unpacked_معلومات[9]
        self.sh_name_str = ""

    تعریف get_packed_معلومات(self):
        ارزښتونه = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_دننه_واټن,
                  self.sh_اندازه,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        أرجع (Elf64_Shdr.s).pack(*ارزښتونه)


صنف SWIV_Seg:
    s = struct.Struct('4sIQ')

    تعریف __init__(self):
        self.magic = SWIV_MAGIC
        self.دوراني_ازموینه = 0
        self.padding = 0

    تعریف get_packed_معلومات(self):
        ارزښتونه = [self.magic,
                  self.دوراني_ازموینه,
                  self.padding
                  ]

        أرجع (SWIV_Seg.s).pack(*ارزښتونه)


تعریف is_elf64(elf_سرلیک):
    أرجع elf_سرلیک.e_ident[ELF_CLASS_شاخص] == ELFCLASS64


تعریف verify_elf_سرلیک(elf_سرلیک):
    ident = elf_سرلیک.e_ident
    أرجع (ident.startswith(ELF_MAGIC)
            و ident[ELF_CLASS_شاخص] في (ELFCLASS32, ELFCLASS64)
            و ident[ELF_نسخه_شاخص] == EV_CURRENT)


تعریف file_copy_دننه_واټن(دننه_fp, دننه_off, بهر_fp, بهر_off, عدد_بایټونه):
    دننه_fp.seek(دننه_off)
    read_دننه = دننه_fp.اقرأ(عدد_بایټونه)
    بهر_fp.seek(بهر_off)
    بهر_fp.اكتب(read_دننه)

    أرجع عدد_بایټونه


تعریف preprocess_elf_file(elf_file_name):
    سره open(elf_file_name, 'rb') كـ elf_fp:
        elf_سرلیک = Elf_Ehdr_common(elf_fp.اقرأ(Elf_Ehdr_common.s.اندازه))

        إذا verify_elf_سرلیک(elf_سرلیک) دی خطأ:
            ارفع RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        إذا is_elf64(elf_سرلیک):
            elf_سرلیک = Elf64_Ehdr(elf_fp.اقرأ(Elf64_Ehdr.s.اندازه))
        وإلا:
            elf_سرلیک = Elf32_Ehdr(elf_fp.اقرأ(Elf32_Ehdr.s.اندازه))

        phdr_جدول = []

        phdr_اندازه = elf_سرلیک.e_phentsize

        file_دننه_واټن = elf_سرلیک.e_phoff
        elf_fp.seek(file_دننه_واټن)

        لكل _ في لړ(elf_سرلیک.e_phnum):
            إذا is_elf64(elf_سرلیک):
                phdr_جدول.append(Elf64_Phdr(elf_fp.اقرأ(phdr_اندازه)))
            وإلا:
                phdr_جدول.append(Elf32_Phdr(elf_fp.اقرأ(phdr_اندازه)))

    أرجع [elf_سرلیک, phdr_جدول]


تعریف process_section_سرلیک(elf_file_name):
    سره open(elf_file_name, 'rb') كـ elf_fp:
        elf_سرلیک = Elf_Ehdr_common(elf_fp.اقرأ(Elf_Ehdr_common.s.اندازه))

        إذا verify_elf_سرلیک(elf_سرلیک) دی خطأ:
            ارفع RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        إذا is_elf64(elf_سرلیک):
            elf_سرلیک = Elf64_Ehdr(elf_fp.اقرأ(Elf64_Ehdr.s.اندازه))
        وإلا:
            elf_سرلیک = Elf32_Ehdr(elf_fp.اقرأ(Elf32_Ehdr.s.اندازه))

        shdr_جدول = []

        shdr_اندازه = elf_سرلیک.e_shentsize
        elf_fp.seek(elf_سرلیک.e_shoff)

        لكل _ في لړ(elf_سرلیک.e_shnum):
            إذا is_elf64(elf_سرلیک):
                shdr_جدول.append(Elf64_Shdr(elf_fp.اقرأ(shdr_اندازه)))
            وإلا:
                shdr_جدول.append(Elf32_Shdr(elf_fp.اقرأ(shdr_اندازه)))

    أرجع shdr_جدول


تعریف دوراني_ازموینه_32_update_file(elf_fp, دننه_واټن, اندازه, دوراني_ازموینه):
    elf_fp.seek(دننه_واټن)
    پاتې_شوی = اندازه
    طالما پاتې_شوی > 0:
        chunk = elf_fp.اقرأ(min(1024 * 1024, پاتې_شوی))
        إذا ليس chunk:
            ارفع RuntimeError("Unexpected EOF while calculating CRC")
        دوراني_ازموینه = zlib.دوراني_ازموینه_32(chunk, دوراني_ازموینه)
        پاتې_شوی -= اوږدوالی(chunk)
    أرجع دوراني_ازموینه


تعریف دوراني_ازموینه_32_update_zeros(اندازه, دوراني_ازموینه):
    zero_chunk = b'\x00' * min(1024 * 1024, اندازه)
    پاتې_شوی = اندازه
    طالما پاتې_شوی > 0:
        chunk = zero_chunk[:min(اوږدوالی(zero_chunk), پاتې_شوی)]
        دوراني_ازموینه = zlib.دوراني_ازموینه_32(chunk, دوراني_ازموینه)
        پاتې_شوی -= اوږدوالی(chunk)
    أرجع دوراني_ازموینه


تعریف دوراني_ازموینه_32_calculate(elf_fp, phdr_جدول):
    دوراني_ازموینه = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    لكل curr_phdr في phdr_جدول:
        إذا curr_phdr.p_type != بارول_TYPE:
            استمر

        دوراني_ازموینه = دوراني_ازموینه_32_update_file(elf_fp, curr_phdr.p_دننه_واټن, curr_phdr.p_filesz, دوراني_ازموینه)

        zero_اندازه = curr_phdr.p_memsz - curr_phdr.p_filesz
        إذا zero_اندازه > 0:
            دوراني_ازموینه = دوراني_ازموینه_32_update_zeros(zero_اندازه, دوراني_ازموینه)

    دوراني_ازموینه &= 0xffffffff
    logger.info("calculated CRC: %#x", دوراني_ازموینه)

    أرجع دوراني_ازموینه


تعریف verify_swiv_seg(elf_دننه_fp, دننه_واټن):

    elf_دننه_fp.seek(دننه_واټن)
    swiv_seg = SWIV_Seg()
    unpacked_معلومات = (SWIV_Seg.s).unpack(elf_دننه_fp.اقرأ(SWIV_SEG_اندازه_MIN))
    swiv_seg.magic = unpacked_معلومات[0]

    أرجع swiv_seg.magic == SWIV_MAGIC


تعریف same_output_file(ننوت_elf, output_elf):
    حاول:
        أرجع os.لار.exists(output_elf) و os.لار.samefile(ننوت_elf, output_elf)
    استثناء File_Not_وموندل_Error:
        أرجع خطأ


تعریف make_output_موخه(ننوت_elf, output_elf):
    إذا ليس same_output_file(ننوت_elf, output_elf):
        أرجع output_elf, هیڅ

    output_dir = os.لار.dirname(os.لار.abspath(output_elf))
    output_بنسټ = os.لار.basename(output_elf)
    fd, لنډمهاله_output = tempfile.mkstemp(مختاړی=output_بنسټ + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(ننوت_elf, لنډمهاله_output)
    أرجع لنډمهاله_output, لنډمهاله_output


تعریف swiv_generate(ننوت_elf, output_elf):
    output_موخه, لنډمهاله_output = make_output_موخه(ننوت_elf, output_elf)

    حاول:
        سره open(ننوت_elf, "rb") كـ elf_دننه_fp, open(output_موخه, "wb+") كـ elf_بهر_fp:
            [elf_سرلیک, phdr_جدول] = preprocess_elf_file(ننوت_elf)

            عدد_phdrs = elf_سرلیک.e_phnum
            عدد_shdrs = elf_سرلیک.e_shnum
            phdr_اندازه = elf_سرلیک.e_phentsize

            إذا عدد_phdrs == 0 أو عدد_phdrs > MAX_PHDR_شمېر:
                ارفع RuntimeError("Input ELF has invalid number of program headers.")

            إذا elf_سرلیک.e_shoff == 0 أو عدد_shdrs == 0:
                ارفع RuntimeError("Input ELF does not contain a section header table.")

            shdr_جدول_ننوت = process_section_سرلیک(ننوت_elf)
            shdr_جدول_output = process_section_سرلیک(ننوت_elf)

            وروستی_phdr = phdr_جدول[عدد_phdrs - 1]
            إذا وروستی_phdr.p_memsz == SWIV_SEG_اندازه_MIN و verify_swiv_seg(elf_دننه_fp, وروستی_phdr.p_دننه_واټن):
                ارفع RuntimeError("A SWIV segment is detected in the input ELF.")

            وروستی_shdr = shdr_جدول_ننوت[elf_سرلیک.e_shnum - 1]
            إذا وروستی_shdr.sh_اندازه == SWIV_SEG_اندازه_MIN و verify_swiv_seg(elf_دننه_fp, وروستی_shdr.sh_دننه_واټن):
                ارفع RuntimeError("A SWIV section is detected in the input ELF.")

            إذا is_elf64(elf_سرلیک):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.اندازه)
            وإلا:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.اندازه)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_اندازه = SWIV_SEG_اندازه_MIN
            swiv_shdr.sh_دننه_واټن = elf_سرلیک.e_shoff + (elf_سرلیک.e_shnum + 1) * elf_سرلیک.e_shentsize
            shdr_جدول_output.append(swiv_shdr)
            elf_سرلیک.e_shnum += 1

            إذا elf_سرلیک.e_phoff != 0:
                لكل phdr في phdr_جدول:
                    file_copy_دننه_واټن(elf_دننه_fp, phdr.p_دننه_واټن, elf_بهر_fp, phdr.p_دننه_واټن, phdr.p_filesz)

            elf_بهر_fp.seek(0)
            elf_بهر_fp.اكتب(elf_سرلیک.get_packed_معلومات())

            phdr_پیل = elf_سرلیک.e_phoff
            لكل phdr في phdr_جدول:
                elf_بهر_fp.seek(phdr_پیل)
                elf_بهر_fp.اكتب(phdr.get_packed_معلومات())
                phdr_پیل += phdr_اندازه

            لكل shdr في shdr_جدول_ننوت[:عدد_shdrs]:
                إذا shdr.sh_type == SHT_NOBITS أو shdr.sh_اندازه == 0:
                    استمر
                file_copy_دننه_واټن(elf_دننه_fp, shdr.sh_دننه_واټن, elf_بهر_fp, shdr.sh_دننه_واټن, shdr.sh_اندازه)

            shdr_پیل = elf_سرلیک.e_shoff
            لكل shdr في shdr_جدول_output:
                elf_بهر_fp.seek(shdr_پیل)
                elf_بهر_fp.اكتب(shdr.get_packed_معلومات())
                shdr_پیل += elf_سرلیک.e_shentsize

            elf_بهر_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.دوراني_ازموینه = دوراني_ازموینه_32_calculate(elf_بهر_fp, phdr_جدول)
            elf_بهر_fp.seek(swiv_shdr.sh_دننه_واټن)
            elf_بهر_fp.اكتب(swiv_seg.get_packed_معلومات())

        إذا لنډمهاله_output دی ليس هیڅ:
            os.replace(لنډمهاله_output, output_elf)
            لنډمهاله_output = هیڅ

        أرجع 0
    أخيرا:
        إذا لنډمهاله_output دی ليس هیڅ و os.لار.exists(لنډمهاله_output):
            os.unlink(لنډمهاله_output)


تعریف اصلي(argv):
    parser = argparse.دلیل_Parser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.جمع_کړه_دلیل("-i", "--input", dest="input_file", metavar="INPUT", required=ریښتیا)
    parser.جمع_کړه_دلیل("-o", "--output", dest="output_file", metavar="OUTPUT", required=ریښتیا)
    parser.جمع_کړه_دلیل("-v", "--verbose", action="store_true", help="show calculated CRC")
    دلیلونه_2 = parser.parse_دلیلونه(argv)

    logging.basicConfig(
        level=logging.INFO إذا دلیلونه_2.verbose وإلا logging.WARNING,
        format="%(message)s",
    )

    حاول:
        أرجع swiv_generate(دلیلونه_2.ننوت_file, دلیلونه_2.output_file)
    استثناء RuntimeError كـ exc:
        logger.error("ERROR: %s", exc)
        أرجع 2


إذا __name__ == "__main__":
    sys.exit(اصلي(sys.argv[1:]))
