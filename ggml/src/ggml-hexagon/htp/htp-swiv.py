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
MAX_PHDR_شمار = 100

بارگیری_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_اندازه_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


صنف Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    تعریف __init__(self, داده):
        unpacked_داده = (Elf_Ehdr_common.s).unpack(داده)
        self.e_ident = unpacked_داده[0]
        self.e_type = unpacked_داده[1]
        self.e_machine = unpacked_داده[2]
        self.e_نسخه = unpacked_داده[3]


صنف Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    تعریف __init__(self, داده):
        unpacked_داده = (Elf32_Ehdr.s).unpack(داده)
        self.e_ident = unpacked_داده[0]
        self.e_type = unpacked_داده[1]
        self.e_machine = unpacked_داده[2]
        self.e_نسخه = unpacked_داده[3]
        self.e_مدخل = unpacked_داده[4]
        self.e_phoff = unpacked_داده[5]
        self.e_shoff = unpacked_داده[6]
        self.e_flags = unpacked_داده[7]
        self.e_ehsize = unpacked_داده[8]
        self.e_phentsize = unpacked_داده[9]
        self.e_phnum = unpacked_داده[10]
        self.e_shentsize = unpacked_داده[11]
        self.e_shnum = unpacked_داده[12]
        self.e_shstrndx = unpacked_داده[13]

    تعریف get_packed_داده(self):
        مقادیر = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_نسخه,
                  self.e_مدخل,
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

        أرجع (Elf32_Ehdr.s).pack(*مقادیر)


صنف Elf32_Phdr:
    s = struct.Struct('I' * 8)

    تعریف __init__(self, داده):
        unpacked_داده = (Elf32_Phdr.s).unpack(داده)
        self.p_type = unpacked_داده[0]
        self.p_جابه_جایی = unpacked_داده[1]
        self.p_vaddr = unpacked_داده[2]
        self.p_paddr = unpacked_داده[3]
        self.p_filesz = unpacked_داده[4]
        self.p_memsz = unpacked_داده[5]
        self.p_flags = unpacked_داده[6]
        self.p_align = unpacked_داده[7]

    تعریف get_packed_داده(self):
        مقادیر = [self.p_type,
                  self.p_جابه_جایی,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        أرجع (Elf32_Phdr.s).pack(*مقادیر)


صنف Elf32_Shdr:
    s = struct.Struct('I' * 10)

    تعریف __init__(self, داده):
        unpacked_داده = (Elf32_Shdr.s).unpack(داده)
        self.sh_name = unpacked_داده[0]
        self.sh_type = unpacked_داده[1]
        self.sh_flags = unpacked_داده[2]
        self.sh_addr = unpacked_داده[3]
        self.sh_جابه_جایی = unpacked_داده[4]
        self.sh_اندازه = unpacked_داده[5]
        self.sh_link = unpacked_داده[6]
        self.sh_info = unpacked_داده[7]
        self.sh_addralign = unpacked_داده[8]
        self.sh_entsize = unpacked_داده[9]
        self.sh_name_str = ""

    تعریف get_packed_داده(self):
        مقادیر = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_جابه_جایی,
                  self.sh_اندازه,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        أرجع (Elf32_Shdr.s).pack(*مقادیر)


صنف Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    تعریف __init__(self, داده):
        unpacked_داده = (Elf64_Ehdr.s).unpack(داده)
        self.e_ident = unpacked_داده[0]
        self.e_type = unpacked_داده[1]
        self.e_machine = unpacked_داده[2]
        self.e_نسخه = unpacked_داده[3]
        self.e_مدخل = unpacked_داده[4]
        self.e_phoff = unpacked_داده[5]
        self.e_shoff = unpacked_داده[6]
        self.e_flags = unpacked_داده[7]
        self.e_ehsize = unpacked_داده[8]
        self.e_phentsize = unpacked_داده[9]
        self.e_phnum = unpacked_داده[10]
        self.e_shentsize = unpacked_داده[11]
        self.e_shnum = unpacked_داده[12]
        self.e_shstrndx = unpacked_داده[13]

    تعریف get_packed_داده(self):
        مقادیر = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_نسخه,
                  self.e_مدخل,
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

        أرجع (Elf64_Ehdr.s).pack(*مقادیر)


صنف Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    تعریف __init__(self, داده):
        unpacked_داده = (Elf64_Phdr.s).unpack(داده)
        self.p_type = unpacked_داده[0]
        self.p_flags = unpacked_داده[1]
        self.p_جابه_جایی = unpacked_داده[2]
        self.p_vaddr = unpacked_داده[3]
        self.p_paddr = unpacked_داده[4]
        self.p_filesz = unpacked_داده[5]
        self.p_memsz = unpacked_داده[6]
        self.p_align = unpacked_داده[7]

    تعریف get_packed_داده(self):
        مقادیر = [self.p_type,
                  self.p_flags,
                  self.p_جابه_جایی,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        أرجع (Elf64_Phdr.s).pack(*مقادیر)


صنف Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    تعریف __init__(self, داده):
        unpacked_داده = (Elf64_Shdr.s).unpack(داده)
        self.sh_name = unpacked_داده[0]
        self.sh_type = unpacked_داده[1]
        self.sh_flags = unpacked_داده[2]
        self.sh_addr = unpacked_داده[3]
        self.sh_جابه_جایی = unpacked_داده[4]
        self.sh_اندازه = unpacked_داده[5]
        self.sh_link = unpacked_داده[6]
        self.sh_info = unpacked_داده[7]
        self.sh_addralign = unpacked_داده[8]
        self.sh_entsize = unpacked_داده[9]
        self.sh_name_str = ""

    تعریف get_packed_داده(self):
        مقادیر = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_جابه_جایی,
                  self.sh_اندازه,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        أرجع (Elf64_Shdr.s).pack(*مقادیر)


صنف SWIV_Seg:
    s = struct.Struct('4sIQ')

    تعریف __init__(self):
        self.magic = SWIV_MAGIC
        self.بررسی_چرخهای = 0
        self.padding = 0

    تعریف get_packed_داده(self):
        مقادیر = [self.magic,
                  self.بررسی_چرخهای,
                  self.padding
                  ]

        أرجع (SWIV_Seg.s).pack(*مقادیر)


تعریف is_elf64(elf_سرآیند):
    أرجع elf_سرآیند.e_ident[ELF_CLASS_شاخص] == ELFCLASS64


تعریف verify_elf_سرآیند(elf_سرآیند):
    ident = elf_سرآیند.e_ident
    أرجع (ident.startswith(ELF_MAGIC)
            و ident[ELF_CLASS_شاخص] في (ELFCLASS32, ELFCLASS64)
            و ident[ELF_نسخه_شاخص] == EV_CURRENT)


تعریف file_copy_جابه_جایی(درون_fp, درون_off, بیرون_fp, بیرون_off, رقم_بایتها):
    درون_fp.seek(درون_off)
    read_درون = درون_fp.اقرأ(رقم_بایتها)
    بیرون_fp.seek(بیرون_off)
    بیرون_fp.اكتب(read_درون)

    أرجع رقم_بایتها


تعریف preprocess_elf_file(elf_file_name):
    با open(elf_file_name, 'rb') كـ elf_fp:
        elf_سرآیند = Elf_Ehdr_common(elf_fp.اقرأ(Elf_Ehdr_common.s.اندازه))

        إذا verify_elf_سرآیند(elf_سرآیند) است خطأ:
            ارفع RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        إذا is_elf64(elf_سرآیند):
            elf_سرآیند = Elf64_Ehdr(elf_fp.اقرأ(Elf64_Ehdr.s.اندازه))
        وإلا:
            elf_سرآیند = Elf32_Ehdr(elf_fp.اقرأ(Elf32_Ehdr.s.اندازه))

        phdr_جدول = []

        phdr_اندازه = elf_سرآیند.e_phentsize

        file_جابه_جایی = elf_سرآیند.e_phoff
        elf_fp.seek(file_جابه_جایی)

        لكل _ في دامنه(elf_سرآیند.e_phnum):
            إذا is_elf64(elf_سرآیند):
                phdr_جدول.append(Elf64_Phdr(elf_fp.اقرأ(phdr_اندازه)))
            وإلا:
                phdr_جدول.append(Elf32_Phdr(elf_fp.اقرأ(phdr_اندازه)))

    أرجع [elf_سرآیند, phdr_جدول]


تعریف process_section_سرآیند(elf_file_name):
    با open(elf_file_name, 'rb') كـ elf_fp:
        elf_سرآیند = Elf_Ehdr_common(elf_fp.اقرأ(Elf_Ehdr_common.s.اندازه))

        إذا verify_elf_سرآیند(elf_سرآیند) است خطأ:
            ارفع RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        إذا is_elf64(elf_سرآیند):
            elf_سرآیند = Elf64_Ehdr(elf_fp.اقرأ(Elf64_Ehdr.s.اندازه))
        وإلا:
            elf_سرآیند = Elf32_Ehdr(elf_fp.اقرأ(Elf32_Ehdr.s.اندازه))

        shdr_جدول = []

        shdr_اندازه = elf_سرآیند.e_shentsize
        elf_fp.seek(elf_سرآیند.e_shoff)

        لكل _ في دامنه(elf_سرآیند.e_shnum):
            إذا is_elf64(elf_سرآیند):
                shdr_جدول.append(Elf64_Shdr(elf_fp.اقرأ(shdr_اندازه)))
            وإلا:
                shdr_جدول.append(Elf32_Shdr(elf_fp.اقرأ(shdr_اندازه)))

    أرجع shdr_جدول


تعریف بررسی_چرخهای_32_update_file(elf_fp, جابه_جایی, اندازه, بررسی_چرخهای):
    elf_fp.seek(جابه_جایی)
    مانده = اندازه
    طالما مانده > 0:
        chunk = elf_fp.اقرأ(min(1024 * 1024, مانده))
        إذا ليس chunk:
            ارفع RuntimeError("Unexpected EOF while calculating CRC")
        بررسی_چرخهای = zlib.بررسی_چرخهای_32(chunk, بررسی_چرخهای)
        مانده -= طول(chunk)
    أرجع بررسی_چرخهای


تعریف بررسی_چرخهای_32_update_zeros(اندازه, بررسی_چرخهای):
    zero_chunk = b'\x00' * min(1024 * 1024, اندازه)
    مانده = اندازه
    طالما مانده > 0:
        chunk = zero_chunk[:min(طول(zero_chunk), مانده)]
        بررسی_چرخهای = zlib.بررسی_چرخهای_32(chunk, بررسی_چرخهای)
        مانده -= طول(chunk)
    أرجع بررسی_چرخهای


تعریف بررسی_چرخهای_32_calculate(elf_fp, phdr_جدول):
    بررسی_چرخهای = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    لكل curr_phdr في phdr_جدول:
        إذا curr_phdr.p_type != بارگیری_TYPE:
            استمر

        بررسی_چرخهای = بررسی_چرخهای_32_update_file(elf_fp, curr_phdr.p_جابه_جایی, curr_phdr.p_filesz, بررسی_چرخهای)

        zero_اندازه = curr_phdr.p_memsz - curr_phdr.p_filesz
        إذا zero_اندازه > 0:
            بررسی_چرخهای = بررسی_چرخهای_32_update_zeros(zero_اندازه, بررسی_چرخهای)

    بررسی_چرخهای &= 0xffffffff
    logger.info("calculated CRC: %#x", بررسی_چرخهای)

    أرجع بررسی_چرخهای


تعریف verify_swiv_seg(elf_درون_fp, جابه_جایی):

    elf_درون_fp.seek(جابه_جایی)
    swiv_seg = SWIV_Seg()
    unpacked_داده = (SWIV_Seg.s).unpack(elf_درون_fp.اقرأ(SWIV_SEG_اندازه_MIN))
    swiv_seg.magic = unpacked_داده[0]

    أرجع swiv_seg.magic == SWIV_MAGIC


تعریف same_output_file(درون_داد_elf, output_elf):
    حاول:
        أرجع os.مسیر.exists(output_elf) و os.مسیر.samefile(درون_داد_elf, output_elf)
    استثناء File_Not_یافت_Error:
        أرجع خطأ


تعریف make_output_هدف(درون_داد_elf, output_elf):
    إذا ليس same_output_file(درون_داد_elf, output_elf):
        أرجع output_elf, هیچ

    output_dir = os.مسیر.dirname(os.مسیر.abspath(output_elf))
    output_پایه = os.مسیر.basename(output_elf)
    fd, موقت_output = tempfile.mkstemp(پیشوند=output_پایه + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(درون_داد_elf, موقت_output)
    أرجع موقت_output, موقت_output


تعریف swiv_generate(درون_داد_elf, output_elf):
    output_هدف, موقت_output = make_output_هدف(درون_داد_elf, output_elf)

    حاول:
        با open(درون_داد_elf, "rb") كـ elf_درون_fp, open(output_هدف, "wb+") كـ elf_بیرون_fp:
            [elf_سرآیند, phdr_جدول] = preprocess_elf_file(درون_داد_elf)

            رقم_phdrs = elf_سرآیند.e_phnum
            رقم_shdrs = elf_سرآیند.e_shnum
            phdr_اندازه = elf_سرآیند.e_phentsize

            إذا رقم_phdrs == 0 أو رقم_phdrs > MAX_PHDR_شمار:
                ارفع RuntimeError("Input ELF has invalid number of program headers.")

            إذا elf_سرآیند.e_shoff == 0 أو رقم_shdrs == 0:
                ارفع RuntimeError("Input ELF does not contain a section header table.")

            shdr_جدول_درون_داد = process_section_سرآیند(درون_داد_elf)
            shdr_جدول_output = process_section_سرآیند(درون_داد_elf)

            آخرین_phdr = phdr_جدول[رقم_phdrs - 1]
            إذا آخرین_phdr.p_memsz == SWIV_SEG_اندازه_MIN و verify_swiv_seg(elf_درون_fp, آخرین_phdr.p_جابه_جایی):
                ارفع RuntimeError("A SWIV segment is detected in the input ELF.")

            آخرین_shdr = shdr_جدول_درون_داد[elf_سرآیند.e_shnum - 1]
            إذا آخرین_shdr.sh_اندازه == SWIV_SEG_اندازه_MIN و verify_swiv_seg(elf_درون_fp, آخرین_shdr.sh_جابه_جایی):
                ارفع RuntimeError("A SWIV section is detected in the input ELF.")

            إذا is_elf64(elf_سرآیند):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.اندازه)
            وإلا:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.اندازه)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_اندازه = SWIV_SEG_اندازه_MIN
            swiv_shdr.sh_جابه_جایی = elf_سرآیند.e_shoff + (elf_سرآیند.e_shnum + 1) * elf_سرآیند.e_shentsize
            shdr_جدول_output.append(swiv_shdr)
            elf_سرآیند.e_shnum += 1

            إذا elf_سرآیند.e_phoff != 0:
                لكل phdr في phdr_جدول:
                    file_copy_جابه_جایی(elf_درون_fp, phdr.p_جابه_جایی, elf_بیرون_fp, phdr.p_جابه_جایی, phdr.p_filesz)

            elf_بیرون_fp.seek(0)
            elf_بیرون_fp.اكتب(elf_سرآیند.get_packed_داده())

            phdr_شروع = elf_سرآیند.e_phoff
            لكل phdr في phdr_جدول:
                elf_بیرون_fp.seek(phdr_شروع)
                elf_بیرون_fp.اكتب(phdr.get_packed_داده())
                phdr_شروع += phdr_اندازه

            لكل shdr في shdr_جدول_درون_داد[:رقم_shdrs]:
                إذا shdr.sh_type == SHT_NOBITS أو shdr.sh_اندازه == 0:
                    استمر
                file_copy_جابه_جایی(elf_درون_fp, shdr.sh_جابه_جایی, elf_بیرون_fp, shdr.sh_جابه_جایی, shdr.sh_اندازه)

            shdr_شروع = elf_سرآیند.e_shoff
            لكل shdr في shdr_جدول_output:
                elf_بیرون_fp.seek(shdr_شروع)
                elf_بیرون_fp.اكتب(shdr.get_packed_داده())
                shdr_شروع += elf_سرآیند.e_shentsize

            elf_بیرون_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.بررسی_چرخهای = بررسی_چرخهای_32_calculate(elf_بیرون_fp, phdr_جدول)
            elf_بیرون_fp.seek(swiv_shdr.sh_جابه_جایی)
            elf_بیرون_fp.اكتب(swiv_seg.get_packed_داده())

        إذا موقت_output است ليس هیچ:
            os.replace(موقت_output, output_elf)
            موقت_output = هیچ

        أرجع 0
    أخيرا:
        إذا موقت_output است ليس هیچ و os.مسیر.exists(موقت_output):
            os.unlink(موقت_output)


تعریف اصلی(argv):
    parser = argparse.آرگومان_Parser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.جمع_کن_آرگومان("-i", "--input", dest="input_file", metavar="INPUT", required=درست)
    parser.جمع_کن_آرگومان("-o", "--output", dest="output_file", metavar="OUTPUT", required=درست)
    parser.جمع_کن_آرگومان("-v", "--verbose", action="store_true", help="show calculated CRC")
    آرگومانها_2 = parser.parse_آرگومانها(argv)

    logging.basicConfig(
        level=logging.INFO إذا آرگومانها_2.verbose وإلا logging.WARNING,
        format="%(message)s",
    )

    حاول:
        أرجع swiv_generate(آرگومانها_2.درون_داد_file, آرگومانها_2.output_file)
    استثناء RuntimeError كـ exc:
        logger.error("ERROR: %s", exc)
        أرجع 2


إذا __name__ == "__main__":
    sys.exit(اصلی(sys.argv[1:]))
