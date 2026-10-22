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
ELF_CLASS_اشاریہ = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_نسخہ_اشاریہ = 6
EV_CURRENT = 1
MAX_PHDR_گنتی = 100

لدو_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_ناپ_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


صنف Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    تعریف __init__(self, انکڑے):
        unpacked_انکڑے = (Elf_Ehdr_common.s).unpack(انکڑے)
        self.e_ident = unpacked_انکڑے[0]
        self.e_type = unpacked_انکڑے[1]
        self.e_machine = unpacked_انکڑے[2]
        self.e_نسخہ = unpacked_انکڑے[3]


صنف Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    تعریف __init__(self, انکڑے):
        unpacked_انکڑے = (Elf32_Ehdr.s).unpack(انکڑے)
        self.e_ident = unpacked_انکڑے[0]
        self.e_type = unpacked_انکڑے[1]
        self.e_machine = unpacked_انکڑے[2]
        self.e_نسخہ = unpacked_انکڑے[3]
        self.e_اندراج = unpacked_انکڑے[4]
        self.e_phoff = unpacked_انکڑے[5]
        self.e_shoff = unpacked_انکڑے[6]
        self.e_flags = unpacked_انکڑے[7]
        self.e_ehsize = unpacked_انکڑے[8]
        self.e_phentsize = unpacked_انکڑے[9]
        self.e_phnum = unpacked_انکڑے[10]
        self.e_shentsize = unpacked_انکڑے[11]
        self.e_shnum = unpacked_انکڑے[12]
        self.e_shstrndx = unpacked_انکڑے[13]

    تعریف get_packed_انکڑے(self):
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

        أرجع (Elf32_Ehdr.s).pack(*اقدار)


صنف Elf32_Phdr:
    s = struct.Struct('I' * 8)

    تعریف __init__(self, انکڑے):
        unpacked_انکڑے = (Elf32_Phdr.s).unpack(انکڑے)
        self.p_type = unpacked_انکڑے[0]
        self.p_وتھ = unpacked_انکڑے[1]
        self.p_vaddr = unpacked_انکڑے[2]
        self.p_paddr = unpacked_انکڑے[3]
        self.p_filesz = unpacked_انکڑے[4]
        self.p_memsz = unpacked_انکڑے[5]
        self.p_flags = unpacked_انکڑے[6]
        self.p_align = unpacked_انکڑے[7]

    تعریف get_packed_انکڑے(self):
        اقدار = [self.p_type,
                  self.p_وتھ,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        أرجع (Elf32_Phdr.s).pack(*اقدار)


صنف Elf32_Shdr:
    s = struct.Struct('I' * 10)

    تعریف __init__(self, انکڑے):
        unpacked_انکڑے = (Elf32_Shdr.s).unpack(انکڑے)
        self.sh_name = unpacked_انکڑے[0]
        self.sh_type = unpacked_انکڑے[1]
        self.sh_flags = unpacked_انکڑے[2]
        self.sh_addr = unpacked_انکڑے[3]
        self.sh_وتھ = unpacked_انکڑے[4]
        self.sh_ناپ = unpacked_انکڑے[5]
        self.sh_link = unpacked_انکڑے[6]
        self.sh_info = unpacked_انکڑے[7]
        self.sh_addralign = unpacked_انکڑے[8]
        self.sh_entsize = unpacked_انکڑے[9]
        self.sh_name_str = ""

    تعریف get_packed_انکڑے(self):
        اقدار = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_وتھ,
                  self.sh_ناپ,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        أرجع (Elf32_Shdr.s).pack(*اقدار)


صنف Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    تعریف __init__(self, انکڑے):
        unpacked_انکڑے = (Elf64_Ehdr.s).unpack(انکڑے)
        self.e_ident = unpacked_انکڑے[0]
        self.e_type = unpacked_انکڑے[1]
        self.e_machine = unpacked_انکڑے[2]
        self.e_نسخہ = unpacked_انکڑے[3]
        self.e_اندراج = unpacked_انکڑے[4]
        self.e_phoff = unpacked_انکڑے[5]
        self.e_shoff = unpacked_انکڑے[6]
        self.e_flags = unpacked_انکڑے[7]
        self.e_ehsize = unpacked_انکڑے[8]
        self.e_phentsize = unpacked_انکڑے[9]
        self.e_phnum = unpacked_انکڑے[10]
        self.e_shentsize = unpacked_انکڑے[11]
        self.e_shnum = unpacked_انکڑے[12]
        self.e_shstrndx = unpacked_انکڑے[13]

    تعریف get_packed_انکڑے(self):
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

        أرجع (Elf64_Ehdr.s).pack(*اقدار)


صنف Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    تعریف __init__(self, انکڑے):
        unpacked_انکڑے = (Elf64_Phdr.s).unpack(انکڑے)
        self.p_type = unpacked_انکڑے[0]
        self.p_flags = unpacked_انکڑے[1]
        self.p_وتھ = unpacked_انکڑے[2]
        self.p_vaddr = unpacked_انکڑے[3]
        self.p_paddr = unpacked_انکڑے[4]
        self.p_filesz = unpacked_انکڑے[5]
        self.p_memsz = unpacked_انکڑے[6]
        self.p_align = unpacked_انکڑے[7]

    تعریف get_packed_انکڑے(self):
        اقدار = [self.p_type,
                  self.p_flags,
                  self.p_وتھ,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        أرجع (Elf64_Phdr.s).pack(*اقدار)


صنف Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    تعریف __init__(self, انکڑے):
        unpacked_انکڑے = (Elf64_Shdr.s).unpack(انکڑے)
        self.sh_name = unpacked_انکڑے[0]
        self.sh_type = unpacked_انکڑے[1]
        self.sh_flags = unpacked_انکڑے[2]
        self.sh_addr = unpacked_انکڑے[3]
        self.sh_وتھ = unpacked_انکڑے[4]
        self.sh_ناپ = unpacked_انکڑے[5]
        self.sh_link = unpacked_انکڑے[6]
        self.sh_info = unpacked_انکڑے[7]
        self.sh_addralign = unpacked_انکڑے[8]
        self.sh_entsize = unpacked_انکڑے[9]
        self.sh_name_str = ""

    تعریف get_packed_انکڑے(self):
        اقدار = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_وتھ,
                  self.sh_ناپ,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        أرجع (Elf64_Shdr.s).pack(*اقدار)


صنف SWIV_Seg:
    s = struct.Struct('4sIQ')

    تعریف __init__(self):
        self.magic = SWIV_MAGIC
        self.چکری_جانچ = 0
        self.padding = 0

    تعریف get_packed_انکڑے(self):
        اقدار = [self.magic,
                  self.چکری_جانچ,
                  self.padding
                  ]

        أرجع (SWIV_Seg.s).pack(*اقدار)


تعریف is_elf64(elf_سرخی):
    أرجع elf_سرخی.e_ident[ELF_CLASS_اشاریہ] == ELFCLASS64


تعریف verify_elf_سرخی(elf_سرخی):
    ident = elf_سرخی.e_ident
    أرجع (ident.startswith(ELF_MAGIC)
            و ident[ELF_CLASS_اشاریہ] في (ELFCLASS32, ELFCLASS64)
            و ident[ELF_نسخہ_اشاریہ] == EV_CURRENT)


تعریف file_copy_وتھ(وچ_fp, وچ_off, باہرلا_fp, باہرلا_off, عدد_بائٹاں):
    وچ_fp.seek(وچ_off)
    read_وچ = وچ_fp.اقرأ(عدد_بائٹاں)
    باہرلا_fp.seek(باہرلا_off)
    باہرلا_fp.اكتب(read_وچ)

    أرجع عدد_بائٹاں


تعریف preprocess_elf_file(elf_file_name):
    نال open(elf_file_name, 'rb') كـ elf_fp:
        elf_سرخی = Elf_Ehdr_common(elf_fp.اقرأ(Elf_Ehdr_common.s.ناپ))

        إذا verify_elf_سرخی(elf_سرخی) اے خطأ:
            ارفع RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        إذا is_elf64(elf_سرخی):
            elf_سرخی = Elf64_Ehdr(elf_fp.اقرأ(Elf64_Ehdr.s.ناپ))
        وإلا:
            elf_سرخی = Elf32_Ehdr(elf_fp.اقرأ(Elf32_Ehdr.s.ناپ))

        phdr_جدول = []

        phdr_ناپ = elf_سرخی.e_phentsize

        file_وتھ = elf_سرخی.e_phoff
        elf_fp.seek(file_وتھ)

        لكل _ في سلسلہ(elf_سرخی.e_phnum):
            إذا is_elf64(elf_سرخی):
                phdr_جدول.append(Elf64_Phdr(elf_fp.اقرأ(phdr_ناپ)))
            وإلا:
                phdr_جدول.append(Elf32_Phdr(elf_fp.اقرأ(phdr_ناپ)))

    أرجع [elf_سرخی, phdr_جدول]


تعریف process_section_سرخی(elf_file_name):
    نال open(elf_file_name, 'rb') كـ elf_fp:
        elf_سرخی = Elf_Ehdr_common(elf_fp.اقرأ(Elf_Ehdr_common.s.ناپ))

        إذا verify_elf_سرخی(elf_سرخی) اے خطأ:
            ارفع RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        إذا is_elf64(elf_سرخی):
            elf_سرخی = Elf64_Ehdr(elf_fp.اقرأ(Elf64_Ehdr.s.ناپ))
        وإلا:
            elf_سرخی = Elf32_Ehdr(elf_fp.اقرأ(Elf32_Ehdr.s.ناپ))

        shdr_جدول = []

        shdr_ناپ = elf_سرخی.e_shentsize
        elf_fp.seek(elf_سرخی.e_shoff)

        لكل _ في سلسلہ(elf_سرخی.e_shnum):
            إذا is_elf64(elf_سرخی):
                shdr_جدول.append(Elf64_Shdr(elf_fp.اقرأ(shdr_ناپ)))
            وإلا:
                shdr_جدول.append(Elf32_Shdr(elf_fp.اقرأ(shdr_ناپ)))

    أرجع shdr_جدول


تعریف چکری_جانچ_32_update_file(elf_fp, وتھ, ناپ, چکری_جانچ):
    elf_fp.seek(وتھ)
    بچیا = ناپ
    طالما بچیا > 0:
        chunk = elf_fp.اقرأ(min(1024 * 1024, بچیا))
        إذا ليس chunk:
            ارفع RuntimeError("Unexpected EOF while calculating CRC")
        چکری_جانچ = zlib.چکری_جانچ_32(chunk, چکری_جانچ)
        بچیا -= لمبائی(chunk)
    أرجع چکری_جانچ


تعریف چکری_جانچ_32_update_zeros(ناپ, چکری_جانچ):
    zero_chunk = b'\x00' * min(1024 * 1024, ناپ)
    بچیا = ناپ
    طالما بچیا > 0:
        chunk = zero_chunk[:min(لمبائی(zero_chunk), بچیا)]
        چکری_جانچ = zlib.چکری_جانچ_32(chunk, چکری_جانچ)
        بچیا -= لمبائی(chunk)
    أرجع چکری_جانچ


تعریف چکری_جانچ_32_calculate(elf_fp, phdr_جدول):
    چکری_جانچ = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    لكل curr_phdr في phdr_جدول:
        إذا curr_phdr.p_type != لدو_TYPE:
            استمر

        چکری_جانچ = چکری_جانچ_32_update_file(elf_fp, curr_phdr.p_وتھ, curr_phdr.p_filesz, چکری_جانچ)

        zero_ناپ = curr_phdr.p_memsz - curr_phdr.p_filesz
        إذا zero_ناپ > 0:
            چکری_جانچ = چکری_جانچ_32_update_zeros(zero_ناپ, چکری_جانچ)

    چکری_جانچ &= 0xffffffff
    logger.info("calculated CRC: %#x", چکری_جانچ)

    أرجع چکری_جانچ


تعریف verify_swiv_seg(elf_وچ_fp, وتھ):

    elf_وچ_fp.seek(وتھ)
    swiv_seg = SWIV_Seg()
    unpacked_انکڑے = (SWIV_Seg.s).unpack(elf_وچ_fp.اقرأ(SWIV_SEG_ناپ_MIN))
    swiv_seg.magic = unpacked_انکڑے[0]

    أرجع swiv_seg.magic == SWIV_MAGIC


تعریف same_output_file(داخلہ_elf, output_elf):
    حاول:
        أرجع os.رستہ.exists(output_elf) و os.رستہ.samefile(داخلہ_elf, output_elf)
    استثناء File_Not_لبھیا_Error:
        أرجع خطأ


تعریف make_output_نشانہ(داخلہ_elf, output_elf):
    إذا ليس same_output_file(داخلہ_elf, output_elf):
        أرجع output_elf, کجھ_نئیں

    output_dir = os.رستہ.dirname(os.رستہ.abspath(output_elf))
    output_ادھار = os.رستہ.basename(output_elf)
    fd, عارضی_output = tempfile.mkstemp(اگیتر=output_ادھار + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(داخلہ_elf, عارضی_output)
    أرجع عارضی_output, عارضی_output


تعریف swiv_generate(داخلہ_elf, output_elf):
    output_نشانہ, عارضی_output = make_output_نشانہ(داخلہ_elf, output_elf)

    حاول:
        نال open(داخلہ_elf, "rb") كـ elf_وچ_fp, open(output_نشانہ, "wb+") كـ elf_باہرلا_fp:
            [elf_سرخی, phdr_جدول] = preprocess_elf_file(داخلہ_elf)

            عدد_phdrs = elf_سرخی.e_phnum
            عدد_shdrs = elf_سرخی.e_shnum
            phdr_ناپ = elf_سرخی.e_phentsize

            إذا عدد_phdrs == 0 أو عدد_phdrs > MAX_PHDR_گنتی:
                ارفع RuntimeError("Input ELF has invalid number of program headers.")

            إذا elf_سرخی.e_shoff == 0 أو عدد_shdrs == 0:
                ارفع RuntimeError("Input ELF does not contain a section header table.")

            shdr_جدول_داخلہ = process_section_سرخی(داخلہ_elf)
            shdr_جدول_output = process_section_سرخی(داخلہ_elf)

            آخری_phdr = phdr_جدول[عدد_phdrs - 1]
            إذا آخری_phdr.p_memsz == SWIV_SEG_ناپ_MIN و verify_swiv_seg(elf_وچ_fp, آخری_phdr.p_وتھ):
                ارفع RuntimeError("A SWIV segment is detected in the input ELF.")

            آخری_shdr = shdr_جدول_داخلہ[elf_سرخی.e_shnum - 1]
            إذا آخری_shdr.sh_ناپ == SWIV_SEG_ناپ_MIN و verify_swiv_seg(elf_وچ_fp, آخری_shdr.sh_وتھ):
                ارفع RuntimeError("A SWIV section is detected in the input ELF.")

            إذا is_elf64(elf_سرخی):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.ناپ)
            وإلا:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.ناپ)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_ناپ = SWIV_SEG_ناپ_MIN
            swiv_shdr.sh_وتھ = elf_سرخی.e_shoff + (elf_سرخی.e_shnum + 1) * elf_سرخی.e_shentsize
            shdr_جدول_output.append(swiv_shdr)
            elf_سرخی.e_shnum += 1

            إذا elf_سرخی.e_phoff != 0:
                لكل phdr في phdr_جدول:
                    file_copy_وتھ(elf_وچ_fp, phdr.p_وتھ, elf_باہرلا_fp, phdr.p_وتھ, phdr.p_filesz)

            elf_باہرلا_fp.seek(0)
            elf_باہرلا_fp.اكتب(elf_سرخی.get_packed_انکڑے())

            phdr_شروع = elf_سرخی.e_phoff
            لكل phdr في phdr_جدول:
                elf_باہرلا_fp.seek(phdr_شروع)
                elf_باہرلا_fp.اكتب(phdr.get_packed_انکڑے())
                phdr_شروع += phdr_ناپ

            لكل shdr في shdr_جدول_داخلہ[:عدد_shdrs]:
                إذا shdr.sh_type == SHT_NOBITS أو shdr.sh_ناپ == 0:
                    استمر
                file_copy_وتھ(elf_وچ_fp, shdr.sh_وتھ, elf_باہرلا_fp, shdr.sh_وتھ, shdr.sh_ناپ)

            shdr_شروع = elf_سرخی.e_shoff
            لكل shdr في shdr_جدول_output:
                elf_باہرلا_fp.seek(shdr_شروع)
                elf_باہرلا_fp.اكتب(shdr.get_packed_انکڑے())
                shdr_شروع += elf_سرخی.e_shentsize

            elf_باہرلا_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.چکری_جانچ = چکری_جانچ_32_calculate(elf_باہرلا_fp, phdr_جدول)
            elf_باہرلا_fp.seek(swiv_shdr.sh_وتھ)
            elf_باہرلا_fp.اكتب(swiv_seg.get_packed_انکڑے())

        إذا عارضی_output اے ليس کجھ_نئیں:
            os.replace(عارضی_output, output_elf)
            عارضی_output = کجھ_نئیں

        أرجع 0
    أخيرا:
        إذا عارضی_output اے ليس کجھ_نئیں و os.رستہ.exists(عارضی_output):
            os.unlink(عارضی_output)


تعریف مکھ(argv):
    parser = argparse.دلیل_Parser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.جوڑو_دلیل("-i", "--input", dest="input_file", metavar="INPUT", required=سچ)
    parser.جوڑو_دلیل("-o", "--output", dest="output_file", metavar="OUTPUT", required=سچ)
    parser.جوڑو_دلیل("-v", "--verbose", action="store_true", help="show calculated CRC")
    دلیلاں_2 = parser.parse_دلیلاں(argv)

    logging.basicConfig(
        level=logging.INFO إذا دلیلاں_2.verbose وإلا logging.WARNING,
        format="%(message)s",
    )

    حاول:
        أرجع swiv_generate(دلیلاں_2.داخلہ_file, دلیلاں_2.output_file)
    استثناء RuntimeError كـ exc:
        logger.error("ERROR: %s", exc)
        أرجع 2


إذا __name__ == "__main__":
    sys.exit(مکھ(sys.argv[1:]))
