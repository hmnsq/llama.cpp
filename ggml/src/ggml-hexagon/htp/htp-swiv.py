#!/usr/bin/env python3

దిగుమతి argparse
దిగుమతి logging
దిగుమతి os
దిగుమతి shutil
దిగుమతి struct
దిగుమతి sys
దిగుమతి tempfile
దిగుమతి zlib

ELF_MAGIC = b'\x7fELF'
ELF_CLASS_సూచీ = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_సంస్కరణ_సూచీ = 6
EV_CURRENT = 1
MAX_PHDR_లెక్క = 100

ఎక్కించు_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_కొలత_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


తరగతి Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    నిర్వచనం __init__(self, దత్తాంశం):
        unpacked_దత్తాంశం = (Elf_Ehdr_common.s).unpack(దత్తాంశం)
        self.e_ident = unpacked_దత్తాంశం[0]
        self.e_type = unpacked_దత్తాంశం[1]
        self.e_machine = unpacked_దత్తాంశం[2]
        self.e_సంస్కరణ = unpacked_దత్తాంశం[3]


తరగతి Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    నిర్వచనం __init__(self, దత్తాంశం):
        unpacked_దత్తాంశం = (Elf32_Ehdr.s).unpack(దత్తాంశం)
        self.e_ident = unpacked_దత్తాంశం[0]
        self.e_type = unpacked_దత్తాంశం[1]
        self.e_machine = unpacked_దత్తాంశం[2]
        self.e_సంస్కరణ = unpacked_దత్తాంశం[3]
        self.e_నమోదు = unpacked_దత్తాంశం[4]
        self.e_phoff = unpacked_దత్తాంశం[5]
        self.e_shoff = unpacked_దత్తాంశం[6]
        self.e_flags = unpacked_దత్తాంశం[7]
        self.e_ehsize = unpacked_దత్తాంశం[8]
        self.e_phentsize = unpacked_దత్తాంశం[9]
        self.e_phnum = unpacked_దత్తాంశం[10]
        self.e_shentsize = unpacked_దత్తాంశం[11]
        self.e_shnum = unpacked_దత్తాంశం[12]
        self.e_shstrndx = unpacked_దత్తాంశం[13]

    నిర్వచనం get_packed_దత్తాంశం(self):
        విలువలు = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_సంస్కరణ,
                  self.e_నమోదు,
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

        తిరిగి (Elf32_Ehdr.s).pack(*విలువలు)


తరగతి Elf32_Phdr:
    s = struct.Struct('I' * 8)

    నిర్వచనం __init__(self, దత్తాంశం):
        unpacked_దత్తాంశం = (Elf32_Phdr.s).unpack(దత్తాంశం)
        self.p_type = unpacked_దత్తాంశం[0]
        self.p_స్థానభ్రంశం = unpacked_దత్తాంశం[1]
        self.p_vaddr = unpacked_దత్తాంశం[2]
        self.p_paddr = unpacked_దత్తాంశం[3]
        self.p_filesz = unpacked_దత్తాంశం[4]
        self.p_memsz = unpacked_దత్తాంశం[5]
        self.p_flags = unpacked_దత్తాంశం[6]
        self.p_align = unpacked_దత్తాంశం[7]

    నిర్వచనం get_packed_దత్తాంశం(self):
        విలువలు = [self.p_type,
                  self.p_స్థానభ్రంశం,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        తిరిగి (Elf32_Phdr.s).pack(*విలువలు)


తరగతి Elf32_Shdr:
    s = struct.Struct('I' * 10)

    నిర్వచనం __init__(self, దత్తాంశం):
        unpacked_దత్తాంశం = (Elf32_Shdr.s).unpack(దత్తాంశం)
        self.sh_name = unpacked_దత్తాంశం[0]
        self.sh_type = unpacked_దత్తాంశం[1]
        self.sh_flags = unpacked_దత్తాంశం[2]
        self.sh_addr = unpacked_దత్తాంశం[3]
        self.sh_స్థానభ్రంశం = unpacked_దత్తాంశం[4]
        self.sh_కొలత = unpacked_దత్తాంశం[5]
        self.sh_link = unpacked_దత్తాంశం[6]
        self.sh_info = unpacked_దత్తాంశం[7]
        self.sh_addralign = unpacked_దత్తాంశం[8]
        self.sh_entsize = unpacked_దత్తాంశం[9]
        self.sh_name_str = ""

    నిర్వచనం get_packed_దత్తాంశం(self):
        విలువలు = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_స్థానభ్రంశం,
                  self.sh_కొలత,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        తిరిగి (Elf32_Shdr.s).pack(*విలువలు)


తరగతి Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    నిర్వచనం __init__(self, దత్తాంశం):
        unpacked_దత్తాంశం = (Elf64_Ehdr.s).unpack(దత్తాంశం)
        self.e_ident = unpacked_దత్తాంశం[0]
        self.e_type = unpacked_దత్తాంశం[1]
        self.e_machine = unpacked_దత్తాంశం[2]
        self.e_సంస్కరణ = unpacked_దత్తాంశం[3]
        self.e_నమోదు = unpacked_దత్తాంశం[4]
        self.e_phoff = unpacked_దత్తాంశం[5]
        self.e_shoff = unpacked_దత్తాంశం[6]
        self.e_flags = unpacked_దత్తాంశం[7]
        self.e_ehsize = unpacked_దత్తాంశం[8]
        self.e_phentsize = unpacked_దత్తాంశం[9]
        self.e_phnum = unpacked_దత్తాంశం[10]
        self.e_shentsize = unpacked_దత్తాంశం[11]
        self.e_shnum = unpacked_దత్తాంశం[12]
        self.e_shstrndx = unpacked_దత్తాంశం[13]

    నిర్వచనం get_packed_దత్తాంశం(self):
        విలువలు = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_సంస్కరణ,
                  self.e_నమోదు,
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

        తిరిగి (Elf64_Ehdr.s).pack(*విలువలు)


తరగతి Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    నిర్వచనం __init__(self, దత్తాంశం):
        unpacked_దత్తాంశం = (Elf64_Phdr.s).unpack(దత్తాంశం)
        self.p_type = unpacked_దత్తాంశం[0]
        self.p_flags = unpacked_దత్తాంశం[1]
        self.p_స్థానభ్రంశం = unpacked_దత్తాంశం[2]
        self.p_vaddr = unpacked_దత్తాంశం[3]
        self.p_paddr = unpacked_దత్తాంశం[4]
        self.p_filesz = unpacked_దత్తాంశం[5]
        self.p_memsz = unpacked_దత్తాంశం[6]
        self.p_align = unpacked_దత్తాంశం[7]

    నిర్వచనం get_packed_దత్తాంశం(self):
        విలువలు = [self.p_type,
                  self.p_flags,
                  self.p_స్థానభ్రంశం,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        తిరిగి (Elf64_Phdr.s).pack(*విలువలు)


తరగతి Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    నిర్వచనం __init__(self, దత్తాంశం):
        unpacked_దత్తాంశం = (Elf64_Shdr.s).unpack(దత్తాంశం)
        self.sh_name = unpacked_దత్తాంశం[0]
        self.sh_type = unpacked_దత్తాంశం[1]
        self.sh_flags = unpacked_దత్తాంశం[2]
        self.sh_addr = unpacked_దత్తాంశం[3]
        self.sh_స్థానభ్రంశం = unpacked_దత్తాంశం[4]
        self.sh_కొలత = unpacked_దత్తాంశం[5]
        self.sh_link = unpacked_దత్తాంశం[6]
        self.sh_info = unpacked_దత్తాంశం[7]
        self.sh_addralign = unpacked_దత్తాంశం[8]
        self.sh_entsize = unpacked_దత్తాంశం[9]
        self.sh_name_str = ""

    నిర్వచనం get_packed_దత్తాంశం(self):
        విలువలు = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_స్థానభ్రంశం,
                  self.sh_కొలత,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        తిరిగి (Elf64_Shdr.s).pack(*విలువలు)


తరగతి SWIV_Seg:
    s = struct.Struct('4sIQ')

    నిర్వచనం __init__(self):
        self.magic = SWIV_MAGIC
        self.చక్రీయ_తనిఖీ = 0
        self.padding = 0

    నిర్వచనం get_packed_దత్తాంశం(self):
        విలువలు = [self.magic,
                  self.చక్రీయ_తనిఖీ,
                  self.padding
                  ]

        తిరిగి (SWIV_Seg.s).pack(*విలువలు)


నిర్వచనం is_elf64(elf_శీర్షిక):
    తిరిగి elf_శీర్షిక.e_ident[ELF_CLASS_సూచీ] == ELFCLASS64


నిర్వచనం verify_elf_శీర్షిక(elf_శీర్షిక):
    ident = elf_శీర్షిక.e_ident
    తిరిగి (ident.startswith(ELF_MAGIC)
            మరియు ident[ELF_CLASS_సూచీ] లో (ELFCLASS32, ELFCLASS64)
            మరియు ident[ELF_సంస్కరణ_సూచీ] == EV_CURRENT)


నిర్వచనం file_copy_స్థానభ్రంశం(లోపల_fp, లోపల_off, బహిర్గతం_fp, బహిర్గతం_off, అంకె_బైట్లు):
    లోపల_fp.seek(లోపల_off)
    read_లోపల = లోపల_fp.పఠ(అంకె_బైట్లు)
    బహిర్గతం_fp.seek(బహిర్గతం_off)
    బహిర్గతం_fp.లిఖ(read_లోపల)

    తిరిగి అంకె_బైట్లు


నిర్వచనం preprocess_elf_file(elf_file_name):
    తోడుగా open(elf_file_name, 'rb') ఇవ elf_fp:
        elf_శీర్షిక = Elf_Ehdr_common(elf_fp.పఠ(Elf_Ehdr_common.s.కొలత))

        ఒకవేళ verify_elf_శీర్షిక(elf_శీర్షిక) అయిన అబద్ధం:
            లేపు RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        ఒకవేళ is_elf64(elf_శీర్షిక):
            elf_శీర్షిక = Elf64_Ehdr(elf_fp.పఠ(Elf64_Ehdr.s.కొలత))
        లేకపోతే:
            elf_శీర్షిక = Elf32_Ehdr(elf_fp.పఠ(Elf32_Ehdr.s.కొలత))

        phdr_పట్టిక = []

        phdr_కొలత = elf_శీర్షిక.e_phentsize

        file_స్థానభ్రంశం = elf_శీర్షిక.e_phoff
        elf_fp.seek(file_స్థానభ్రంశం)

        కొరకు _ లో శ్రేణి(elf_శీర్షిక.e_phnum):
            ఒకవేళ is_elf64(elf_శీర్షిక):
                phdr_పట్టిక.append(Elf64_Phdr(elf_fp.పఠ(phdr_కొలత)))
            లేకపోతే:
                phdr_పట్టిక.append(Elf32_Phdr(elf_fp.పఠ(phdr_కొలత)))

    తిరిగి [elf_శీర్షిక, phdr_పట్టిక]


నిర్వచనం process_section_శీర్షిక(elf_file_name):
    తోడుగా open(elf_file_name, 'rb') ఇవ elf_fp:
        elf_శీర్షిక = Elf_Ehdr_common(elf_fp.పఠ(Elf_Ehdr_common.s.కొలత))

        ఒకవేళ verify_elf_శీర్షిక(elf_శీర్షిక) అయిన అబద్ధం:
            లేపు RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        ఒకవేళ is_elf64(elf_శీర్షిక):
            elf_శీర్షిక = Elf64_Ehdr(elf_fp.పఠ(Elf64_Ehdr.s.కొలత))
        లేకపోతే:
            elf_శీర్షిక = Elf32_Ehdr(elf_fp.పఠ(Elf32_Ehdr.s.కొలత))

        shdr_పట్టిక = []

        shdr_కొలత = elf_శీర్షిక.e_shentsize
        elf_fp.seek(elf_శీర్షిక.e_shoff)

        కొరకు _ లో శ్రేణి(elf_శీర్షిక.e_shnum):
            ఒకవేళ is_elf64(elf_శీర్షిక):
                shdr_పట్టిక.append(Elf64_Shdr(elf_fp.పఠ(shdr_కొలత)))
            లేకపోతే:
                shdr_పట్టిక.append(Elf32_Shdr(elf_fp.పఠ(shdr_కొలత)))

    తిరిగి shdr_పట్టిక


నిర్వచనం చక్రీయ_తనిఖీ_32_update_file(elf_fp, స్థానభ్రంశం, కొలత, చక్రీయ_తనిఖీ):
    elf_fp.seek(స్థానభ్రంశం)
    మిగిలిన = కొలత
    అయితే మిగిలిన > 0:
        chunk = elf_fp.పఠ(min(1024 * 1024, మిగిలిన))
        ఒకవేళ కాదు chunk:
            లేపు RuntimeError("Unexpected EOF while calculating CRC")
        చక్రీయ_తనిఖీ = zlib.చక్రీయ_తనిఖీ_32(chunk, చక్రీయ_తనిఖీ)
        మిగిలిన -= పొడవు(chunk)
    తిరిగి చక్రీయ_తనిఖీ


నిర్వచనం చక్రీయ_తనిఖీ_32_update_zeros(కొలత, చక్రీయ_తనిఖీ):
    zero_chunk = b'\x00' * min(1024 * 1024, కొలత)
    మిగిలిన = కొలత
    అయితే మిగిలిన > 0:
        chunk = zero_chunk[:min(పొడవు(zero_chunk), మిగిలిన)]
        చక్రీయ_తనిఖీ = zlib.చక్రీయ_తనిఖీ_32(chunk, చక్రీయ_తనిఖీ)
        మిగిలిన -= పొడవు(chunk)
    తిరిగి చక్రీయ_తనిఖీ


నిర్వచనం చక్రీయ_తనిఖీ_32_calculate(elf_fp, phdr_పట్టిక):
    చక్రీయ_తనిఖీ = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    కొరకు curr_phdr లో phdr_పట్టిక:
        ఒకవేళ curr_phdr.p_type != ఎక్కించు_TYPE:
            కొనసాగు

        చక్రీయ_తనిఖీ = చక్రీయ_తనిఖీ_32_update_file(elf_fp, curr_phdr.p_స్థానభ్రంశం, curr_phdr.p_filesz, చక్రీయ_తనిఖీ)

        zero_కొలత = curr_phdr.p_memsz - curr_phdr.p_filesz
        ఒకవేళ zero_కొలత > 0:
            చక్రీయ_తనిఖీ = చక్రీయ_తనిఖీ_32_update_zeros(zero_కొలత, చక్రీయ_తనిఖీ)

    చక్రీయ_తనిఖీ &= 0xffffffff
    logger.info("calculated CRC: %#x", చక్రీయ_తనిఖీ)

    తిరిగి చక్రీయ_తనిఖీ


నిర్వచనం verify_swiv_seg(elf_లోపల_fp, స్థానభ్రంశం):

    elf_లోపల_fp.seek(స్థానభ్రంశం)
    swiv_seg = SWIV_Seg()
    unpacked_దత్తాంశం = (SWIV_Seg.s).unpack(elf_లోపల_fp.పఠ(SWIV_SEG_కొలత_MIN))
    swiv_seg.magic = unpacked_దత్తాంశం[0]

    తిరిగి swiv_seg.magic == SWIV_MAGIC


నిర్వచనం same_output_file(ఆదానం_elf, output_elf):
    ప్రయత్నించు:
        తిరిగి os.మార్గం.exists(output_elf) మరియు os.మార్గం.samefile(ఆదానం_elf, output_elf)
    మినహాయింపు File_Not_దొరికింది_Error:
        తిరిగి అబద్ధం


నిర్వచనం make_output_లక్ష్యం(ఆదానం_elf, output_elf):
    ఒకవేళ కాదు same_output_file(ఆదానం_elf, output_elf):
        తిరిగి output_elf, ఏమీలేదు

    output_dir = os.మార్గం.dirname(os.మార్గం.abspath(output_elf))
    output_పునాది = os.మార్గం.basename(output_elf)
    fd, తాత్కాలిక_output = tempfile.mkstemp(ఉపసర్గ=output_పునాది + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(ఆదానం_elf, తాత్కాలిక_output)
    తిరిగి తాత్కాలిక_output, తాత్కాలిక_output


నిర్వచనం swiv_generate(ఆదానం_elf, output_elf):
    output_లక్ష్యం, తాత్కాలిక_output = make_output_లక్ష్యం(ఆదానం_elf, output_elf)

    ప్రయత్నించు:
        తోడుగా open(ఆదానం_elf, "rb") ఇవ elf_లోపల_fp, open(output_లక్ష్యం, "wb+") ఇవ elf_బహిర్గతం_fp:
            [elf_శీర్షిక, phdr_పట్టిక] = preprocess_elf_file(ఆదానం_elf)

            అంకె_phdrs = elf_శీర్షిక.e_phnum
            అంకె_shdrs = elf_శీర్షిక.e_shnum
            phdr_కొలత = elf_శీర్షిక.e_phentsize

            ఒకవేళ అంకె_phdrs == 0 లేదా అంకె_phdrs > MAX_PHDR_లెక్క:
                లేపు RuntimeError("Input ELF has invalid number of program headers.")

            ఒకవేళ elf_శీర్షిక.e_shoff == 0 లేదా అంకె_shdrs == 0:
                లేపు RuntimeError("Input ELF does not contain a section header table.")

            shdr_పట్టిక_ఆదానం = process_section_శీర్షిక(ఆదానం_elf)
            shdr_పట్టిక_output = process_section_శీర్షిక(ఆదానం_elf)

            చివరి_phdr = phdr_పట్టిక[అంకె_phdrs - 1]
            ఒకవేళ చివరి_phdr.p_memsz == SWIV_SEG_కొలత_MIN మరియు verify_swiv_seg(elf_లోపల_fp, చివరి_phdr.p_స్థానభ్రంశం):
                లేపు RuntimeError("A SWIV segment is detected in the input ELF.")

            చివరి_shdr = shdr_పట్టిక_ఆదానం[elf_శీర్షిక.e_shnum - 1]
            ఒకవేళ చివరి_shdr.sh_కొలత == SWIV_SEG_కొలత_MIN మరియు verify_swiv_seg(elf_లోపల_fp, చివరి_shdr.sh_స్థానభ్రంశం):
                లేపు RuntimeError("A SWIV section is detected in the input ELF.")

            ఒకవేళ is_elf64(elf_శీర్షిక):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.కొలత)
            లేకపోతే:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.కొలత)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_కొలత = SWIV_SEG_కొలత_MIN
            swiv_shdr.sh_స్థానభ్రంశం = elf_శీర్షిక.e_shoff + (elf_శీర్షిక.e_shnum + 1) * elf_శీర్షిక.e_shentsize
            shdr_పట్టిక_output.append(swiv_shdr)
            elf_శీర్షిక.e_shnum += 1

            ఒకవేళ elf_శీర్షిక.e_phoff != 0:
                కొరకు phdr లో phdr_పట్టిక:
                    file_copy_స్థానభ్రంశం(elf_లోపల_fp, phdr.p_స్థానభ్రంశం, elf_బహిర్గతం_fp, phdr.p_స్థానభ్రంశం, phdr.p_filesz)

            elf_బహిర్గతం_fp.seek(0)
            elf_బహిర్గతం_fp.లిఖ(elf_శీర్షిక.get_packed_దత్తాంశం())

            phdr_ప్రారంభం = elf_శీర్షిక.e_phoff
            కొరకు phdr లో phdr_పట్టిక:
                elf_బహిర్గతం_fp.seek(phdr_ప్రారంభం)
                elf_బహిర్గతం_fp.లిఖ(phdr.get_packed_దత్తాంశం())
                phdr_ప్రారంభం += phdr_కొలత

            కొరకు shdr లో shdr_పట్టిక_ఆదానం[:అంకె_shdrs]:
                ఒకవేళ shdr.sh_type == SHT_NOBITS లేదా shdr.sh_కొలత == 0:
                    కొనసాగు
                file_copy_స్థానభ్రంశం(elf_లోపల_fp, shdr.sh_స్థానభ్రంశం, elf_బహిర్గతం_fp, shdr.sh_స్థానభ్రంశం, shdr.sh_కొలత)

            shdr_ప్రారంభం = elf_శీర్షిక.e_shoff
            కొరకు shdr లో shdr_పట్టిక_output:
                elf_బహిర్గతం_fp.seek(shdr_ప్రారంభం)
                elf_బహిర్గతం_fp.లిఖ(shdr.get_packed_దత్తాంశం())
                shdr_ప్రారంభం += elf_శీర్షిక.e_shentsize

            elf_బహిర్గతం_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.చక్రీయ_తనిఖీ = చక్రీయ_తనిఖీ_32_calculate(elf_బహిర్గతం_fp, phdr_పట్టిక)
            elf_బహిర్గతం_fp.seek(swiv_shdr.sh_స్థానభ్రంశం)
            elf_బహిర్గతం_fp.లిఖ(swiv_seg.get_packed_దత్తాంశం())

        ఒకవేళ తాత్కాలిక_output అయిన కాదు ఏమీలేదు:
            os.replace(తాత్కాలిక_output, output_elf)
            తాత్కాలిక_output = ఏమీలేదు

        తిరిగి 0
    చివరగా:
        ఒకవేళ తాత్కాలిక_output అయిన కాదు ఏమీలేదు మరియు os.మార్గం.exists(తాత్కాలిక_output):
            os.unlink(తాత్కాలిక_output)


నిర్వచనం ప్రధాన(argv):
    parser = argparse.వాదం_Parser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.కూడు_వాదం("-i", "--input", dest="input_file", metavar="INPUT", required=నిజం)
    parser.కూడు_వాదం("-o", "--output", dest="output_file", metavar="OUTPUT", required=నిజం)
    parser.కూడు_వాదం("-v", "--verbose", action="store_true", help="show calculated CRC")
    వాదనలు_2 = parser.parse_వాదనలు(argv)

    logging.basicConfig(
        level=logging.INFO ఒకవేళ వాదనలు_2.verbose లేకపోతే logging.WARNING,
        format="%(message)s",
    )

    ప్రయత్నించు:
        తిరిగి swiv_generate(వాదనలు_2.ఆదానం_file, వాదనలు_2.output_file)
    మినహాయింపు RuntimeError ఇవ exc:
        logger.error("ERROR: %s", exc)
        తిరిగి 2


ఒకవేళ __name__ == "__main__":
    sys.exit(ప్రధాన(sys.argv[1:]))
