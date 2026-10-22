#

దిగుమతి sys
దిగుమతి logging
logger = logging.getLogger("opencl-embed-kernel")


నిర్వచనం ప్రధాన():
    logging.basicConfig(level=logging.INFO)

    ఒకవేళ పొడవు(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    కొరకు i లో ifile:
        ofile.లిఖ('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


ఒకవేళ __name__ == "__main__":
    ప్రధాన()
