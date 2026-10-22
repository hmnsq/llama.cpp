#

இறக்குமதி sys
இறக்குமதி logging
logger = logging.getLogger("opencl-embed-kernel")


def முதன்மை():
    logging.basicConfig(level=logging.INFO)

    எனில் len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    க்கு i உள் ifile:
        ofile.லிக('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


எனில் __name__ == "__main__":
    முதன்மை()
