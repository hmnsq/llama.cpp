#

ਆਯਾਤ sys
ਆਯਾਤ logging
logger = logging.getLogger("opencl-embed-kernel")


ਪਰਿਭਾਸ਼ਾ ਮੁੱਖ():
    logging.basicConfig(level=logging.INFO)

    ਜੇ ਲੰਬਾਈ(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    ਲਈ i ਵਿੱਚ ifile:
        ofile.ਲਿਖ('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


ਜੇ __name__ == "__main__":
    ਮੁੱਖ()
