#

आनयन sys
आनयन logging
logger = logging.getLogger("opencl-embed-kernel")


परिभाषा मुख्यम्():
    logging.basicConfig(level=logging.INFO)

    यदि दीर्घता(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    प्रति i अन्तः ifile:
        ofile.लिख('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


यदि __name__ == "__main__":
    मुख्यम्()
