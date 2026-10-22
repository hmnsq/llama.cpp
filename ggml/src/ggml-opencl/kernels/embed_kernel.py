#

import sys
import logging
logger = logging.getLogger("opencl-embed-kernel")


def main():
    logging.basicConfig(level=logging.INFO)

    ប្រសិនបើ len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    សម្រាប់ i anta_h ifile:
        ofile.write('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


ប្រសិនបើ __name__ == "__main__":
    main()
