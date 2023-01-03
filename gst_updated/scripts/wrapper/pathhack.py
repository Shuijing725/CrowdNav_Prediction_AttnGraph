import sys
from os.path import abspath, join, split
file_path = split(abspath(__file__))[0]
pkg_path = join(file_path, '../..')
sys.path.append(pkg_path)
