"""Small checks of the destructive step's byte and mutation guards."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from clean_ks_transport_fragments import verified_open,unchanged


class FragmentGuards(unittest.TestCase):
    def test_exact_file_and_after_hash_change(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'fragment.bin';p.write_bytes(b'known bytes')
            expected={'bytes':11,'sha256':hashlib.sha256(b'known bytes').hexdigest()}
            with verified_open(p,expected) as (fd,identity):
                unchanged(p,fd,identity)
                p.write_bytes(b'other bytes')
                with self.assertRaises(RuntimeError):unchanged(p,fd,identity)
            self.assertTrue(p.exists())

    def test_bad_digest_is_not_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'fragment.bin';p.write_bytes(b'known bytes')
            with self.assertRaises(ValueError):
                with verified_open(p,{'bytes':11,'sha256':'0'*64}):pass
            self.assertEqual(p.read_bytes(),b'known bytes')

    def test_link_is_not_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'fragment.bin';p.write_bytes(b'known bytes')
            link=p.with_name('linked.bin');link.symlink_to(p)
            with self.assertRaises(ValueError):
                with verified_open(link,{'bytes':11,'sha256':hashlib.sha256(b'known bytes').hexdigest()}):pass
            self.assertTrue(p.exists())


if __name__=='__main__':unittest.main()
