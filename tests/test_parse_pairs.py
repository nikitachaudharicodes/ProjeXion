from utils import parse_pairs

def test_parse_pairs():
   pairs_path = 'data_sample/BlendedMVS/dataset_low_res/5a3ca9cb270f0e3f14d0eddb/cams/pair.txt'
   pairs = parse_pairs(pairs_path=pairs_path)
   assert len(pairs) == 64
   assert pairs[0] == [33, 9, 55, 10, 20, 37, 6, 24, 61, 35]
   assert pairs[1] == [45, 15, 39, 50, 34, 11, 28, 29, 21, 47]
   assert pairs[2] == [22, 51, 36, 17, 25, 60, 8, 21, 61, 29]
