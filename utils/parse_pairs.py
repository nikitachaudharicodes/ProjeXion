from typing import List


def parse_pairs(pairs_path: str) -> List[List[int]]:
   with open(pairs_path, 'r') as pair_file:
      # First line has the number of views
      n = int(pair_file.readline().strip())
      pairs = [None] * n
      for i in range(n):
         # Parse the reference view ID
         view_id = pair_file.readline().strip()
         assert view_id != '', f"View ID {view_id} in line {i * 2 + 2} of {pairs_path} is empty" 
         view_id = int(view_id)
         assert view_id <= n, f"View ID {view_id} in line {i * 2 + 2} of {pairs_path} is out of range (>{n})"
         views = pair_file.readline().strip().split()
         # The first value is the number of views
         n_views = views[0]
         # The following values are the view IDs, sorted by relevance
         view_ids = [int(view_id) for view_id in views[1::2] if view_id != '']
         view_scores = [float(view_score) for view_score in views[2::2] if view_score != '']
         # Validate results for this image
         error_msg = f"Number of view scores and IDs does not match in {pairs_path} for reference image {view_id}"
         assert len(view_ids) == len(view_scores), error_msg
         pairs[view_id] = view_ids
      # Validate results for all images
      reference_with_missing_pairs = [index for index, views in enumerate(pairs) if views is None]
      error_msg = f"Missing pairs in {pairs_path} for reference images {reference_with_missing_pairs}"
      assert reference_with_missing_pairs == [], error_msg

      return pairs