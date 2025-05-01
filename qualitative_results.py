from pathlib import Path
from  PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import load_pfm
from matplotlib import rcParams
from torchvision.transforms.v2 import Resize

rcParams['font.family'] = 'serif'

results = [
   # {'object': '58cf4771d0f5fb221defe6da', 'view': '00000000'},
   {'object': '59056e6760bb961de55f3501', 'view': '00000023'},
   {'object': '58d36897f387231e6c929903', 'view': '00000000'},
   {'object': '58c4bb4f4a69c55606122be4', 'view': '00000000'},
   {'object': '59f363a8b45be22330016cad', 'view': '00000002'},
]

output_path = Path('visualizations')

for result in results:
   object = result['object']
   view = result['view']
   # Original
   original_image_path = Path('data', 'dataset_low_res', object, 'blended_images', view + '.jpg')
   with Image.open(original_image_path) as image:
      original_image = np.array(image) # (H, W, C)
   # Ground Truth
   ground_truth_path = Path('data', 'dataset_low_res', object, 'rendered_depth_maps', view + '.pfm')
   ground_truth = load_pfm(str(ground_truth_path))
   ground_truth = ground_truth.transpose((1, 2, 0))
   # Baseline
   baseline_path = Path('predictions', 'baseline', object, view + '.npy')
   baseline = np.load(baseline_path)
   baseline = baseline.transpose((1, 2, 0))
   # Proposed
   proposed_path = Path('predictions', 'proposed', object, view + '.npy')
   proposed = np.load(proposed_path)
   proposed = proposed.transpose((1, 2, 0))

   fig, axs = plt.subplots(1, 4, figsize=(12, 4))
   axs[0].imshow(original_image)
   axs[0].set_title('Original Image')
   axs[1].imshow(ground_truth, cmap='gray')
   axs[1].set_title('Ground Truth Depth Map')
   axs[2].imshow(proposed, cmap='gray')
   axs[2].set_title('Projexion Predicted Depth Map')
   axs[3].imshow(baseline, cmap='gray')
   axs[3].set_title('MVSNet Predicted Depth Map')
   for ax in axs:
      ax.set_axis_off()
   fig.tight_layout()
   fig.savefig(output_path / (object[:5] + '_' + view + '.jpeg'))
   