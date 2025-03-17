from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

def load_pfm(file_path: str):
    with open(file_path, "rb") as f:
        # Header
        header = f.readline().decode("utf-8").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError("Not a valid PFM file")
        num_channels = 1 if header == "Pf" else 3

        # Width and height
        width, height = map(int, f.readline().decode("utf-8").strip().split())

        # Byte format
        scale = float(f.readline().decode("utf-8").strip())
        byte_format = "<f" if scale < 0 else ">f"  # Little-endian if scale is negative

        # Image data
        data = np.fromfile(f, byte_format)

        # Reshape to (height, width, channels)
        data = data.reshape((height, width, num_channels)).transpose((2, 0, 1)) # channels first
        data = np.ascontiguousarray(data[:, ::-1, :])  # Reverse rows since PFM are stored bottom-to-top

        return data

if __name__ == '__main__':
   parser = ArgumentParser()
   parser.add_argument('pfm_file')
   args = parser.parse_args()
   image_data = load_pfm(args.pfm_file)
   print(image_data)

   # # Normalize image for display (PFM values may be HDR)
   # image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

   # Display the image
   plt.imshow(image_data.transpose((1, 2, 0)), cmap="gray")  # Use 'gray' colormap for grayscale images
   plt.axis("off")  # Hide axes
   plt.show()
