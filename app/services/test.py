from matplotlib import pyplot as plt
from PIL import Image
import io

def test_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    plt.imshow(image)
    plt.axis("off")
    plt.show()