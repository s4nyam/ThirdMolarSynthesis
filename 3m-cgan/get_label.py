from pathlib import Path
import random
random_image_from_folder = random.choice(list(Path("thirdmolar").glob("*.png")))
filechosen = random_image_from_folder.name
print(filechosen)
label = int(random_image_from_folder.stem.split("_")[-1][0])
print(label)