import pixellib
from pixellib.semantic import semantic_segmentation

segment_video = semantic_segmentation()
segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
segment_video.segmentAsAde20k("car.mp4", output_image_name= "out.mp4")