import os
import cv2

fragment_id = '20230509182749'
layer_path = f'./train_scrolls/{fragment_id}/layers'
mask_path = f'./train_scrolls/{fragment_id}/{fragment_id}_mask.png'

crop_fragment_id = 'pi_small'
crop_layer_path = f'./train_scrolls/{crop_fragment_id}/layers'
crop_mask_path = f'./train_scrolls/{crop_fragment_id}/{crop_fragment_id}_mask.png'

if __name__ == "__main__":
  mask = cv2.imread(mask_path, 0)
  h, w = mask.shape

  # cropped coodinates (upper & lower bound)
  x0, y0, x1, y1 = 0.719, 0.652, 0.741, 0.712
  # x0, y0, x1, y1 = 0.719, 0.652, 0.786, 0.831
  x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

  # create a folder for cropped layers data
  if not os.path.exists(crop_layer_path): os.makedirs(crop_layer_path)

  # crop the layer data & save
  for i in range(0, 65):
    image = cv2.imread(f"{layer_path}/{i:02}.tif", 0)
    h, w = image.shape
    crop_image = image[y0:y1, x0:x1]
    cv2.imwrite(f"{crop_layer_path}/{i:02}.tif", crop_image)

  # crop the mask & save
  crop_mask = mask[y0:y1, x0:x1]
  cv2.imwrite(crop_mask_path, crop_mask)