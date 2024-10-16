import os
import json
import cv2

with open("REVERIE/BBoxes.json", "r") as f:
    bbox_data = json.load(f)

num = 0
with open("REVERIE/REVERIE_speaker_aug_enc.jsonl", "r") as f:
    data = f.readlines()
    for line in data:
        line = json.loads(line)
        
        obj_id = line['instr_id'].split("_")[1]
        scan = line['scan']
        path = line['path'][-1]
        img_path = os.path.join("/mnt/data/haodong/REVERIE/obj_views", scan, path, obj_id)
        print(img_path)
        if not os.path.exists(img_path):
            key = scan + "_" + path
            obj_info = bbox_data[key][obj_id]
            pose_num = len(obj_info['visible_pos'])
            if pose_num == 0:
                num += 1
                print(scan, path, obj_id)
                continue
            bboxes = obj_info['bbox2d']
            poses = obj_info['visible_pos']
            max_area = 0
            for i in range(pose_num):
                pose = poses[i]
                bbox = bboxes[i]
                view_path = os.path.join("/mnt/data/haodong/envedit/views_img/", scan, path, f"{pose}.jpg")
                assert os.path.exists(view_path)
                image = cv2.imread(view_path)
                x, y, w, h = bbox
                
                # calculate bbox area
                area = w * h
                if area > max_area:
                    max_area = area
                    max_idx = pose

                Height, Width = image.shape[:2]
                # Expand the bbox for 50 pixels
                expand = 30
                x1 = max(0, x-expand)
                y1 = max(0, y-expand)
                x2 = min(Width, x+w+expand)
                y2 = min(Height, y+h+expand)

                # crop the image
                crop_img = image[y1:y2, x1:x2]

                # save the image
                save_path = os.path.join("/mnt/data/haodong/REVERIE/obj_views", scan, path, obj_id, f"{pose}.jpg")
                print(save_path)
                exit(0)
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # cv2.imwrite(save_path, crop_img)

            # rename the image with max area
            # os.rename(os.path.join("/mnt/data/haodong/REVERIE/obj_views", scan, path, obj_id, f"{max_idx}.jpg"), os.path.join("/mnt/data/haodong/REVERIE/obj_views", scan, path, obj_id, f"{max_idx}_max.jpg"))


print(num)