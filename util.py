import cv2

def detect_face(img , face_detection_object , W ,H):
    img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    out = face_detection_object.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections :

            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1 , y1 , w , h = bbox.xmin , bbox.ymin , bbox.width , bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            img[y1:y1+h ,x1:x1+w, ] = cv2.blur(img[y1:y1+h ,x1:x1+w ] , (30,30))
    
    return img