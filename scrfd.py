import numpy as np
import cv2
from pprint import pprint

class SCRFD:
    def __init__(self, session):
        self.session = session
        self.input_size = 320   # <<< MAJOR SPEED BOOST
        self.strides = [8,16,32]
        self.mean = 127.5
        self.std = 128.0

        # -------- PRECOMPUTE ANCHORS ONCE --------
        self.anchor_cache = {}
        for s in self.strides:
            self.anchor_cache[s] = self._anchors(s)

        self.input_name = session.get_inputs()[0].name

    def _anchors(self, stride):
        f = self.input_size // stride
        shift_x = (np.arange(f) + 0.5) * stride
        shift_y = (np.arange(f) + 0.5) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        anchors = np.stack([xv, yv], axis=-1).reshape(-1, 2)
        anchors = np.repeat(anchors, 2, axis=0)
        return anchors.astype(np.float32)

    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(h*scale), int(w*scale)

        img = cv2.resize(img, (nw, nh))
        canvas = np.zeros((self.input_size, self.input_size, 3), np.uint8)
        top = (self.input_size-nh)//2
        left = (self.input_size-nw)//2
        canvas[top:top+nh, left:left+nw] = img

        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = (canvas - self.mean) / self.std
        blob = canvas.transpose(2,0,1)[None].astype(np.float32)

        return blob, scale, left, top

    def detect(self, img, thresh=0.6):
        blob, scale, left, top = self.preprocess(img)
        outs = self.session.run(None, {self.input_name: blob})

        boxes_all, scores_all, kps_all = [], [], []

        for stride, cls, box, kps in zip(self.strides, outs[0:3], outs[3:6], outs[6:9]):

            scores = cls.reshape(-1)
            mask = scores > thresh
            if not mask.any():
                continue

            anchors = self.anchor_cache[stride][mask]
            box = box.reshape(-1,4)[mask]
            kps = kps.reshape(-1,10)[mask]

            cx, cy = anchors[:,0], anchors[:,1]
            x1 = (cx - box[:,0]*stride - left) / scale
            y1 = (cy - box[:,1]*stride - top) / scale
            x2 = (cx + box[:,2]*stride - left) / scale
            y2 = (cy + box[:,3]*stride - top) / scale

            boxes = np.stack([x1,y1,x2,y2],1)

            lm = kps.reshape(-1,5,2)
            lm[:,:,0] = (lm[:,:,0]-left)/scale
            lm[:,:,1] = (lm[:,:,1]-top)/scale

            boxes_all.append(boxes)
            scores_all.append(scores[mask])
            kps_all.append(lm)

        if not boxes_all:
            return [], []

        boxes = np.concatenate(boxes_all)
        scores = np.concatenate(scores_all)
        kps = np.concatenate(kps_all)

        keep = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 0.6, 0.4
        )

        if len(keep)==0:
            return [], []

        keep = np.array(keep).flatten()
        return boxes[keep].astype(int).tolist(), kps[keep]
