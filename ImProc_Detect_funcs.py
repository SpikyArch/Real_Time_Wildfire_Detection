
import cv2
import numpy as np
import pywt
from skimage.util import view_as_blocks

# Declare Image Processor Class
class ImageProcessor:
    # Class initialisation
    def __init__(self, image) -> None:
        self.image = image

    # Preprocessing the image
    def preprocessor(self):
        # Initialising empty matrices for splitting the image into constituent channels
        B_n = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint16) # In .shape, 0 - rows(height), 1 - columns(width) and 2 - channels 
        G_n = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint16)
        R_n = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint16)

        img_copy = np.zeros((self.image.shape), dtype=np.uint16)
        img_copy = self.image.copy()

        # Excluding pixels from the image which are above certain thresholds
        color_criteria = (img_copy[:, :, 0]>150) & (img_copy[:, :, 1]>190) & (img_copy[:, :, 2]>210) # Open CV does RGB backwards so this affects Blue, Green and Red in that order
        img_copy[color_criteria] = [0, 0, 0]
        
        # Calculating the sum of pixel values across all channels
        channel_sum = np.uint32(img_copy[:, :, 0] + img_copy[:, :, 1] + img_copy[:, :, 2])
        channel_sum = np.nan_to_num(channel_sum, False, 1)
        channel_sum_corrected = np.where(channel_sum==0, 1, channel_sum)

        # Applying the preprocessing
        B_n = (255*img_copy[:, :, 0])/channel_sum_corrected # RGB normalisation is done across every pixel for every channel (mentioned in the energy maps paper)
        G_n = (255*img_copy[:, :, 1])/channel_sum_corrected
        R_n = (255*img_copy[:, :, 2])/channel_sum_corrected

        self.pp_image = np.zeros((self.image.shape), dtype=np.uint16)
        self.pp_image = cv2.merge([B_n, G_n, R_n]) # Combine these normalisations

        return self.pp_image
    


    def vbi_idx(self):
        # Defining the vegetation index

        # Splitting the image into constituent channels
        B_n, G_n, R_n = cv2.split(self.pp_image)
        self.vbi = np.zeros((B_n.shape[0], B_n.shape[1]), dtype=np.int16) # Placeholder initialisation

        # Calculating the vegetation index as specified in the paper
        add_result = cv2.add(R_n, B_n)
        self.vbi = cv2.subtract(2*G_n, add_result)
        self.vbi = cv2.GaussianBlur(self.vbi, (7, 7), 125, 200) # 5x5 and stds of 125(x) and 200(y) meaning it is a strong blur

        return self.vbi
    
    def fi_idx(self):
        B_n, G_n, R_n = cv2.split(self.pp_image) # Again it is RGB backwards as we are using opencv
        self.fi = np.zeros(B_n.shape, dtype=np.int16)

        # Calculating the fire index as specified in the paper
        add_result = cv2.add(G_n, B_n)
        self.fi = cv2.subtract(2*R_n, add_result)
        self.fi = cv2.GaussianBlur(self.fi, (7, 7), 125, 200)

        return self.fi
    
    def ffi_idx(self, alpha):
        # Calculating the forest fire index as specified in the paper
        self.ffi = np.zeros((self.image.shape[0:2]), dtype=np.int16)
        self.ffi = alpha*self.fi - self.vbi
        self.ffi = cv2.GaussianBlur(self.ffi, (7, 7), 125, 200)
        self.ffi = self.ffi.astype(np.uint16)

        return self.ffi
    
    def calc_tf(self, alpha):
        # Threshold calculation
        ffi_sigma = np.std(self.ffi)
        vbi_sigma = np.std(self.vbi)

        self.tf = (alpha*ffi_sigma + vbi_sigma)/(alpha+1)

        return self.tf
    
    def ffi_binarize(self):
        # Binarization of the resultant images
        self.ffi_bin = np.zeros((self.ffi.shape), dtype=np.uint8)
        # self.ffi_bin = np.where(self.ffi>self.tf, 255, 0).astype(np.uint8)

        min_thresh = 30  # Experiment with a raneg in threshold to make 
        self.ffi_bin = np.where((self.ffi > self.tf) & (self.ffi > min_thresh), 255, 0).astype(np.uint8)

        return self.ffi_bin
    
    def erosion(self):
        # Morphological operation to smooth out detections
        self.ffi_eroded = cv2.erode(self.ffi_bin, (11, 11), iterations=1)

        return self.ffi_eroded
    
    def dilation(self):
        # Morphological operation to smooth out detections
        self.ffi_dilated = cv2.dilate(self.ffi_eroded, (7, 7), iterations=1)

        return self.ffi_dilated
    
    def blur(self):
        # Morphological operation to smooth out detections
        self.ffi_blurred = cv2.GaussianBlur(self.ffi_dilated, (7, 7), 200, 275)

        return self.ffi_blurred
    # Now apply the three rules from the energy maps paper
    def rule_1(self, beta):
        # Color rule #1
        self.rule1_result = np.zeros(np.shape(self.vbi), dtype=np.uint8)
        # Extract R, G, B from preprocessed image (remember: OpenCV uses BGR)
        B = self.pp_image[:, :, 0]
        G = self.pp_image[:, :, 1]
        R = self.pp_image[:, :, 2]

        # Create boolean masks
        cond1 = np.abs(R - G) < beta
        cond2 = np.abs(G - B) < beta
        cond3 = np.abs(R - B) < beta

        # Combine conditions with logical AND
        combined = cond1 & cond2 & cond3

        # Convert boolean mask to uint8 image (0 or 255)
        self.rule1_result = (combined.astype(np.uint8)) * 255
        return self.rule1_result

    def rule_2(self, R_thresh, B_thresh):
        # Color rule #2
        self.rule2_result = np.zeros(np.shape(self.vbi), dtype=np.uint8)
        # Extract R and B channels
        B = self.pp_image[:, :, 0]
        R = self.pp_image[:, :, 2]

        # Logical condition: R > R_thresh AND B < B_thresh
        mask = (R > R_thresh) & (B < B_thresh)

        # Convert to 0/255 binary image
        self.rule2_result = (mask.astype(np.uint8)) * 255

        return self.rule2_result
    
    def rule_3(self):
        # Color rule #3
        self.I_ty = np.zeros((np.shape(self.vbi)[0], np.shape(self.vbi)[1]), dtype=np.uint16)
        self.I_ty = (self.pp_image[:, :, 0] + self.pp_image[:, : , 1] + self.pp_image[:, :, 2])/3
        self.rule3_result = np.zeros(np.shape(self.rule1_result), dtype=np.uint8)

        cond1 = np.where(self.I_ty > 80, 255, 0)
        cond2 = np.where(self.I_ty < 150, 255, 0)
        cond3 = np.where(self.I_ty > 150, 255, 0)
        cond4 = np.where(self.I_ty < 220, 255, 0)
        cond1_result = cv2.bitwise_and(cond1, cond2)
        cond2_result = cv2.bitwise_and(cond3, cond4)
        cond_result = cv2.bitwise_or(cond1_result, cond2_result)

        self.rule3_result = cond_result

        return self.rule3_result
    
    def wavelet_transform(self, D_fs, D_o): # Origional image = D_o, Forest fire reigon = D_f, Flame smoke area = D_fs 
        # Wavelet transform on the grayscale image
        D_fs_g = cv2.cvtColor(D_fs, cv2.COLOR_BGR2GRAY)
        D_o_g = cv2.cvtColor(D_o, cv2.COLOR_BGR2GRAY)

        # # Extracting the coefficients
        # _, (LH_fs, HL_fs, HH_fs) = pywt.wavedec2(D_fs_g, 'haar') # IF you want multilevel wavelet - need to use pywt.wavedec2 instead
        # _, (LH_o, HL_o, HH_o) = pywt.wavedec2(D_o_g, 'haar')
        coeffs_fs = pywt.wavedec2(D_fs_g, 'sym8', level=5)
        coeffs_o = pywt.wavedec2(D_o_g, 'sym8', level=5)

        LH_fs, HL_fs, HH_fs = coeffs_fs[1]  # Level 5 details
        LH_o, HL_o, HH_o = coeffs_o[1]



        # Initialising the energy images
        E_fs = np.zeros(LH_fs.shape, dtype=np.float32)
        E_o = np.zeros(LH_o.shape, dtype=np.float32)

        # Calculating the energy images
        E_fs = LH_fs**2 + HL_fs**2 + HH_fs**2
        E_o = LH_o**2 + HL_o**2 + HH_o**2

        if (E_fs.shape == E_o.shape):
            print("E_fs and E_o are the same size and there is no need to take away rows")
        

        # Axis homogenization  
        # Specific fix? - Need to have a look at new results to see what to do
        # E_fs = np.delete(E_fs, -1, axis=0)
        # E_fs = np.delete(E_fs, -1, axis=1)
        # E_o = np.delete(E_o, -1, axis=0)
        # E_o = np.delete(E_o, -1, axis=1)

        # Dividing the image into blocks
        # Axis homogenisation (robust?)
        block_shape = (5, 5)
        h, w = E_fs.shape
        new_h = h - (h % block_shape[0])
        new_w = w - (w % block_shape[1])
        E_fs = E_fs[:new_h, :new_w]
        E_o = E_o[:new_h, :new_w]

        # Now block division works
        E_fs_blocks = view_as_blocks(E_fs, block_shape)
        E_o_blocks = view_as_blocks(E_o, block_shape)


        # Initializing the energy maps
        E_fs_map = np.zeros((E_fs_blocks.shape[0], E_fs_blocks.shape[1]), dtype=np.uint16)
        E_o_map = np.zeros((E_o_blocks.shape[0], E_o_blocks.shape[1]), dtype=np.uint16)
        E_noise = np.zeros((E_o_blocks.shape[0], E_o_blocks.shape[1]), dtype=np.uint16)

        # Calculating the energy maps
        for i in range(E_fs_blocks.shape[0]):
            for j in range(E_o_blocks.shape[1]):
                E_fs_map[i, j] = np.ndarray.sum(E_fs_blocks[i, j])
                E_o_map[i, j] = np.ndarray.sum(E_o_blocks[i, j])
                
                if E_fs_map[i, j] > 1.5 * E_o_map[i, j]:  # Only strong signals survive
                    E_noise[i, j] = 255
                else:
                    E_noise[i, j] = 0

                if E_fs_map[i, j] > 0:
                    E_fs_map[i, j] = 255
                else:
                    E_fs_map[i, j] = 0

        # Final result extraction and type correction
        E_result = np.zeros((E_o_blocks.shape[0], E_o_blocks.shape[1]), dtype=np.uint8)
        E_result = (E_fs_map - E_noise).astype(np.uint8)
        # E_result = cv2.resize(E_result, (500, 500), cv2.INTER_AREA)
        E_result = cv2.resize(E_result, (640,512), cv2.INTER_LINEAR)

        

        return E_result


if __name__=="__main__":
    pass
