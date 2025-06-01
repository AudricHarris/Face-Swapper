import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
import os
from pathlib import Path

class FaceSwapper:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh with static image mode for images
        self.face_mesh_static = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Initialize face mesh for video processing
        self.face_mesh_video = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key facial landmark indices for face boundary (MediaPipe 468 landmarks)
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        self.prev_target_landmarks = None
        self.smoothing_factor = 0.7
        self.triangles = None  # Fixed triangle indices for source face
        self.source_reference_points = None
        self.use_all_landmarks = False  # Set to True for all 468 points, False for keypoints only

    def get_face_landmarks(self, image, use_static=True):
        """Extract facial landmarks from an image using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if use_static:
            results = self.face_mesh_static.process(rgb_image)
        else:
            results = self.face_mesh_video.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert normalized coordinates to pixel coordinates
        h, w = image.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        
        return np.array(landmarks, dtype=np.int32)
    
    def smooth_landmarks(self, prev, current):
        """Exponential moving average smoothing for landmarks."""
        if prev is None:
            return current
        return (self.smoothing_factor * prev + (1 - self.smoothing_factor) * current).astype(np.int32)

    def get_key_points(self, landmarks):
        """Extract key facial points for face swapping"""
        if landmarks is None:
            return None
        if self.use_all_landmarks:
            return landmarks  # Use all 468 points
        
        # Select key points for better face swapping
        key_indices = [
            # Face oval
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
            # Eyes
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            # Eyebrows
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
            296, 334, 293, 300, 276, 283, 282, 295, 285, 336,
            # Nose
            1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 363, 358, 279, 360, 440,
            # Mouth
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78
        ]
        
        # Remove duplicates and ensure indices are valid
        key_indices = list(set(key_indices))
        key_indices = [i for i in key_indices if i < len(landmarks)]
        
        # Ensure key_indices is always sorted for consistent order
        key_indices = sorted(key_indices)
        
        return landmarks[key_indices]
    
    def get_face_mask(self, image, landmarks):
        """Create a mask for the face region using MediaPipe face oval"""
        if landmarks is None:
            return None
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Use face oval landmarks for mask
        face_points = landmarks[self.FACE_OVAL]
        
        # Create convex hull and fill
        hull = cv2.convexHull(face_points)
        cv2.fillPoly(mask, [hull], 255)
        
        return mask
    
    def delaunay_triangulation(self, points, shape):
        """Compute Delaunay triangulation and return triangle indices."""
        # Add corner points to avoid edge effects
        h, w = shape[:2]
        boundary_points = [
            [0, 0], [w//2, 0], [w-1, 0],
            [0, h//2], [w-1, h//2],
            [0, h-1], [w//2, h-1], [w-1, h-1]
        ]
        
        all_points = np.vstack([points, boundary_points])
        
        # Remove duplicate points
        unique_points = []
        seen = set()
        for point in all_points:
            point_tuple = tuple(point)
            if point_tuple not in seen:
                unique_points.append(point)
                seen.add(point_tuple)
        
        all_points = np.array(unique_points)
        tri = Delaunay(all_points)
        return tri.simplices, all_points
    
    def apply_affine_transform(self, src, src_tri, dst_tri, size):
        """Apply affine transformation to a triangle"""
        # Get transformation matrix
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        
        # Apply transformation
        dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), 
                           flags=cv2.INTER_LINEAR, 
                           borderMode=cv2.BORDER_REFLECT_101)
        
        return dst
    
    def warp_triangle(self, img1, img2, t1, t2):
        """Warp triangular region from img1 to img2"""
        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        
        # Offset points by left top corner of rectangle
        t1_rect = []
        t2_rect = []
        t2_rect_int = []
        
        for i in range(3):
            t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        
        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)
        
        # Apply warp to input image
        img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        size = (r2[2], r2[3])
        
        if img1_rect.size == 0:
            return
        
        img2_rect = self.apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
        img2_rect = img2_rect * mask
        
        # Copy triangular region to output image
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_rect
    
    def color_match(self, source, target, mask):
        """Match colors between source and target faces"""
        # Convert to LAB color space for better color matching
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        
        # Calculate mean and std for each channel in face region
        source_mean = cv2.mean(source_lab, mask)[:3]
        target_mean = cv2.mean(target_lab, mask)[:3]
        
        source_std = []
        target_std = []
        
        for i in range(3):
            source_channel = source_lab[:,:,i]
            target_channel = target_lab[:,:,i]
            
            source_vals = source_channel[mask > 0]
            target_vals = target_channel[mask > 0]
            
            if len(source_vals) > 0 and len(target_vals) > 0:
                source_std.append(np.std(source_vals))
                target_std.append(np.std(target_vals))
            else:
                source_std.append(1.0)
                target_std.append(1.0)
        
        # Apply color transfer
        result_lab = source_lab.copy().astype(np.float32)
        
        for i in range(3):
            if source_std[i] > 0:
                result_lab[:,:,i] = ((result_lab[:,:,i] - source_mean[i]) * 
                                   (target_std[i] / source_std[i])) + target_mean[i]
            
            result_lab[:,:,i] = np.clip(result_lab[:,:,i], 0, 255)
        
        return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def seamless_clone(self, source, target, mask):
        """Apply seamless cloning for smooth blending"""
        # Find center of face for cloning
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            center = (int(moments['m10']/moments['m00']), 
                     int(moments['m01']/moments['m00']))
        else:
            center = (mask.shape[1]//2, mask.shape[0]//2)
        
        # Use Poisson blending
        try:
            result = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
            return result
        except Exception as e:
            print(f"Seamless cloning failed: {e}, using alpha blending")
            # Fallback to simple alpha blending
            alpha = mask.astype(float) / 255
            alpha = np.stack([alpha, alpha, alpha], axis=2)
            return (source * alpha + target * (1 - alpha)).astype(np.uint8)
    
    def swap_face(self, source_img, target_img, use_static=True):
        """Main face swap function"""
        # Get landmarks for both images
        source_landmarks = self.get_face_landmarks(source_img, use_static=True)
        target_landmarks = self.get_face_landmarks(target_img, use_static=use_static)
        if source_landmarks is None or target_landmarks is None:
            print("Could not detect face in one or both images")
            return None

        source_key_points = self.get_key_points(source_landmarks)
        target_key_points = self.get_key_points(target_landmarks)
        if source_key_points is None or target_key_points is None:
            print("Could not extract key points")
            return None

        # --- Landmark smoothing ---
        target_key_points = self.smooth_landmarks(self.prev_target_landmarks, target_key_points)
        self.prev_target_landmarks = target_key_points

        # --- Triangulation: compute ONCE on source face, reuse for all frames ---
        if self.triangles is None or self.source_reference_points is None:
            tri = Delaunay(source_key_points)
            self.triangles = tri.simplices
            self.source_reference_points = source_key_points.copy()
        triangles = self.triangles

        # --- Warp triangles from source to target ---
        output = target_img.copy().astype(np.float32)
        for triangle in triangles:
            if all(idx < len(source_key_points) and idx < len(target_key_points) for idx in triangle):
                t1 = [source_key_points[triangle[0]], source_key_points[triangle[1]], source_key_points[triangle[2]]]
                t2 = [target_key_points[triangle[0]], target_key_points[triangle[1]], target_key_points[triangle[2]]]
                try:
                    self.warp_triangle(source_img.astype(np.float32), output, t1, t2)
                except Exception as e:
                    continue  # Skip problematic triangles

        # Convert back to uint8
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # Get face mask for blending
        target_mask = self.get_face_mask(target_img, target_landmarks)
        if target_mask is None:
            print("Could not create face mask")
            return None
        
        # Apply color matching
        try:
            output = self.color_match(output, target_img, target_mask)
        except Exception as e:
            print(f"Color matching failed: {e}")
        
        # Apply seamless blending
        try:
            final_result = self.seamless_clone(output, target_img, target_mask)
        except Exception as e:
            print(f"Blending failed: {e}")
            final_result = output
        
        return final_result
    
    def process_video(self, source_image_path, video_path, output_path):
        """Process entire video for face swapping"""
        # Load source image
        source_img = cv2.imread(source_image_path)
        if source_img is None:
            print(f"Could not load source image: {source_image_path}")
            return
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform face swap (use video mode for better tracking)
            swapped_frame = self.swap_face(source_img, frame, use_static=False)
            
            if swapped_frame is not None:
                out.write(swapped_frame)
            else:
                # If face swap fails, use original frame
                out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Face swap completed! Output saved to: {output_path}")

def main():
    # Initialize face swapper
    swapper = FaceSwapper()
    
    # Define paths
    data_folder = Path("data")
    source_image = data_folder / "person.png"
    input_video = data_folder / "scene.mp4"
    output_video = data_folder / "face_swapped_output.mp4"
    
    # Check if files exist
    if not source_image.exists():
        print(f"Source image not found: {source_image}")
        return
    
    if not input_video.exists():
        print(f"Input video not found: {input_video}")
        return
    
    # Process the video
    swapper.process_video(str(source_image), str(input_video), str(output_video))

if __name__ == "__main__":
    main()