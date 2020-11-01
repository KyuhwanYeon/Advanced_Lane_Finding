
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import calibration, cal_undistort, unwarp
from threshold_filter import grad_color_filter
from find_lane import fit_polynomial, Line, find_left_right_lanes, draw_lane
input_type = 'video'
img_file_name = 'test_images/test4.jpg'
video_file_name = 'project_video.mp4'
left_line = Line()
right_line = Line()

matrix, distortion = calibration()
print("Complete calibration")
if __name__ == '__main__':
    if input_type == 'image':
        img = cv2.imread(img_file_name)
        
        
        #undistort the image
        undistort_img = cal_undistort(img, matrix, distortion)
        print("Complete undistortion")
        filtered_img = grad_color_filter(undistort_img)
        print("Complete filter")
        row_len, col_len = filtered_img.shape[:2]
        left_top, right_top = [601, 439], [700, 439]
        left_bottom, right_bottom = [200, row_len], [1200, row_len]
        src = np.float32([left_bottom, left_top, right_top, right_bottom])
        dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])
        unwarp_img , M, M_for_warp= unwarp(filtered_img, src, dst, (720, 720))
        lane_img = find_left_right_lanes(unwarp_img, left_line, right_line)
        result, fill_lane_img = draw_lane(lane_img, left_line, right_line)
        rewarp_img = cv2.warpPerspective(fill_lane_img, M_for_warp, (col_len, row_len))
        result = cv2.addWeighted(undistort_img, 1, rewarp_img, 0.3, 0)
        
        f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(12, 8))
        ax1.imshow(img)
        ax1.set_title('1. original Image', fontsize=10)
        ax2.imshow(undistort_img)
        ax2.set_title('2. undistortion', fontsize=10)
        ax3.imshow(filtered_img, cmap='gray')
        ax3.set_title('3. filtered', fontsize=10)
        ax4.imshow(unwarp_img, cmap='gray')
        ax4.set_title('4. unwarp', fontsize=10)        
        ax5.imshow(lane_img)
        ax5.set_title('5. find lane', fontsize=10)  
        ax6.imshow(fill_lane_img)
        ax6.set_title('6. fill lane', fontsize=10) 
        ax7.imshow(rewarp_img)
        ax7.set_title('7. warp', fontsize=10) 
        ax8.imshow(result)
        ax8.set_title('8. final_result', fontsize=10) 
        # ax5.plot(left_fitx, ploty, 20, color='yellow')
        # ax5.plot(right_fitx, ploty, 20, color='yellow')
        
    elif input_type == 'video':
        cap = cv2.VideoCapture(video_file_name)
        while (cap.isOpened()):
            _, frame = cap.read()
             #undistort the image
            undistort_img = cal_undistort(frame, matrix, distortion)
            filtered_img = grad_color_filter(undistort_img)
            row_len, col_len = filtered_img.shape[:2]
            left_top, right_top = [601, 439], [700, 439]
            left_bottom, right_bottom = [200, row_len], [1200, row_len]
            src = np.float32([left_bottom, left_top, right_top, right_bottom])
            dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])
            unwarp_img , M, M_for_warp= unwarp(filtered_img, src, dst, (720, 720))
            lane_img = find_left_right_lanes(unwarp_img, left_line, right_line)
            result, fill_lane_img = draw_lane(lane_img, left_line, right_line)
            rewarp_img = cv2.warpPerspective(fill_lane_img, M_for_warp, (col_len, row_len))
            result = cv2.addWeighted(undistort_img, 1, rewarp_img, 0.3, 0)
            
            
            info = np.zeros_like(result)
            info[5:200, 5:600] = (255, 255, 255)
            info = cv2.addWeighted(result, 1, info, 0.2, 0)
            cv2.imshow('advanced_lane_detection', info)
                  

            # out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.waitKey(0)
            #if cv2.waitKey(1) & 0xFF == ord('r'):
            #    cv2.imwrite('check1.jpg', undist_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()                
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(undistort_img)
        # plt.show()
        # plt.imshow(filtered_img)
        # plt.show()