import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
from moviepy.editor import VideoFileClip




def undistorted(img, mtx, dist):
    dst=cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def abs_sobel_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    scale_sobelx = np.uint8(255 * sobelx / np.max(sobelx))
    sx_binary = np.zeros_like(scale_sobelx)
    sx_binary[(scale_sobelx >= thresh[0]) & (scale_sobelx <= thresh[1])] = 1
    return sx_binary


def hls_thresh(img, thresh_s=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_s[0]) & (s_channel <= thresh_s[1])] = 1
    return s_binary


def rgb_thresh(img, Channel='r', thresh=(0, 255)):
    if Channel == 'b':
        dim = 0
    elif Channel == 'g':
        dim = 1
    else:
        dim = 2
    channel = img[:, :, dim]
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary


def hsv_thresh(img, Channel='h', thresh=(0, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if Channel == 'h':
        dim = 0
    elif Channel == 's':
        dim = 1
    else:
        dim = 2
    channel = hsv[:, :, dim]
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary


def mag_sobel_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mag_binary = np.zeros_like(scale_sobel)
    mag_binary[(scale_sobel <= thresh[1]) & (scale_sobel >= thresh[0])] = 1
    return mag_binary


def dir_sobel_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    dir_sobel = np.arctan2(sobely, sobelx)
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel <= thresh[1]) & (dir_sobel >= thresh[0])] = 1
    return binary_output


def Luv_thresh(img, Channel='L', thresh=(0, 255)):
    Luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    if Channel == 'L':
        dim = 0
    elif Channel == 'u':
        dim = 1
    else:
        dim = 2
    channel = Luv[:, :, dim]
    binary = np.zeros_like(channel)
    binary[(channel <= thresh[1]) & (channel >= thresh[0])] = 1
    return binary


def Lab_thresh(img, Channel='b', thresh=(0, 255)):
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    if Channel == 'L':
        dim = 0
    elif Channel == 'a':
        dim = 1
    else:
        dim = 2
    channel = Lab[:, :, dim]
    binary = np.zeros_like(channel)
    binary[(channel <= thresh[1]) & (channel >= thresh[0])] = 1
    return binary


def combine_Lab_thresh(img, thresh_L=(0, 255), thresh_a=(0, 255), thresh_b=(0, 255)):
    img = cv2.GaussianBlur(img, (5, 5), 2.0)
    L_binary = Lab_thresh(img, 'L', thresh_L)
    a_binary = Lab_thresh(img, 'a', thresh_a)
    b_binary = Lab_thresh(img, 'b', thresh_b)
    color_binary = np.zeros_like(b_binary)
    color_binary[(L_binary == 1) & (a_binary == 1) & (b_binary == 1)] = 1
    return color_binary


def combine_Luv_thresh(img, thresh_L=(0, 255), thresh_u=(0, 255), thresh_v=(0, 255)):
    img = cv2.GaussianBlur(img, (5, 5), 2.0)
    L_binary = Luv_thresh(img, 'L', thresh_L)
    u_binary = Luv_thresh(img, 'u', thresh_u)
    v_binary = Luv_thresh(img, 'v', thresh_v)
    color_binary = np.zeros_like(L_binary)
    color_binary[((L_binary == 1) & (u_binary == 1) & (v_binary == 1))] = 1
    return color_binary


def combine_rgb_thresh(img, thresh_r=(0, 255), thresh_g=(0, 255), thresh_b=(0, 255)):
    img = cv2.GaussianBlur(img, (5, 5), 2.0)
    r_binary = rgb_thresh(img, 'r', thresh_r)
    g_binary = rgb_thresh(img, 'g', thresh_g)
    b_binary = rgb_thresh(img, 'b', thresh_b)
    color_binary = np.zeros_like(r_binary)
    color_binary[(r_binary == 1) & (g_binary == 1) & (b_binary == 1)] = 1
    return color_binary


def combine_thresh(img, thresh_L=(0, 255), thresh_a=(0, 255), b=(0, 255), thresh_r=(0, 255), thresh_g=(0, 255),
                   thresh_b=(0, 255)):
    Lab_thresh = combine_Lab_thresh(img, thresh_L, thresh_a, b)
    rgb_thresh = combine_rgb_thresh(img, thresh_r, thresh_g, thresh_b)
    color_binary = np.zeros_like(Lab_thresh)
    color_binary[(Lab_thresh == 1) | (rgb_thresh == 1)] = 1
    return color_binary


def fit_lines(binary_img, plot=True):

    histogram = np.sum(binary_img[binary_img.shape[0] // 2:, :], axis=0)#1.shape[0]求矩阵第二维的长度，2. sum,当axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行
    out_img = np.dstack((binary_img, binary_img, binary_img))
    out_img = np.uint8(out_img * 255)
    out_img = 255 * out_img
    midpoint = histogram.shape[0] // 2# 第二维度长度除2
    left_base = np.argmax(histogram[:midpoint])#获取图片左侧最大值得索引
    right_base = np.argmax(histogram[midpoint:]) + midpoint#获取图片右侧最大值得索引
    nwindows = 9
    window_height = np.int(binary_img.shape[0] / nwindows)#图片高度分成9份（矩形框的高度）
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()#返回非零元素的索引值数组
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_current = left_base
    right_current = right_base
    margin = 80
    minpix = 55
    left_lane_inds = []#左车道积分
    right_lane_inds = []#右车道积分

    for window in range(nwindows):
        win_y_low = binary_img.shape[0] - (window + 1) * window_height#矩形框的下点
        win_y_high = binary_img.shape[0] - window * window_height#矩形框的上点
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), #矩形框的左下点坐标
                      (win_xleft_high, win_y_high), #矩形框的左上点坐标
                      (0, 255, 0),#颜色
                      2) #线段宽度
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)#画矩形
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]


        # Append these indices to the lists
        # 将这些索引追加到列表中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        # 如果找到大于“MimPix像素”，则重新定位下一个窗口的平均位置
        if len(good_left_inds) > minpix:
            left_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)#连接
    right_lane_inds = np.concatenate(right_lane_inds)#连接

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    #其中2表示多项式的最高阶数，lefty, leftx为将要拟合的数据，它是用数组的方式输入．输出参数为拟合多项式 y=a1xn+...+anx+a n+1的系数
    left_fit = np.polyfit(lefty, leftx, 2) #曲线拟合函数，自由度2
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])#（0-（binary_img.shape[0] - 1），binary_img.shape[0]个数）在指定的间隔内返回均匀间隔的数字
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    #========================
    plt.plot(left_fitx,ploty)
    plt.plot(right_fitx,ploty)
    #=======================
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if plot == True:
        fig = plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')#x轴数据，y轴数据
        plt.plot(right_fitx, ploty, color='yellow')#x轴数据，y轴数据
        plt.xlim(0, 1280)# 设置x轴刻度范围
        plt.ylim(720, 0)# 设置y轴刻度范围
    return left_fit, right_fit, out_img


def fit_lines_continous(binary_warped, left_fit, right_fit, nwindows=9, plot=True):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions 再次，提取左、右行像素位置。
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each 二阶多项式拟合
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting 生成x和y值用于绘图
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    if plot == True:
        # Draw the lane onto the warped blank image
        cv2.fillPoly(out_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(out_img, np.int_([right_line_pts]), (0, 255, 0))
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    return left_fit, right_fit, out_img


def cal_curverature(img_shape, left_fit, right_fit):
    ploty=np.linspace(0, img_shape[0]-1, num=img_shape[0])
    ym_per_pix = 30/720 # meters per pixel in y dimension Y维像素数
    xm_per_pix = 3.7/700 # meters per pixel in x dimension X维像素数
# Fit new polynomials to x,y in world space 在世界空间中将新多项式拟合到x，y
    y_eval = np.max(ploty)
    leftx=left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
    rightx=right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
# Calculate the new radii of curvature 计算曲率半径
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    xoffset=(left_fit_cr[2]+right_fit_cr[2])/2-img.shape[1]*xm_per_pix/2
    return left_curverad, right_curverad, xoffset



def warp_perspective_back(img, warped, left_fit, right_fit, Minv):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty=np.linspace(0, img.shape[0]-1, num=img.shape[0])
    leftx=left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
    rightx=right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image 在左右车道之间填充颜色
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv) 利用逆透视矩阵（MIV）将空白返回到原始图像空间
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
# Combine the result with the original image 将结果与原始图像相结合
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result,newwarp,color_warp

def add_text(result, left_fit, right_fit):
    left_curverad, right_curverad, xoffset=cal_curverature(result.shape, left_fit, right_fit)
    cur=(left_curverad+right_curverad)/2
    font= cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius of Curvature (%.2f)' % cur , (50, 50), font, 1, (255, 0, 0), 2)
    direction='left' if xoffset>0 else 'right'
    cv2.putText(result, ('Vehicle is at %.2f m %s of the center' % (np.abs(xoffset),direction)), (50, 100), font, 1, (255, 0, 0), 2)
    return result


def sanity_check(left_fit, right_fit):
    if len(left_fit) < 3 or len(right_fit) < 3:
        return False

    ploty = np.linspace(0, 720, 30)
    leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    mean_dist = np.mean(rightx - leftx)
    dmin = np.min(rightx - leftx)
    dmax = np.max(rightx - leftx)
    if mean_dist >= 550 and mean_dist <= 750:
        status = True
    else:
        return False
    if dmax - dmin <= 80:
        status = True
    else:
        return False
    slope_left = 2 * left_fit[0] * ploty + left_fit[
        1]  # calualuate the slope of some random selected points and compare them
    slope_right = 2 * right_fit[0] * ploty + right_fit[1]
    if np.mean(np.abs(slope_left - slope_right)) < 0.12:
        status = True
    else:
        return False
    return status

#修改图片大小
def resize(color_binary, warp, color_warp, newwarp):
    height, width = color_binary.shape[:2]  # 获取原图像的水平方向尺寸和垂直方向尺寸。
    color_binary1 = cv2.resize(color_binary, (int(0.2 * width), int(0.2 * height)), interpolation=cv2.INTER_CUBIC)
    height, width = warp.shape[:2]  # 获取原图像的水平方向尺寸和垂直方向尺寸。
    warp1 = cv2.resize(warp, (int(0.2 * width), int(0.2 * height)), interpolation=cv2.INTER_CUBIC)
    height, width = color_warp.shape[:2]  # 获取原图像的水平方向尺寸和垂直方向尺寸。
    color_warp1 = cv2.resize(color_warp, (int(0.2 * width), int(0.2 * height)), interpolation=cv2.INTER_CUBIC)
    height, width = newwarp.shape[:2]  # 获取原图像的水平方向尺寸和垂直方向尺寸。
    newwarp1 = cv2.resize(newwarp, (int(0.2 * width), int(0.2 * height)), interpolation=cv2.INTER_CUBIC)
    return color_binary1, warp1, color_warp1, newwarp1

def preprocess(img):
    global last_left
    global last_right
    global left_fit
    global right_fit
    global counter
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #修正畸形
    undistort = undistorted(img, mtx, dist)
    color_binary = combine_thresh(undistort, (0, 255), (0, 255), (155, 255), (177, 255), (177, 255), (177, 255))

    # INTER_AREA = 3
    # INTER_BITS = 5
    # INTER_BITS2 = 10
    # INTER_CUBIC = 2
    # INTER_LANCZOS4 = 4
    # INTER_LINEAR = 1

    #空间透视变换
    warp = cv2.warpPerspective(color_binary, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA)
    # Plot the examples绘制实例
    #fig = plt.figure()

    if counter == 0:
        left_fit, right_fit, _ = fit_lines(warp, plot=False)
    else:
        try:
            left_fit, right_fit, _ = fit_lines_continous(warp, left_fit, right_fit, warp, plot=False)
        except TypeError:
            pass
    # Do Sanity check

    #status = sanity_check(left_fit, right_fit)

    # Decide if to use calculated points
    if sanity_check(left_fit, right_fit) == True:#清醒检查
        last_left, last_right = left_fit, right_fit
        counter += 1
    else:  # Use the last realible fit
        left_fit, right_fit = last_left, last_right

    result,newwarp,color_warp = warp_perspective_back(undistort, warp, left_fit, right_fit, Minv)

    color_binary1, warp1, color_warp1, newwarp1 = resize(color_binary,warp,color_warp,newwarp)


    plt.subplot(2,2,1)#一行两列第一个图
    plt.imshow(color_binary1)
    plt.title('Before',color='blue')
    plt.subplot(2,2,2)
    plt.imshow(warp1)
    plt.title('After',color='blue')
    plt.subplot(2,2,3)
    plt.imshow(color_warp1)
    plt.title('After',color='blue')
    plt.subplot(2,2,4)
    plt.imshow(newwarp1)
    plt.title('After',color='blue')
    plt.suptitle("rectangle transformation",FontSize='12')
    plt.show()

#     #图片合并
#     htitch = np.hstack((color_binary1, warp1))
#     plt.imshow(htitch)
#     plt.show()

    result = add_text(result, left_fit, right_fit)


    # 使用numpy数组切片，获取ROI(region of interest)，也就是感兴趣区域
    imgeROI = result[10:10 + newwarp1.shape[0], 0:0 + newwarp1.shape[1]]
    cv2.addWeighted(imgeROI, 1, newwarp1, 0.6, 0, imgeROI)
    imgeROI = result[10:10 + color_warp1.shape[0], 250:250 + color_warp1.shape[1]]
    cv2.addWeighted(imgeROI, 1, color_warp1, 0.6, 0, imgeROI)
    # imgeROI = result[10:10 + warp1.shape[0], 500:500 + warp1.shape[1]]
    # cv2.addWeighted(imgeROI, 1, warp1, 0.6, dst, imgeROI)

    print(warp1.shape,warp1.shape,color_warp1.shape,newwarp1.shape)

    plt.imshow(result)
    plt.show()

    return result


#==================================================================
path = '../camera_cal'
cal_imgs = os.listdir(path)

(nx, ny) = (9, 6)
img_points = []
obj_points = []
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

for name in cal_imgs:
    img = cv2.imread(path + '/' + name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        img_points.append(corners)
        obj_points.append(objp)
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # plt.imshow(img)
        # plt.show()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape, None, None)
#==================================================================
#获取矩形变换矩阵
im_shape=(img.shape[1],img.shape[0])
src = np.float32([[293, 668], [587, 458], [703, 458], [1028, 668]])
dst = np.float32([[310, im_shape[1]], [310, 0], [950, 0], [950, im_shape[1]]])
M=cv2.getPerspectiveTransform(src, dst)#矩形装换矩阵
Minv=cv2.getPerspectiveTransform(dst, src)

#======================================================================

counter=0
left_fit,right_fit=None,None
last_left,last_right=None, None
output = '../test_videos_output/out_project_video.mp4'
#clip1 = VideoFileClip("../test_videos/harder_challenge_video.mp4")
clip1 = VideoFileClip("../test_videos/project_video.mp4").subclip(20,23)#获取00:00:20 -- 00:00:28秒的视频数据

out_clip = clip1.fl_image(preprocess) #NOTE: this function expects color images!!
out_clip.write_videofile(output, audio=False)
print(counter)



