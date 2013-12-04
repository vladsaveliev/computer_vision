from itertools import count
import cv2 as cv
import numpy as np

def put_points(fpath):
    video = cv.VideoCapture(fpath)
    _, frame = video.read()
    cv.imshow('checkers', frame)

    field_angle_points = set()

    def onclick(e, x, y, flags, _):
        if flags & cv.EVENT_FLAG_LBUTTON:
            print (x, y)
            field_angle_points.add((x, y))
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
            cv.imshow('checkers', frame)

    cv.setMouseCallback('checkers', onclick)
    cv.waitKey()
    return field_angle_points

def draw_points(frame, pt1, pt2, pt3, pt4):
    cv.circle(frame, pt1, 1, (0, 0, 255), -1)    # red
    cv.circle(frame, pt2, 1, (255, 0, 0), -1)    # blue
    cv.circle(frame, pt3, 1, (255, 0, 255), -1)  # purple
    cv.circle(frame, pt4, 1, (0, 255, 0), -1)    # green

# board size
w, h = 480, 480
cell_w = w / 8
cell_h = h / 8

def Y((b, g, r, _)):
    #return 0.299 * r + 0.587 * g + 0.114 * b
    return 0.299 * r + 0.587 * g + 0.114 * b

def R((b, g, r, _)):
    return r / g + r / b

def converted(src_fpath, frame_points):
    board_points = [(w - 1, h - 1), (w, 0), (0, 0), (0, h - 1)]
    video = cv.VideoCapture(fpath)
    M = cv.getPerspectiveTransform(
        src=np.array(frame_points, np.float32),
        dst=np.array(board_points, np.float32))
    corrected = np.empty(shape=(w, h, 3), dtype=np.uint8)

    for i in count(1):
        print i
        ok, frame = video.read()
        if not ok: break

        cv.warpPerspective(src=frame, M=M, dst=corrected, dsize=(w, h))
        cv.waitKey(1)
        yield corrected

def convert(src_fpath, dst_fpath, frame_points):
    board_points = [(w - 1, h - 1), (w, 0), (0, 0), (0, h - 1)]
    video = cv.VideoCapture(fpath)
    M = cv.getPerspectiveTransform(
        src=np.array(frame_points, np.float32),
        dst=np.array(board_points, np.float32))
    corrected = np.empty(shape=(w, h, 3), dtype=np.uint8)

    output = cv.VideoWriter(dst_fpath, cv.cv.CV_FOURCC(*'XVID'), 24.0, (w, h))

    for i in range(500):
        ok, frame = video.read()
        if not ok: break

        cv.warpPerspective(src=frame, M=M, dst=corrected, dsize=(w, h))
        #draw_points(frame, *frame_points)
        #draw_points(corrected, *board_points)
        #for i in range(0, 8):
        #    for j in range(0, 8):
        #        cv.rectangle(corrected, (i * cell_h, j * cell_w), ((i + 1) * cell_h, (j + 1) * cell_w), color=(0, 200, 0))

        #cv.imshow('video', frame)
        #cv.imshow('corrected', corrected)
        output.write(corrected)
        #if cv.waitKey() in (ord('q'), 27): break

reds = []
whites = []

color0 = []
color10 = []
color20 = []
def check_red(color):
    if sum(color) < 300:
        #reds.append(color)
        #print color[0],
        #print float(color[2]) / float(color[0]),
        #print float(color[1]) / float(color[0])
        return True
    #return 29 < color[0] < 56 and \
    #       26 < color[1] < 60 and \
    #       70 < color[2] < 140

def check_white(color):
    if sum(color) >= 300:
    #    whites.append(color)
    #    color0.append(color[0])
    #    color10.append(float(color[1]) / float(color[0]))
    #    color20.append(float(color[2]) / float(color[0]))
        return True
    #return 75 < color[0] < 180 and \
    #       0.83 < color[1] / float(color[0]) < 1.2 and \
    #       0.80 < color[2] / float(color[0]) < 1.8

def analyze(converted):
    frame = next(converted)

    # we can set a common normal cell color for all cells
    #normal_white = cv.mean(frame[cell_h + 15:cell_h + cell_h - 10,
    #                             cell_w + 15:cell_w + cell_w - 10])
    #normal_black = cv.mean(frame[cell_h * 3 + 15:cell_h * 4 - 10,
    #                             cell_w * 2 + 15:cell_w * 3 - 10])

    # or pick different normal colors for each cell
    #normal_colors = []
    #for i in range(0, 8):
    #    normal_colors.append([])
    #    for j in range(0, 8):
    #        if (i + j) % 2 == 0:
    #            #normal_colors[i].append(cv.mean(
    #            #    frame[cell_h * i + 15:cell_h * (i + 1) - 10,
    #            #          cell_w * j + 15:cell_w * (j + 1) - 10]))
    #            normal_colors[i].append(normal_white)
    #        else:
    #            #x, y = j * cell_w + 8, i * cell_h + 15
    #            #color = frame[x, y]
    #            normal_colors[i].append(normal_black)
    #            #cv.circle(frame, (x, y), 1, (255, 0, 0), 1)

    blc_k = 1
    wht_k = 2
    red_k = 20
    red_k_bottom = 0
    #wht_piece_cell_color = cv.mean(frame[wht_k : cell_h - wht_k, w - cell_w + wht_k : w - wht_k])
    #red_piece_cell_color = cv.mean(frame[h - cell_h + red_k : h - red_k, red_k : cell_w - red_k])
    wht_y = 0.
    wht_r = 0.
    red_y = 0.
    red_r = 0.
    blc_y = 0.
    for i in range(0, 8):
        print
        for j in range(0, 8):
            if (i + j) % 2 == 1:
                if i < 3:
                    color = cv.mean(frame[i * cell_h + wht_k : (i + 1) * cell_h - wht_k,
                                          j * cell_w + wht_k : (j + 1) * cell_w - wht_k])
                    wht_y += Y(color)
                    wht_r += R(color)
                elif i > 4:
                    color = cv.mean(frame[i * cell_h + red_k : (i + 1) * cell_h - red_k_bottom,
                                          j * cell_w + red_k : (j + 1) * cell_w - red_k])
                    red_y += Y(color)
                    red_r += R(color)
                else:
                    color = cv.mean(frame[i * cell_h + blc_k : (i + 1) * cell_h - blc_k,
                                          j * cell_w + blc_k : (j + 1) * cell_w - blc_k])
                    print Y(color),
                    blc_y += Y(color)

    wht_y /= 12.
    wht_r /= 12.
    red_y /= 12.
    red_r /= 12.
    blc_y /= 8.
    print
    print 'wht_y', wht_y
    print 'wht_r', wht_r
    print 'red_y', red_y
    print 'red_r', red_r
    print 'blc_y', blc_y
    #frame[h - cell_h + k : h - k, k : cell_w - k] = 0, 0, 200
    #frame[k : cell_h - k, w - cell_w + k : w - k] = 200, 0, 0
    #cv.rectangle(frame, (cell_h - k, w - k), (k, w - cell_w + k), color=(0, 0, 200))
    #cv.rectangle(frame, (h - cell_h + k, k), (h - k, cell_w - k), color=(200, 0, 0))

    # initiate a board
    board = np.zeros(shape=(w, h, 3), dtype=np.uint8)
    for i in range(0, 8):
        for j in range(0, 8):
            cv.rectangle(board,
                pt1=(i * cell_w, j * cell_h),
                pt2=((i + 1) * cell_w, (j + 1) * cell_h),
                thickness=-1,
                color=(255, 255, 255) if (i + j) % 2 == 0 else (0, 0, 0)),
                #color=map(int, normal_colors[i][j]))
                #color=normal_white if (i + j) % 2 == 0 else normal_black)

    #check_piece_pos_x, check_piece_pos_y = cell_w / 2, cell_h - 20
    #for i in range(0, 8):
    #    for j in range(0, 8):
    #        cv.circle(frame, (i * cell_w + check_piece_pos_x, j * cell_h + check_piece_pos_y), 1, (0, 0, 255))
    #        cv.circle(board, (i * cell_w + check_piece_pos_x, j * cell_h + check_piece_pos_y), 1, (0, 0, 255))

    resulted_video = cv.VideoWriter('result.avi', cv.cv.CV_FOURCC(*'XVID'), 24.0, (w, h * 2), isColor=True)
    for _ in count(1):
        print _
        next_board = board.copy()

        for i in range(0, 8):
            print
            for j in range(0, 8):
                # detect pieces
                #color = cv.mean(frame[i * cell_h + cell_h / 2:(i + 1) * cell_h - 5:,
                #                      j * cell_w + cell_w / 2:(j + 1) * cell_w - cell_w / 3] )
                if (i + j) % 2 == 0:
                    continue

                check_wht_cell_color = cv.mean(
                    frame[i * cell_h + wht_k : (i + 1) * cell_h - wht_k,
                          j * cell_w + wht_k : (j + 1) * cell_w - wht_k])
                check_red_cell_color = cv.mean(
                    frame[i * cell_h + red_k : (i + 1) * cell_h - red_k_bottom,
                          j * cell_w + red_k : (j + 1) * cell_w - red_k])
                check_blc_cell_color = cv.mean(
                    frame[i * cell_h + blc_k : (i + 1) * cell_h - blc_k,
                          j * cell_w + blc_k : (j + 1) * cell_w - blc_k])

                ##frame[i * cell_h + cell_h / 2:(i + 1) * cell_h - 5:,
                #       j * cell_w + cell_w / 2:(j + 1) * cell_w - cell_w / 3] = 255, 0, 0
                #diff = cv.absdiff(color, normal_colors[i][j])
                #red_diff = cv.absdiff(check_red_cell_color, red_piece_cell_color)
                #wht_diff = cv.absdiff(check_wht_cell_color, wht_piece_cell_color)
                #print '-'.join(map(str, map(int, wht_diff))) + ' ',
                check_wht_y = Y(check_wht_cell_color)
                check_wht_r = R(check_wht_cell_color)
                check_red_y = Y(check_red_cell_color)
                check_red_r = R(check_red_cell_color)
                check_blc_y = Y(check_blc_cell_color)
                wht_y_diff = abs(check_wht_y - wht_y)
                wht_r_diff = abs(check_wht_r - wht_r)
                red_y_diff = abs(check_red_y - red_y)
                red_r_diff = abs(check_red_r - red_r)
                blc_y_diff = abs(check_blc_y - blc_y)

                if red_r_diff < 1.2:
                    cv.circle(next_board,
                              (j * cell_w + cell_w / 2, i * cell_h + cell_h / 2),
                              radius=cell_w / 3,
                              thickness=-1,
                              color=(0, 0, 255))

                elif wht_r_diff < 0.1 and wht_y_diff < 20:
                    cv.circle(next_board,
                              (j * cell_w + cell_w / 2, i * cell_h + cell_h / 2),
                              radius=cell_w / 3,
                              thickness=-1,
                              color=(255, 255, 255))

                elif blc_y_diff < 15:
                    pass

                else:
                    next_board[i * cell_h : (i + 1) * cell_h,
                               j * cell_w : (j + 1) * cell_w] = check_blc_cell_color[:3]
                #cv.rectangle(frame,
                #             (i * cell_h + k, j * cell_w + k),
                #             ((i + 1) * cell_h - k, (j + 1) * cell_w - k),
                #             color=(0, 200, 0))

        result = np.concatenate([frame, next_board])
        cv.imshow('result', result)
        resulted_video.write(result)
        #if cv.waitKey() in (ord('q'), 27): break
        cv.waitKey(1)
        frame = next(converted)
        if frame is None: break

    #print max(color0), min(color0)
    #print max(color10), min(color10)
    #print max(color20), min(color20)

if __name__ == '__main__':
    fpath = 'checkers.mp4'
    corrected_fpath = 'corrected.avi'

    #points = put_points(fpath)

    #         red        blue       purple       green
    #        [(w - 1, h - 1), (w, 0), (0, 0), (0, h - 1)]
    #points = [(313, 226), (163, 636), (1155, 627), (997, 217)]
    points = [(313, 230), (161, 633), (1154, 627), (994, 223)]
    #convert(fpath, corrected_fpath, points)

    analyze(converted(fpath, points))