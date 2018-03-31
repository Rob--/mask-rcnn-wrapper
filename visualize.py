import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from skimage.measure import find_contours
import colorsys

color_dict = {}

def name_to_color(name):
    if name in color_dict:
        return color_dict[name]

    color = tuple(int(i * 255) for i in colorsys.hls_to_rgb(np.random.rand() * 360, 0.5, 1))
    color_dict[name] = color
    return color

def random_colors(N):
    np.random.seed(1)
    return [tuple(255 * np.random.rand(3)) for _ in range(N)]

def score_to_color(score):
    h = (120 * score) / 360.0
    s = 1
    l = 0.5
    return tuple(int(i * 255) for i in colorsys.hls_to_rgb(h, l, s))

def float_tuple_to_int(t):
    return tuple(int(n) for n in t) + (255,)

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def draw_line(image, point1, point2, color, thickness=1, style='dotted', gap=20):
    hypotenuse = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** .5
    points = []

    for i in np.arange(0, hypotenuse, gap):
        r = i / hypotenuse
        x = int((point1[0] * (1 - r) + point2[0] * r) + .5)
        y = int((point1[1] * (1 - r) + point2[1] * r) + .5)
        p = (x, y)
        points.append(p)

    if style == 'dotted':
        for p in points:
            cv2.circle(image, p, thickness, color, -1)
    else:
        e = points[0]
        i = 0
        for p in points:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(image, s, e, color, thickness)
            i = i + 1


def draw_polygon(img, points, color, thickness=1, style='dotted', gap=20):
    # Goes through each corner and joins them up
    current = points[0]
    points.append(points.pop(0))
    # points = np.concatenate(points, points.pop(0))
    for p in points:
        destination = current
        current = p
        draw_line(img, current, destination, color, thickness, style, gap)


def draw_rectangle(img, point1, point2, color, thickness=1, style='dotted', gap=20):
    # Takes the two boundary points and converts them to their 4 corners
    points = [point1, (point2[0], point1[1]), point2, (point1[0], point2[1])]
    draw_polygon(img, points, color, thickness, style, gap)

def draw_pil_rectangle(draw, point1, point2, color, width=1):
    for i in range(width):
        start = (point1[0] - i, point1[1] - i)
        end = (point2[0] + i, point2[1] + i)
        draw.rectangle((start, end), outline=color)

def draw_text(draw, caption, point, color):
    x = point[0]
    y = point[1]

    font = ImageFont.truetype('arial.ttf', 14)

    try:
        font = ImageFont.truetype('Hack-Bold.ttf', 14)
    except OSError:
        pass

    color = float_tuple_to_int(color)

    # Add the outline
    for i in range(2):
        direction = (-1) ** i
        draw.text((x + direction, y - 16), caption, font=font, fill=(0, 0, 0))
        draw.text((x, y - 16 + direction), caption, font=font, fill=(0, 0, 0))

    draw.text((x, y - 16), caption, font=font, fill=color)

def display_instances(image, boxes, masks, ids, names, scores, showMasks, showBoxes):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if n_instances:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.1f}%'.format(label, score * 100) if score else label

        color = name_to_color(label) # score_to_color(score)

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw_text(draw, caption, (x1, y1), color)

        if showBoxes:
            # Use PIL to draw the rectangle
            draw_pil_rectangle(draw, (x1, y1), (x2, y2), float_tuple_to_int(color), 2)

            # Use CV2 to draw the rectangle
            # draw_rectangle(image, (x1, y1), (x2, y2), color, 1, 'dashed', 6)

        image = np.array(img_pil)

        if showMasks:
            mask = masks[:, :, i]
            image = apply_mask(image, mask, color)

            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                cv2.polylines(image, np.int32([verts]), False, color, thickness=1, lineType=cv2.LINE_AA)

        # Draw text using CV2
        # image = cv2.putText(
        #     image, caption, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1,
        # )
    return image

# w, h = verts.shape
# for i in range(0, w, 2):
#     if i + 2 > w:
#         break
#
#     x1 = verts[i][0]
#     y1 = verts[i][1]
#     x2 = verts[i + 1][0]
#     y2 = verts[i + 1][1]
#     draw.line([(int(x1), int(y1)) , (int(x2), int(y2))], fill=float_tuple_to_int(color), width=2)
#     cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=1, lineType=cv2.LINE_AA)
