from typing import Optional

from type_hints import KeypointData
from helpers.format_box import format_box


def get_interpolated_point(i: int, points: list[tuple[float, float, float]], threshold: float, default: tuple[float, float] = (0,0)) -> tuple[float, float]:
    '''Returns for a point, if confidence lower than threshold, the interpolation of the next and previous point with confidence over threshold'''
    next_point = next(((point[0], point[1]) for point in points[(i+1):] if point[2] > threshold), None)
    prev_point = next(((point[0], point[1]) for point in reversed(points[:i]) if point[2] > threshold), None)
    return ((prev_point[0]+next_point[0])/2, (prev_point[1]+next_point[1])/2) if (prev_point is not None and next_point is not None) else (
        next_point if next_point is not None else (
            prev_point if prev_point is not None else default
        )
    )

def interpolate(keypoints: list[tuple[float, float, float]], threshold: float, max_missing_percent: float, default: Optional[tuple[float, float]] = None) -> Optional[list[tuple[float, float]]]:
    '''For a list of frames of points, replaces those with confidence lower than threshold with the interpolation of the next and previous point with confidence over threshold'''
    # keypoints contains [x,y,z] for each frame 
    missing = sum(1 for point in keypoints if point[2] < threshold)
    if missing / len(keypoints) <= max_missing_percent:
        return [
            (each[0], each[1]) if each[2] > threshold else get_interpolated_point(i, keypoints, threshold) for i, each in enumerate(keypoints)
        ]
    return None if not default else [default for _ in keypoints]

def get_keypoints_centers(keypoints: list[KeypointData], threshold: float = 0.5, max_missing_percent: float = 0.05) -> list[tuple[float, float]]:
    nose = [(each['keypoints'][0], each['keypoints'][1], each['keypoints'][2]) for each in keypoints]
    centers = interpolate(nose, threshold, max_missing_percent)
    if centers is None:
        neck = [(each['keypoints'][18*3], each['keypoints'][18*3+1], each['keypoints'][18*3+2]) for each in keypoints]
        centers = interpolate(neck, threshold, max_missing_percent)
    if centers is None:
        boxes = [format_box(each['box']) for each in keypoints]
        centers = [(box['x1']+box['width']/2, box['y1']+box['height']/2) for box in boxes]
    return centers
