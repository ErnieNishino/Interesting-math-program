from math import sqrt


def calculate_distance(point1: tuple[int, int], point2: tuple[int, int]):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def brute_force_closest_pair(points: list[tuple[int, int]]):
    min_distance = float('inf')
    closest_pair = None

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = calculate_distance(points[i], points[j])
            if distance < min_distance:
                min_distance = distance
                closest_pair = (points[i], points[j])

    return closest_pair


def closest_pair(points: list[tuple[int, int]]):
    points = sorted(points, key=lambda x: x[0])

    if len(points) <= 3:
        return brute_force_closest_pair(points)

    mid = len(points) // 2
    left_half = points[:mid]
    right_half = points[mid:]

    left_closest = closest_pair(left_half)
    right_closest = closest_pair(right_half)

    min_distance = min(calculate_distance(left_closest[0], left_closest[1]),
                       calculate_distance(right_closest[0], right_closest[1]))

    mid_strip_closest = min_distance_strip(points, min_distance)

    if mid_strip_closest is None:
        return min(left_closest, right_closest, key=lambda x: calculate_distance(x[0], x[1]))

    return min(left_closest, right_closest, mid_strip_closest, key=lambda x: calculate_distance(x[0], x[1]))


def min_distance_strip(points: list[tuple[int, int]], min_distance: float):
    mid_x = points[len(points) // 2][0]
    strip = [point for point in points if mid_x - min_distance <= point[0] <= mid_x + min_distance]
    strip = sorted(strip, key=lambda x: x[1])

    strip_closest = None

    for i in range(len(strip)):
        for j in range(i + 1, min(i + 7, len(strip))):
            distance = calculate_distance(strip[i], strip[j])
            if distance < min_distance:
                min_distance = distance
                strip_closest = (strip[i], strip[j])

    return strip_closest


if __name__ == "__main__":
    points = [(4, 6), (8, 16), (18, 20), (10, 12), (2, 18), (14, 8), (16, 10), (6, 14), (12, 6), (10, 16), (12, 18)]
    result = closest_pair(points)
    print(The closest points:", result)
