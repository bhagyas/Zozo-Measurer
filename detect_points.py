#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from markers import find_marker_ellipses, unskew_point, get_point_id, collect_points, is_small_point
from markers import find_best_10_confidences
import cv2
import sys
import numpy as np


def scale_preview(preview_img, max_preview_size=(1800, 960)):
    scale_fac = min(max_preview_size[1] / preview_img.shape[0], max_preview_size[0] / preview_img.shape[1], 1)
    scaled_im = cv2.resize(preview_img, (int(preview_img.shape[1] * scale_fac), int(preview_img.shape[0] * scale_fac)),
                           interpolation=cv2.INTER_CUBIC)
    return scaled_im, scale_fac

def generate_obj_file(point_ids, positions, distances, output_path):
    """
    Generate a .obj file from the detected points.

    :param point_ids: List of point IDs
    :param positions: List of 2D positions (x, y)
    :param distances: List of distances from the camera
    :param output_path: Path to save the .obj file
    """
    with open(output_path, 'w') as f:
        # Write vertices
        for i, (pos, dist) in enumerate(zip(positions, distances)):
            x, y = pos
            z = dist / 100  # Convert distance to appropriate scale
            f.write(f"v {x} {y} {z}\n")

        # Write faces (triangles)
        # This is a simple triangulation and may not be optimal for all point configurations
        for i in range(len(point_ids) - 2):
            for j in range(i + 1, len(point_ids) - 1):
                for k in range(j + 1, len(point_ids)):
                    # Add 1 to vertex indices because OBJ files are 1-indexed
                    f.write(f"f {i+1} {j+1} {k+1}\n")

def generate_point_cloud_obj(point_ids, positions, distances, confidences, output_path, sphere_radius=8.5, sphere_resolution=8, confidence_threshold=0.3):
    """
    Generate a point cloud .obj file with small red spheres for each point with confidence > 0.3.

    :param point_ids: List of point IDs
    :param positions: List of 2D positions (x, y)
    :param distances: List of distances from the camera
    :param confidences: List of confidence values for each point
    :param output_path: Path to save the .obj file
    :param sphere_radius: Radius of each sphere representing a point
    :param sphere_resolution: Resolution of each sphere (number of segments)
    :param confidence_threshold: Minimum confidence value for a point to be included
    """
    def generate_sphere(center, radius, resolution):
        vertices = []
        faces = []
        for i in range(resolution):
            for j in range(resolution):
                theta = np.pi * i / (resolution - 1)
                phi = 2 * np.pi * j / (resolution - 1)
                x = center[0] + radius * np.sin(theta) * np.cos(phi)
                y = center[1] + radius * np.sin(theta) * np.sin(phi)
                z = center[2] + radius * np.cos(theta)
                vertices.append((x, y, z))

        for i in range(resolution - 1):
            for j in range(resolution - 1):
                v1 = i * resolution + j
                v2 = i * resolution + (j + 1)
                v3 = (i + 1) * resolution + (j + 1)
                v4 = (i + 1) * resolution + j
                faces.append((v1, v2, v3))
                faces.append((v1, v3, v4))

        return vertices, faces

    with open(output_path, 'w') as f:
        f.write("mtllib sphere.mtl\n")
        f.write("usemtl red_sphere\n")

        vertex_offset = 1
        for point_id, pos, dist, conf in zip(point_ids, positions, distances, confidences):
            if conf > confidence_threshold:
                x, y = pos
                z = dist * 5  # Convert distance to appropriate scale
                sphere_vertices, sphere_faces = generate_sphere((x, y, z), sphere_radius, sphere_resolution)

                for vertex in sphere_vertices:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

                for face in sphere_faces:
                    f.write(f"f {face[0] + vertex_offset} {face[1] + vertex_offset} {face[2] + vertex_offset}\n")

                vertex_offset += len(sphere_vertices)

    # Create a simple MTL file for the red material
    with open(output_path.replace('.obj', '.mtl'), 'w') as f:
        f.write("newmtl red_sphere\n")
        f.write("Kd 1.0 0.0 0.0\n")  # Red diffuse color

def detect_points(img):
    skewed_points, origins, ellipses = find_marker_ellipses(img)
    unskewed_points = [unskew_point(skewed_points[i], origins[i], ellipses[i])
                       for i in range(len(skewed_points))]
    point_ids = []
    confidences = []
    positions = []
    distances = []
    for i in range(len(unskewed_points)):
        positions.append(ellipses[i][0])
        confidence_for_small_point = is_small_point(unskewed_points[i], ellipses[i])
        if confidence_for_small_point > 0.3:
            point_ids.append(0)
            confidences.append(confidence_for_small_point)
        else:
            point_id, confidence = get_point_id(unskewed_points[i], ellipses[i])
            point_ids.append(point_id)
            confidences.append(confidence)
    _, best_10_indexes = find_best_10_confidences(
        [confidences[i] if point_ids[i] > 0 else 0 for i in range(len(confidences))])
    ellipses_size_sum = 0
    for i in best_10_indexes:
        ellipses_size_sum += max(ellipses[i][1])
    avg_big_point_size = ellipses_size_sum / 10
    for i in range(len(unskewed_points)):
        if point_ids[i] == 0:
            distances.append((max(ellipses[i][1]) / avg_big_point_size) * 400)
        else:
            distances.append((max(ellipses[i][1]) / avg_big_point_size) * 200)
    raw_data = [{
        "skewed_point": skewed_points[i],
        "unskewed_point": unskewed_points[i],
        "origin": origins[i],
        "ellipse": ellipses[i],
        "confidence": confidences[i],
        "point_id": point_ids[i],
        "point_type": "small_point" if point_ids[i] == 0 else "big_point",
        "position": positions[i],
        "distance": distances[i]
    } for i in range(len(unskewed_points))]
    return point_ids, confidences, positions, distances, raw_data


if __name__ == '__main__':
    args = sys.argv
    input_file_name = args[1]
    im = cv2.imread(input_file_name)
    point_ids, confidences, positions, distances, raw_data = detect_points(im)

    p_coll_img = collect_points((64, 64), raw_data)
    if min(p_coll_img.shape[0:1]) > 0:
        # cv2.imshow('Collected Points', p_coll_img)
        cv2.imwrite("./output/collected_points.png", p_coll_img)

    im_draw = im.copy()
    for i in range(len(point_ids)):
        cv2.ellipse(im_draw, raw_data[i]["ellipse"],
                    color=(0, int(confidences[i] * 255), int((1 - confidences[i]) * 255)),
                    thickness=3)
        cv2.line(im_draw,
                 (int(positions[i][0]-5), int(positions[i][1]-5)),
                 (int(positions[i][0]+5), int(positions[i][1]+5)),
                 color=(255, 0, 0), thickness=3)
        cv2.line(im_draw,
                 (int(positions[i][0] - 5), int(positions[i][1] + 5)),
                 (int(positions[i][0] + 5), int(positions[i][1] - 5)),
                 color=(255, 0, 0), thickness=3)
        if confidences[i] > 0.3:
            cv2.putText(im_draw, str(point_ids[i]),
                        (int(positions[i][0] + raw_data[i]["ellipse"][1][0] * 0.5),
                         int(positions[i][1] - raw_data[i]["ellipse"][1][0] * 0.2)),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2,
                        color=(0, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.putText(im_draw, "{:02.2f}".format(distances[i]/100),
                        (int(positions[i][0] + raw_data[i]["ellipse"][1][0] * 0.4),
                         int(positions[i][1] + raw_data[i]["ellipse"][1][0] * 0.8)),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2,
                        color=(255, 0, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA)
    cv2.imwrite("./output/point_positions.png", im_draw)
    im_preview, scale_factor = scale_preview(im_draw)
    # write the preview image to the disk
    cv2.imwrite("./output/point_positions_preview.png", im_preview)

    generate_point_cloud_obj(point_ids, positions, distances, confidences, "./output/mesh.obj")


    # we are using the following code to display the preview image only when running the script in the interactive mode
    # cv2.imshow('Point positions', im_preview)
    # key = cv2.waitKey(0)
    # cv2.destroyAllWindows()

