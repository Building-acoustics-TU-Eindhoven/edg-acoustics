"""
Loads a mesh and then finds the simplices that contain some points.
"""

import os
import sys
import numpy
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Find the element with this point
points_to_find = numpy.array([[0.1, 0.2], [0.7, 0.2], [0.8, 0.7]])

# Generate the mesh in Scipy
points = numpy.array([[0, 0], [0, 1.1], [1, 0], [0.5, 0.5], [1, 1]])
tri = Delaunay(points)

print('EToV Scipy: ')
print(tri.simplices)

# Find the location of the points
where_points = tri.find_simplex(points_to_find)
print('Location in Scipy mesh: ')
print(where_points)

# Plot the Scipy mesh and the points to search
plt.figure()
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.plot(points_to_find[:, 0], points_to_find[:, 1], 'x')

# Plot the nodes' index
for point_idx, point in enumerate(points):
    plt.text(point[0], point[1], str(point_idx))

# Plot the elements' index
for element_idx, element in enumerate(tri.simplices):
    barycenter = points[element, :].mean(axis=0)
    plt.text(barycenter[0], barycenter[1], str(element_idx))

# Plot the point's element index
for point_idx, point in enumerate(points_to_find):
    plt.text(point[0], point[1], str(where_points[point_idx]))

plt.show()

# Construct a new mesh with the same nodes
# The triangles are the same, just set element 1 as element 0 and vice-versa
simplices = numpy.array([[3, 1, 0], [2, 3, 0], [4, 3, 2], [3, 4, 1]], dtype=numpy.int32)
tri.simplices = simplices
tri.nsimplex = 4

print('EToV new: ')
print(tri.simplices)

where_points = tri.find_simplex(points_to_find)
print('Location in new mesh: ')
print(where_points)

# Plot the new mesh and the points to search
plt.figure()
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.plot(points_to_find[:, 0], points_to_find[:, 1], 'x')

# Plot the nodes' index
for point_idx, point in enumerate(points):
    plt.text(point[0], point[1], str(point_idx))

# Plot the elements' index
for element_idx, element in enumerate(tri.simplices):
    barycenter = points[element, :].mean(axis=0)
    plt.text(barycenter[0], barycenter[1], str(element_idx))

# Plot the point's element index
for point_idx, point in enumerate(points_to_find):
    plt.text(point[0], point[1], str(where_points[point_idx]))

plt.show()