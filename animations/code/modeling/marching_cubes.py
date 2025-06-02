from manimlib import *
from klampt.math import so3

manim_config.camera.background_color = WHITE

class Cube(Group):
    # https://sketchfab.com/3d-models/15-cases-marching-cubes-representation-78927bfd96694fccbc064bb5fe820f3d
    # https://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html
    #                 v7_______e6_____________v6
    #                  /|                    /|
    #                 / |                   / |
    #              e7/  |                e5/  |
    #               /___|______e4_________/   |
    #            v4|    |                 |v5 |e10
    #              |    |                 |   |
    #              |    |e11              |e9 |
    #            e8|    |                 |   |
    #              |    |_________________|___|
    #              |   / v3      e2       |   /v2
    #              |  /                   |  /
    #              | /e3                  | /e1
    #              |/_____________________|/
    #              v0         e0          v1

    CUBE_VERTICES = np.array([
        [-0.5, -0.5, -0.5],  # 0
        [ 0.5, -0.5, -0.5],  # 1
        [ 0.5,  0.5, -0.5],  # 2
        [-0.5,  0.5, -0.5],  # 3
        [-0.5, -0.5,  0.5],  # 4
        [ 0.5, -0.5,  0.5],  # 5
        [ 0.5,  0.5,  0.5],  # 6
        [-0.5,  0.5,  0.5]   # 7
    ])

    # Edge → (start corner index, end corner index)
    CUBE_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # verticals
    ]

    def __init__(self, scene, scale=1.0, vertex_idx=[], triangles=[]):
        super().__init__()
        self.scene = scene
        self.edges = []
        self.vertices = []
        self.vertex_idx = []
        self.scale_val = scale
        self.rotation_matrix = so3.identity()

        self.triangles = []
        self.triangles_idx = []
        self._create_cube()
        self.add_triangles(triangles)
        self.add_vertices(vertex_idx)
        

    def _create_cube(self):
        # Draw cube edges
        for i, j in Cube.CUBE_EDGES:
            edge = Line(
                Cube.CUBE_VERTICES[i] * self.scale_val,
                Cube.CUBE_VERTICES[j] * self.scale_val,
                color=BLACK,
                stroke_width=1.0 * self.scale_val,
            )
            self.edges.append(edge)
            self.add(edge)

    def clear_vertices(self, animation_speed=0.0):
        # Clear previous vertices
        for dot in self.vertices:
            self.remove(dot)

        if animation_speed > 0.0:
            self.scene.play(
                *[FadeOut(dot) for dot in self.vertices],
                run_time=animation_speed
            )
        else:
            self.scene.remove(*self.vertices)

        self.vertices = []
        self.vertex_idx = []

    def add_vertices(self, vertex_idx, animation_speed=0.0):
        self.vertex_idx = vertex_idx.copy()

        # Visualize corners
        for idx, pos in enumerate(Cube.CUBE_VERTICES):
            o = 1.0 if idx in self.vertex_idx else 0.0
            r = 0.05 * self.scale_val
            dot = Sphere(radius=r, opacity=o, color="#62AFE0")
            # Get the rotation frame of the group
            dot_pos = np.array(so3.apply(self.rotation_matrix, pos)) * self.scale_val + self.get_center()
            dot.move_to(dot_pos)
            self.vertices.append(dot)

        if animation_speed > 0.0:
            self.scene.play(
                *[FadeIn(dot) for dot in self.vertices],
                run_time=animation_speed
            )
            self.add(*self.vertices)
        else:
            self.add(*self.vertices)

    def add_triangles(self, triangles):
        self.triangles_idx = triangles.copy()

        for t in triangles:
            triangle_points = []
            for edge in t:
                e = Cube.CUBE_EDGES[edge]
                v1 = Cube.CUBE_VERTICES[e[0]] * self.scale_val
                v2 = Cube.CUBE_VERTICES[e[1]] * self.scale_val
                triangle_points.append((v1 + v2)/2.0)
            triangle = Polygon(triangle_points[0], triangle_points[1], triangle_points[2], fill_color=YELLOW, fill_opacity=1.0, stroke_color=BLACK, stroke_width=1.0)
            self.triangles.append(triangle)
            self.add(triangle)

    def get_vertex_locations(self):
        return Cube.CUBE_VERTICES[self.vertex_idx]
    
    def rotate(
        self,
        angle,
        axis,
        **kwargs
    ):
        super().rotate(angle, axis=axis, **kwargs)
        axis = np.array(axis)/np.linalg.norm(axis)
        R = so3.rotation(axis, angle)
        self.rotation_matrix = so3.mul(R, self.rotation_matrix)

    def scale(self, scale):
        super().scale(scale)
        self._scale(scale)

    def _scale(self, scale):
        self.scale_val *= scale
        
    def custom_copy(self):
        # Create a new cube with the same parameters
        new_cube = Cube(
            self.scene,
            scale=self.scale_val,
        )
        new_cube.rotation_matrix = self.rotation_matrix.copy()
        new_cube.clear_vertices()
        new_cube.add_vertices(self.vertex_idx)
        new_cube.add_triangles(self.triangles_idx)
        new_cube.move_to(self.get_center())

        return new_cube

class MarchingCubes(InteractiveScene):
    def construct(self):
        # Animation script

        # Show 256 combos
        # Show 15 base cases rotating around
        # Minimize to the top banner
        # Show a few cubes intersecting
        # - Find the cube in the table that shows in the top banner
        # - Show the cube rotating
        # - Show the cube moving to the location of the intersection
        # Fast forward to the end without the intersection animations

        # Set up camera orientation
        self.camera.frame.reorient(phi_degrees=90, theta_degrees=0)

        # Initial orientation
        self.prev_ori_camera = self.camera.frame.get_orientation()

        self.lookup_table = self._create_lookup_table()
        self._animate_different_corners()
        self._show_base_cases()

        # Sphere
        sphere = Sphere(radius=1, color=BLUE, opacity=0.2)
        self.shape = sphere

        # Axes
        axes = ThreeDAxes(
            x_range=(-3, 3, 1.0),
            y_range=(-3, 3, 1.0),
            z_range=(-2, 2, 1.0)
        )
        axes.set_color(GREY)
        self.add(axes)
        self.add(sphere)

        self._sample_marching_cubes()
        self._write_info(
            """
            The smaller the cubes, the more accurate the approximation as illustrated by the following example.
            """,
            {}
        )
        self._full_marching_cubes(0.2)


    def _animate_different_corners(self):
        cube = Cube(self, 3.0)
        self.add(cube)

        # Define an updater function
        def rotate_cube(mob, dt):
            mob.rotate(PI/4 * dt, axis=OUT)  # rotate around z-axis
        
        # Add the updater
        cube.add_updater(rotate_cube)
        combos = [
            [0, 1, 2, 3],
            [0, 3, 4, 7],
            [0],
            [0, 3, 5, 6],
            [0, 1, 2],
            [0, 5],
            [0, 1, 2, 4],
            [0, 1, 5, 6],
        ]
        self._write_info(
            """
            When a surface intersects a cube, some vertices are inside the surface and others are outside. 
            There are 256 different ways this can happen, some of which are shown here.
            """, 
            {"256": BLUE}
        )
        
        for combo in combos:
            cube.clear_vertices(animation_speed=0.0)
            cube.add_vertices(combo, animation_speed=0.0)
            self.wait(1.5)

        # Remove the updater
        cube.remove_updater(rotate_cube)

        self.play(FadeOut(cube), run_time=1)
        self._delete_info()
        

    def _show_base_cases(self):
        # Show the base cases
        rows = 3
        cols = 5
        spacing = 1.8

        cubes = []
        # Arrange manually
        for i, cube in enumerate(self.lookup_table):
            row = i // cols
            col = i % cols
            x = (col - (cols - 1)/2) * spacing
            z = - (row - (rows - 1)/2) * spacing
            cube.move_to(np.array([x, 0, z]))
            cubes.append(cube)

        self.play(*[FadeIn(cube) for cube in cubes])
        self._write_info(
            """
            The 256 combinations can be reduced to 15 base cases, which are shown here. Each base case
            also has a polygonal surface that can be used to approximate the surface that contains the 
            highlighted vertices.
            """, 
            {"256": BLUE, "15": RED}
        )

        # Create target arrangement
        target_group = Group(*cubes).copy()
        target_group.arrange(
            direction=RIGHT,
            buff=1.0  # spacing between cubes
        ).scale(0.4)  # scale down the entire group

        target_group.move_to([0, 0, 3])  # move to top of screen
        
        # Animate each cube to its target position
        self.play(
            *[
                Transform(cube, target)
                for cube, target in zip(cubes, target_group)
            ],
            run_time=2
        )

        # Manually update scale. Hacky, but it works.
        for cube in cubes:
            cube._scale(0.4/cube.scale_val)

        self._delete_info()

        # Create and animate a large centered copy of the first cube
        large_cube = (cubes[1]).custom_copy()
        self.add(large_cube)
        self.play(
            large_cube.animate.scale(5).move_to(ORIGIN),
            run_time=1.5
        )
        large_cube._scale(5)

        # Show rotation
        self._write_info(
            """
            The 15 base cases can be used for mirrored vertices and can be rotated to satisfy all 256 combinations.
            """, 
            { "15": RED, "256": BLUE}
        )
        self._delete_info()

        self._write_info(
            """
            The base cube can be rotated around 6 different axes and 4 different angles. Some rotations are redundant.
            The following animation shows how a cube rotated around different axes can match different vertex
            combinations.
            """, 
            {"6": BLUE, "4": RED}
        )


        large_cube.add_updater(lambda m, dt: m.rotate((PI/2)/(1.0/dt) if dt > 0.0 else 0.0, axis=OUT) )
        self.wait(1.0)
        large_cube.clear_updaters()
        
        self.wait(0.5)

        large_cube.add_updater(lambda m, dt: m.rotate((PI/2)/(1.0/dt) if dt > 0.0 else 0.0, axis=UP))
        self.wait(1.0)
        large_cube.clear_updaters()

        self.wait(0.5)

        large_cube.add_updater(lambda m, dt: m.rotate((PI/2)/(1.0/dt) if dt > 0.0 else 0.0, axis=RIGHT))
        self.wait(1.0)
        large_cube.clear_updaters()

        self.wait(0.5)

        large_cube.add_updater(lambda m, dt: m.rotate((2 * PI / 3)/(1.0/dt) if dt > 0.0 else 0.0, axis=np.array([1.0, 1.0, 1.0])))
        self.wait(1.0)
        large_cube.clear_updaters()

        self.wait(0.5)
        self._delete_info()

        # Show mirroring
        self._write_info(
            """
            When mirroring or swapping the inside and outside vertices, the base case can still be used since the approximated surface
            is the same.
            """, 
            {}
        )
        mirror_idx = {0, 1, 2, 3, 4, 5, 6, 7} - set(large_cube.vertex_idx)
        large_cube.clear_vertices(animation_speed=3.0)
        large_cube.add_vertices(mirror_idx, animation_speed=3.0)
        self._delete_info()

        self.play(
            FadeOut(large_cube),
            run_time=1
        )

        
    def _sample_marching_cubes(self):
        scale = 0.5
        points = [
            [1, 0, 0],
            [1, 0, 0.5],
            [1, -0.5, 0.5],
            [1, -0.5, 0],
        ]

        # Move camera to a new view cinematically
        self.play(
            self.camera.frame.animate
            .scale(0.23)
            .move_to([1.3, -1, 0.5])
            .reorient(
                phi_degrees=70,  # Tilt down
                theta_degrees=38  # Rotate around
            ),  # Pull back and shift perspective
            run_time=2,
            rate_func=smooth
        )
        self._write_info("""
                Marching cubes works by finding the vertices of the cube inside the surface at a specific coordinate.
                Then finding the base case that when rotated or having its vertices mirrored, matches the vertices
                of the cube at the specific coordinate. Finally, the base case is placed at the specific coordinate.
                """,
                {}
        )
        self._delete_info()
        cubes = []
        for i, p in enumerate(points):
            vertex_idx = self._find_cube_at_point(*p, scale)
            cube = Cube(self, scale)
            cube.move_to(p)
            cube.clear_vertices()
            cube.add_vertices(vertex_idx, animation_speed=0.0)
            self.add(cube)

            self.play(
                cube.animate.move_to([2, 0, 0]),
                run_time=2
            )
            matching_cube, axis, angle = self._find_matching_cube(vertex_idx)
            matching_cube.scale(cube.scale_val/matching_cube.scale_val)
            self.add(matching_cube)
            self.play(
                matching_cube.animate.move_to([2, 0, 1]),
                run_time=2
            )
            self.play(
                Rotate(matching_cube, angle, axis),
                run_time=2
            )
            self.play(
                FadeOut(cube),
                run_time=1
            )
            self.play(
                matching_cube.animate.move_to(p),
                run_time=2
            )
            cubes.append(matching_cube)
            if i == len(points)-2:
                self.play(
                    self.camera.frame.animate
                    .scale(1.0/self.camera.frame.get_scale())
                    .move_to([0.0, 0.0, 0.0])
                    .reorient(
                        phi_degrees=90,  # Tilt down
                        theta_degrees=0  # Rotate around
                    ),  # Pull back and shift perspective
                    run_time=2,
                    rate_func=smooth
                )
        self.play(
            *[FadeOut(cube) for cube in cubes],
            run_time=1
        )
        self._write_info(
            """
            The final product of applying the marching cubes algorithm in a 3D grid that covers the surface is the following.
            """, 
            {}
        )
        self._full_marching_cubes(0.5)
        self._delete_info()


    def _full_marching_cubes(self, scale):
        cubes = []
        # Create a 3D grid of points
        x = np.arange(-1, 2, scale)
        y = np.arange(-1, 2, scale)
        z = np.arange(-1, 2, scale)
        X, Y, Z = np.meshgrid(x, y, z)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    vertex_idx = self._find_cube_at_point(X[i, j, k], Y[i, j, k], Z[i, j, k], scale)
                    if len(vertex_idx) == 0 or len(vertex_idx) == 8:
                        continue
                    matching_cube, axis, angle = self._find_matching_cube(vertex_idx)
                    
                    matching_cube.scale(scale/matching_cube.scale_val)
                    matching_cube.rotate(angle, axis=axis)
                    matching_cube.move_to([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                    cubes.append(matching_cube)

        self.play(*[FadeIn(cube) for cube in cubes], run_time=1)
        # Fade out the edges and vertices of the cubes
        for cube in cubes:
            cube.clear_vertices()
        self.play(*[FadeOut(e) for cube in cubes for e in cube.edges], run_time=1)
        for cube in cubes:
            for edge in cube.edges:
                cube.remove(edge)
        self.play(
            self.camera.frame.animate
            .scale(0.5/self.camera.frame.get_scale())
            .reorient(
                phi_degrees=70,
                theta_degrees=0
            ),
            run_time=2,
            rate_func=linear
        )
        self.camera.frame.add_updater(lambda m, dt: m.increment_theta(2.0*PI/(5.0/dt)) if dt > 0.0 else 0.0)
        self.wait(5)
        self.camera.frame.clear_updaters()
        self.play(*[FadeOut(cube) for cube in cubes], run_time=1)


    def _find_cube_at_point(self, x, y, z, scale):
        vertex_locations_scaled = Cube.CUBE_VERTICES * scale
        # Check if the point is inside the sphere
        vertex_idx = []
        for index, v in enumerate(vertex_locations_scaled):
            if ((x + v[0])**2 + (y + v[1])**2 + (z + v[2])**2)**0.5 < self.shape.radius:
                vertex_idx.append(index)
        if len(vertex_idx) > 4:
            # mirror the vertices
            vertex_idx = {0, 1, 2, 3, 4, 5, 6, 7} - set(vertex_idx)  
        return vertex_idx
    

    def _find_matching_cube(self, vertex_idx):
        # 24 possible rotations
        cube_rotations = [
            # Identity
            (np.array([0, 0, 1]),   0),

            # 90°, 180°, 270° about X
            (np.array([1, 0, 0]),  PI / 2),
            (np.array([1, 0, 0]),  PI),
            (np.array([1, 0, 0]),  3 * PI / 2),

            # 90°, 180°, 270° about Y
            (np.array([0, 1, 0]),  PI / 2),
            (np.array([0, 1, 0]),  PI),
            (np.array([0, 1, 0]),  3 * PI / 2),

            # 90°, 180°, 270° about Z
            (np.array([0, 0, 1]),  PI / 2),
            (np.array([0, 0, 1]),  PI),
            (np.array([0, 0, 1]),  3 * PI / 2),

            # 120°, 240° about body diagonals
            (np.array([1, 1, 1]),   2 * PI / 3),
            (np.array([1, 1, 1]),   4 * PI / 3),
            (np.array([-1, 1, 1]),  2 * PI / 3),
            (np.array([-1, 1, 1]),  4 * PI / 3),
            (np.array([1, -1, 1]),  2 * PI / 3),
            (np.array([1, -1, 1]),  4 * PI / 3),
            (np.array([1, 1, -1]),  2 * PI / 3),
            (np.array([1, 1, -1]),  4 * PI / 3),

            # 180° about face diagonals (edge centers)
            (np.array([0, 1, 1]),   PI),
            (np.array([0, -1, 1]),  PI),
            (np.array([1, 0, 1]),   PI),
            (np.array([-1, 0, 1]),  PI),
            (np.array([1, 1, 0]),   PI),
            (np.array([-1, 1, 0]),  PI),
        ]

        # Find the cube in the lookup table that matches the given vertices
        vertices = [tuple(Cube.CUBE_VERTICES[vert]) for vert in vertex_idx]  
        for cube in self.lookup_table:
            for axis, angle in cube_rotations:
                rotated_cubeVs = self._rotate_points(cube.get_vertex_locations(), axis, angle)
                if set(vertices) == set(rotated_cubeVs):
                    return (cube.custom_copy(), axis, angle)

    def _rotate_points(self, points, axis, angle):
        matrix = rotation_matrix(angle, axis)
        return [tuple(np.round(matrix.dot(p), 2)) for p in points]

    def _create_lookup_table(self):
        baseCubes = [
            {
                "vertices": [0, 2, 3, 6],
                "triangles": [
                    [0, 8, 11],
                    [0, 1, 5],
                    [0, 11, 5],
                    [11, 5, 6]
                ]
            },
            {
                "vertices": [0, 4, 6, 2],
                "triangles": [
                    [0, 3, 7],
                    [7, 4, 0],
                    [1, 2, 6],
                    [6, 5, 1]
                ]
            },
            {
                "vertices": [1, 4, 6],
                "triangles": [
                    [8, 4, 7],
                    [0, 1, 9],
                    [10, 5, 6]
                ]
            },
            {
                "vertices": [0, 1, 6],
                "triangles": [
                    [3, 1, 8],
                    [1, 8, 9],
                    [10, 5, 6]
                ]
            },
            {
                "vertices": [0, 6],
                "triangles": [
                    [8, 0, 3],
                    [10, 5, 6],
                ]
            },
            {
                "vertices": [1, 2, 3, 7],
                "triangles": [
                    [3, 0, 7],
                    [0, 10, 9],
                    [0, 10, 7],
                    [7, 6, 10]
                ]
            },
            {
                "vertices": [0, 2, 3, 7],
                "triangles": [
                    [8, 7, 6],
                    [8, 6, 0],
                    [0, 6, 10],
                    [0, 1, 10]  
                ]
            },
            {
                "vertices": [0, 2, 5, 7],
                "triangles": [
                    [0, 3, 8],
                    [6, 7, 11],
                    [4, 9, 6],
                    [1, 2, 10]
                ]
            },
            {
                "vertices": [1, 2, 3, 4],
                "triangles": [
                    [8, 7, 4],
                    [0, 3, 9],
                    [3, 11, 9],
                    [9, 10, 11]
                ]
            },
            {
                "vertices": [0, 1, 2, 3],
                "triangles": [
                    [8, 9, 11],
                    [9, 10, 11]
                ]
            },
            {
                "vertices": [1, 2, 3],
                "triangles": [
                    [9, 10, 11],
                    [3, 11, 9], 
                    [0, 3, 9]
                ]
            },
            {
                "vertices": [0, 5],
                "triangles": [
                    [0, 3, 8],
                    [4, 5, 9]
                ]
            },
            {
                "vertices": [0, 1],
                "triangles":[
                    [3, 1, 8],
                    [8, 9, 1]
                ]
            },
            {
                "vertices": [0],
                "triangles":[
                    [3, 0, 8],
                ]
            },
            {
                "vertices": [],
                "triangles":[]
            },
        ]
        table = []
        for c in baseCubes:
            cube = Cube(self, 1.0, c["vertices"], c["triangles"])
            table.append(cube)
        return table

    def _write_info(self, text, coloring):
        # Create a Text object with the given text
        words = text.split()
        lines = []
        current_line = ""
        max_line_length = 90  # Maximum length of each line

        for word in words:
            if len(current_line) + len(word) + 1 <= max_line_length:
                current_line += word + " "
            else:
                lines.append(current_line.rstrip())
                current_line = word + " "

        if current_line:
            lines.append(current_line.rstrip())

        # Join lines with LaTeX line break
        final_text = "\n".join(lines)
        text_obj = Text(
            final_text,
            font="Arial", 
            font_size=30,
        )
        text_obj.set_color(BLACK)
        
        text_obj.move_to(DOWN*3.3 + LEFT*6, aligned_edge=LEFT)
        text_obj.fix_in_frame(True)
        for word, color in coloring.items():
            text_obj.set_color_by_text(word, color)

        self.text_obj = text_obj 
        # Add the text to the scene
        for letter in self.text_obj:
            self.play(FadeIn(letter), run_time=0.05)

    def _delete_info(self):
        # Remove the text from the scene
        self.play(FadeOut(self.text_obj))