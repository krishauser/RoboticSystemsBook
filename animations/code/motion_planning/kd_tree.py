from manimlib import *
import numpy as np

manim_config.camera.background_color = WHITE

class KDTreeNode:
    def __init__(self, name, dim=None, point=None, leaf=False):
        self.name = name
        if not leaf:
            self.dim = dim
            self.val = [point.local_x, point.local_y][dim]
            self.dot = point.dot
        self.leaf = leaf
        self.children = []
        self.graph_node = None
        self.edge = None
        self.areas = []

    def __repr__(self):
        return f"KDTreeNode({self.point})"

class Point():
    def __init__(self, x, y, axes):
        self.axes = axes
        self.local_x = x
        self.local_y = y
        self.dot = Dot(point=axes.c2p(x, y), fill_color=YELLOW, stroke_color=BLACK, stroke_width=3)
        self.dot.set_z_index(10)
        self.world_x = self.dot.get_center()[0]
        self.world_y = self.dot.get_center()[1]

class KDTree(Scene):
    def construct(self):
        self.text_obj = None
        self._title_screen("Create KD-Tree")
        self.points, self.axes = self._create_scene()
        kd_tree = self._create_kd_tree()
        self._title_screen("Find Nearest Neighbor using the KD-Tree")
        point = Point(5.9, 4.3, self.axes)
        self._find_nearest_neighbor(kd_tree, point)

    def _find_nearest_neighbor(self, kd_tree, point):
        # Create a dot for the point
        point.dot.set_color("#0000FF")
        # Create a dashed circle
        circle = Circle(radius=0.01, stroke_color="#0000FF", stroke_width=3)
        circle.move_to(point.dot.get_center())
        line = Line(
            start=point.dot.get_center(),
            end=point.dot.get_center(),
            color="#0000FF",
            stroke_width=3,
        )

        self.play(FadeIn(circle), FadeIn(line), FadeIn(point.dot), run_time=1)

        # KD search algorithm
        def dfs(node, nn):
            if node.leaf:
                self.play(
                    node.graph_node.animate.set_stroke(GREEN),
                )
                if node.name == "E-Left":
                    self._write_info("""
                    Now, we are at a leaf node, we iterate through the points in the node store the closest one (highlighted red). 
                    """, {"red": RED})
                    self._delete_info()
                for p in node.children:
                    # Calculate distance
                    distance = ((point.local_x - p.local_x)**2 + (point.local_y - p.local_y)**2)**0.5
                    if min(distance, nn[0]) == distance:
                        if nn[1] is not None:
                            # Reset the color of the previous nearest neighbor
                            self.play(
                                nn[1].dot.animate.set_fill(YELLOW, opacity=1),
                            )
                        nn[0] = distance
                        nn[1] = p
                        # Highlight the nearest neighbor and scale the circle
                        # Distance between point and p
                        actual_distance = ((p.dot.get_center()[0] - point.dot.get_center()[0])**2 + (p.dot.get_center()[1] - point.dot.get_center()[1])**2)**0.5
                        target_circle = Circle(radius=actual_distance, stroke_color="#0000FF", stroke_width=3).move_to(circle)

                        self.play(
                            Transform(circle, target_circle),
                            line.animate.put_start_and_end_on(
                                point.dot.get_center(),
                                p.dot.get_center(),
                            ),
                            nn[1].dot.animate.set_fill(RED),
                        )
                self.play(
                    node.graph_node.animate.set_stroke(BLACK),
                )
                return nn
    
            coord = [point.local_x, point.local_y]
            first_child_index = 0 if node.val > coord[node.dim] else 1
            second_child_index = 1 - first_child_index
            self.play(
                node.graph_node.animate.set_stroke(GREEN),
                node.edge.animate.set_stroke(GREEN),
            )
            # Choose the side of the tree to search
            if node.name == "A":
                self._write_info("""
                To search the KD-Tree, we start at the root node and check if the (blue) point is on the left or right side of the tree.
                If the point's value is less than the root's value in the x-direction, we go left. Otherwise, we go right. In this case, we go right.
                """, {"x-direction": BLUE})
                self._delete_info()
            elif node.name == "C":
                self._write_info("""
                    Now, if the point's value is less than the root's value in the y-direction, we go left. Otherwise, we go right. In this case, we go right.
                    """, {"y-direction": RED})
                self._delete_info()
            self.play(
                node.areas[first_child_index].animate.set_fill(opacity=0.5),
            )
            self.play(
                node.areas[first_child_index].animate.set_fill(opacity=0),
            )
            nn = dfs(node.children[first_child_index], nn)
            if node.name == "C":
                self._write_info("""
                C is not closer than the nearest neighbor found so far, so we don't need to check the other side of the tree at C. We continue back up the tree.
                """, {})
                self._delete_info()
            if nn[0] > abs(coord[node.dim] - node.val):
                if node.name == "E":
                    self._write_info("""
                    As we go back up the tree, we check if the distance to the current node's value is less than the distance to the nearest neighbor found so far.
                    If it is, we check the other side of the tree like for this E node. This is because the nearest neighbor could be on the other side.
                    """, {})
                    self._delete_info()
                self.play(
                    node.areas[second_child_index].animate.set_fill(opacity=0.5),
                )
                self.play(
                    node.areas[second_child_index].animate.set_fill(opacity=0),
                )
                nn = dfs(node.children[second_child_index], nn)
            self.play(
                node.graph_node.animate.set_stroke(BLACK),
                node.edge.animate.set_stroke(BLACK),
            )
            return nn
        dfs(kd_tree, [float("inf"), None])
        self._write_info("""
            The point highlighted in red is the nearest neighbor. 
            """, {"red": RED})

    def _create_scene(self):
        points = [
            (0.8, 2.8),
            (1.9, 5.9),
            (2.6, 3.4),
            (2.8, 5.2),
            (3.6, 6.4),
            (3.9, 3.2),
            (4.3, 0.3),
            (4.7, 3.8),
            (5.1, 1.9),
            (5.8, 3.4),
            (6.2, 3.0),
            (6.2, 3.8),
            (6.8, 1.6),
            (7.1, 5.3)
        ]
        points.sort()
        axes = Axes(
            x_range=[0, 8, 1],
            y_range=[0, 7, 1],
            axis_config={"color": BLUE},
            height=5,
            width=5,
        )
        axes.set_color(BLACK)
        axes.shift(LEFT*3.5)
        self.play(ShowCreation(axes), run_time=3.0)

        points_dots = [Point(p[0], p[1], axes) for p in points]
        self.play(*[FadeIn(point.dot) for point in points_dots], run_time=2.0)
        return points_dots, axes

    def _create_area_highlight(self, p1, p2, p3, p4):
        # Create a polygon to highlight the area
        area = Polygon(
            self.axes.c2p(p1[0], p1[1], 0.0),
            self.axes.c2p(p2[0], p2[1], 0.0),
            self.axes.c2p(p3[0], p3[1], 0.0),
            self.axes.c2p(p4[0], p4[1], 0.0),
            fill_color=RED,
            fill_opacity=0.0,
            stroke_width=0
        )
        self.add(area)
        return area

    def _create_kd_tree(self):
        # A
        self._write_info(
            """
            The first node of the KD-Tree has a value that is the median of the x values of the points. This divides the points into halves.
            The points with x values less than the median (left side) and greater than the median (right side) will be associated with 
            the left and right child of the node, respectively. The point at the median is associated with the right child.
            """, {"x": BLUE})
        a_p = self.points[5]
        a_edge = self._draw_line((a_p.local_x, self.axes.y_range[0]), (a_p.local_x, self.axes.y_range[1]), "A", LEFT*0.2)
        a_node = KDTreeNode("A", 0, a_p)
        a_graph_node = self._draw_graph_node((3, 3), "A", 0, a_p.local_x)
        a_node.graph_node = a_graph_node
        a_node.edge = a_edge
        # Left area
        a_node.areas.append(self._create_area_highlight(
            (self.axes.x_range[0], self.axes.y_range[0]),
            (self.axes.x_range[0], self.axes.y_range[1]),
            (a_p.local_x, self.axes.y_range[1]),
            (a_p.local_x, self.axes.y_range[0])
        ))
        # Right area
        a_node.areas.append(self._create_area_highlight(
            (a_p.local_x, self.axes.y_range[0]),
            (a_p.local_x, self.axes.y_range[1]),
            (self.axes.x_range[1], self.axes.y_range[1]),
            (self.axes.x_range[1], self.axes.y_range[0])
        ))
        self._delete_info()

        self._write_info(
            """
            In the next step, the median of the points in the next dimension, the y-dimension, for each side of A is used for the B and C nodes.
            """, {"y-dimension": RED})
        # B
        b_p = self.points[3]
        b_edge = self._draw_line((self.axes.x_range[0], b_p.local_y), (a_p.local_x, b_p.local_y), "B", DOWN*0.2)
        b_node = KDTreeNode("B", 1, b_p)
        b_node.edge = b_edge
        b_graph_node = self._draw_graph_node((1.7, 1.5), "B", 1, b_p.local_y)
        self._draw_arrow(a_graph_node, b_graph_node)
        b_node.graph_node = b_graph_node
        a_node.children.append(b_node)
        # Bottom area
        b_node.areas.append(self._create_area_highlight(
            (self.axes.x_range[0], self.axes.y_range[0]),
            (self.axes.x_range[0], b_p.local_y),
            (a_p.local_x, b_p.local_y),
            (a_p.local_x, self.axes.y_range[0])
        ))
        # Top area
        b_node.areas.append(self._create_area_highlight(
            (self.axes.x_range[0], b_p.local_y),
            (self.axes.x_range[0], self.axes.y_range[1]),
            (a_p.local_x, self.axes.y_range[1]),
            (a_p.local_x, b_p.local_y)
        ))

        # C
        c_p = self.points[9]
        c_edge = self._draw_line((a_p.local_x, c_p.local_y), (self.axes.x_range[1], c_p.local_y), "C", RIGHT + DOWN*0.2)
        c_node = KDTreeNode("C", 1, c_p)
        c_node.edge = c_edge
        c_graph_node = self._draw_graph_node((4.3, 1.5), "C", 1, c_p.local_y)
        self._draw_arrow(a_graph_node, c_graph_node)
        c_node.graph_node = c_graph_node
        a_node.children.append(c_node)
        # Bottom area
        c_node.areas.append(self._create_area_highlight(
            (a_p.local_x, self.axes.y_range[0]),
            (a_p.local_x, c_p.local_y),
            (self.axes.x_range[1], c_p.local_y),
            (self.axes.x_range[1], self.axes.y_range[0])
        ))
        # Top area
        c_node.areas.append(self._create_area_highlight(
            (a_p.local_x, c_p.local_y),
            (a_p.local_x, self.axes.y_range[1]),
            (self.axes.x_range[1], self.axes.y_range[1]),
            (self.axes.x_range[1], c_p.local_y)
        ))
        self._delete_info()

        self._write_info(
            """
            Now, for B, the next nodes are leaf nodes that CONTAIN the points because there are only 3 or less points on each side. 
            The same is not true for C, so it will continue adding nodes to split the points.
            """, {})
        # B leaf node left
        b_node_leaf_left = KDTreeNode("B-Left", leaf=True)
        square = Square(side_length=0.6, stroke_color=BLACK, fill_opacity=0.0, stroke_width=3)
        square.move_to(np.array([0.9, 0.0, 0.0]))
        self.play(ShowCreation(square), run_time=1)
        leaf_points = [
            self.points[0],
            self.points[2]
        ]
        b_node_leaf_left.children = leaf_points
        point_group = VGroup(*[point.dot.copy() for point in leaf_points])
        self.play(point_group.animate.move_to(square.get_center()).scale(0.3), run_time=1)
        self._draw_arrow(b_graph_node, square)
        b_node_leaf_left.graph_node = VGroup(square, point_group)
        b_node.children.append(b_node_leaf_left)

        # B leaf node right
        b_node_leaf_right = KDTreeNode("B-Right", leaf=True)
        square = Square(side_length=0.6, stroke_color=BLACK, fill_opacity=0.0, stroke_width=3)
        square.move_to(np.array([2.5, 0.0, 0.0]))
        self.play(ShowCreation(square), run_time=1)
        leaf_points = [
            self.points[1],
            self.points[3],
            self.points[4]
        ]
        b_node_leaf_right.children = leaf_points
        point_group = VGroup(*[point.dot.copy() for point in leaf_points])
        self.play(point_group.animate.move_to(square.get_center()).scale(0.3), run_time=1)
        self._draw_arrow(b_graph_node, square)
        b_node_leaf_right.graph_node = VGroup(square, point_group)
        b_node.children.append(b_node_leaf_right)
        self._delete_info()

        self._write_info(
            """
            To split C, we loop back to splitting amongst the x-dimension since we have already split the points in all the dimensions.
            """, {"x-dimension": BLUE})
        # D
        d_p = self.points[8]
        d_edge = self._draw_line((d_p.local_x, c_p.local_y), (d_p.local_x, self.axes.y_range[0]), "D", LEFT*0.2)
        d_node = KDTreeNode("D", 0, d_p)
        d_node.edge = d_edge
        d_graph_node = self._draw_graph_node((3.5, 0.0), "D", 0, d_p.local_x)
        self._draw_arrow(c_graph_node, d_graph_node)
        d_node.graph_node = d_graph_node
        c_node.children.append(d_node)
        # Left area
        d_node.areas.append(self._create_area_highlight(
            (a_p.local_x, self.axes.y_range[0]),
            (a_p.local_x, c_p.local_y),
            (d_p.local_x, c_p.local_y),
            (d_p.local_x, self.axes.y_range[0])
        ))
        # Right area
        d_node.areas.append(self._create_area_highlight(
            (d_p.local_x, self.axes.y_range[0]),
            (d_p.local_x, c_p.local_y),
            (self.axes.x_range[0], c_p.local_y),
            (self.axes.x_range[0], self.axes.y_range[0])
        ))
        
        # E
        e_p = self.points[11]
        e_edge = self._draw_line((e_p.local_x, c_p.local_y), (e_p.local_x, self.axes.y_range[1]), "E", LEFT*0.2)
        e_node = KDTreeNode("E", 0, e_p)
        e_node.edge = e_edge
        e_graph_node = self._draw_graph_node((5.1, 0.0), "E", 0, e_p.local_x)
        self._draw_arrow(c_graph_node, e_graph_node)
        e_node.graph_node = e_graph_node
        c_node.children.append(e_node)
        # Left area
        e_node.areas.append(self._create_area_highlight(
            (a_p.local_x, c_p.local_y),
            (a_p.local_x, self.axes.y_range[1]),
            (e_p.local_x, self.axes.y_range[1]),
            (e_p.local_x, c_p.local_y)
        ))
        # Right area
        e_node.areas.append(self._create_area_highlight(
            (e_p.local_x, c_p.local_y),
            (e_p.local_x, self.axes.y_range[1]),
            (self.axes.x_range[1], self.axes.y_range[1]),
            (self.axes.x_range[1], c_p.local_y)
        ))
        self._delete_info()

        self._write_info(
            """
            Now, for D and E, the next nodes are leaf nodes that CONTAIN the points on each side.
            """, {})
        # D leaf node left
        d_node_leaf_left = KDTreeNode("D-Left", leaf=True)
        square = Square(side_length=0.6, stroke_color=BLACK, fill_opacity=0.0, stroke_width=3)
        square.move_to(np.array([3.1, -1.5, 0.0]))
        self.play(ShowCreation(square), run_time=1)
        leaf_points = [
            self.points[5],
            self.points[6]
        ]
        d_node_leaf_left.children = leaf_points
        point_group = VGroup(*[point.dot.copy() for point in leaf_points])
        self.play(point_group.animate.move_to(square.get_center()).scale(0.2), run_time=1)
        self._draw_arrow(d_graph_node, square)
        d_node_leaf_left.graph_node = VGroup(square, point_group)
        d_node.children.append(d_node_leaf_left)

        # D leaf node right
        d_node_leaf_right = KDTreeNode("D-Right", leaf=True)
        square = Square(side_length=0.6, stroke_color=BLACK, fill_opacity=0.0, stroke_width=3)
        square.move_to(np.array([3.9, -1.5, 0.0]))
        self.play(ShowCreation(square), run_time=1)
        leaf_points = [
            self.points[8],
            self.points[10],
            self.points[12]
        ]
        d_node_leaf_right.children = leaf_points
        point_group = VGroup(*[point.dot.copy() for point in leaf_points])
        self.play(point_group.animate.move_to(square.get_center()).scale(0.3), run_time=1)
        self._draw_arrow(d_graph_node, square)
        d_node_leaf_right.graph_node = VGroup(square, point_group)
        d_node.children.append(d_node_leaf_right)

        # E leaf node left
        e_node_leaf_left = KDTreeNode("E-Left", leaf=True)
        square = Square(side_length=0.6, stroke_color=BLACK, fill_opacity=0.0, stroke_width=3)
        square.move_to(np.array([4.7, -1.5, 0.0]))
        self.play(ShowCreation(square), run_time=1)
        leaf_points = [
            self.points[7],
            self.points[9]
        ]
        e_node_leaf_left.children = leaf_points
        point_group = VGroup(*[point.dot.copy() for point in leaf_points])
        self.play(point_group.animate.move_to(square.get_center()).scale(0.3), run_time=1)
        self._draw_arrow(e_graph_node, square)
        e_node_leaf_left.graph_node = VGroup(square, point_group)
        e_node.children.append(e_node_leaf_left)

        # E leaf node right
        e_node_leaf_right = KDTreeNode("E-Right", leaf=True)
        square = Square(side_length=0.6, stroke_color=BLACK, fill_opacity=0.0, stroke_width=3)
        square.move_to(np.array([5.5, -1.5, 0.0]))
        self.play(ShowCreation(square), run_time=1)
        leaf_points = [
            self.points[11],
            self.points[13]
        ]
        e_node_leaf_right.children = leaf_points
        point_group = VGroup(*[point.dot.copy() for point in leaf_points])
        self.play(point_group.animate.move_to(square.get_center()).scale(0.3), run_time=1)
        self._draw_arrow(e_graph_node, square)
        e_node_leaf_right.graph_node = VGroup(square, point_group)
        e_node.children.append(e_node_leaf_right)
        self._delete_info()
        return a_node


    def _draw_arrow(self, n1, n2):
        arrow = Arrow(
            start=n1.get_bottom(),
            end=n2.get_top(),
            buff=0,              # No spacing at start/end
            stroke_width=1,
            fill_color=BLACK,
        )
        self.play(FadeIn(arrow), run_time=1.0)

    def _draw_graph_node(self, p, label, dim, value):
        graph_node = Dot(point=np.array([p[0], p[1], 0.0]), fill_color=WHITE, stroke_color=BLACK, stroke_width=3, radius=0.3)
        label = Text(label, font_size=24, fill_color=BLACK)
        label.move_to(graph_node.get_center())
        dim = "x" if dim == 0 else "y"
        line1 = Text(f"dim: {dim}", font_size=20, fill_color=BLACK)
        line1.set_color_by_text("x", BLUE)
        line1.set_color_by_text("y", RED)
        line2 = Text(f"value: {value}", font_size=20, fill_color=BLACK)
        # Stack them vertically with a small gap
        val_label = VGroup(line1, line2).arrange(DOWN, buff=0.2)
        val_label.move_to(graph_node.get_center())
        val_label.shift(RIGHT*0.8)
        self.play(FadeIn(graph_node), FadeIn(label), FadeIn(val_label), run_time=1.0)
        return graph_node
        
    def _draw_line(self, p1, p2, label, label_shift):
        line = Line(self.axes.c2p(p1[0], p1[1]), self.axes.c2p(p2[0], p2[1]), color=BLACK, stroke_width=3)
        # Add label to the line
        label = Text(label, font_size=24, fill_color=BLACK)
        label.move_to(line.get_center())
        label.shift(label_shift)
        self.play(ShowCreation(line), FadeIn(label), run_time=1.0)
        return VGroup(line, label)

    def _title_screen(self, text):
        # Create title text
        title = Text(text, font_size=72, fill_color=BLACK)
        white_screen = Rectangle(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            color=WHITE,
            fill_opacity=1,
            stroke_width=0
        )
        
        # Center it
        white_screen.move_to(ORIGIN)
        title.move_to(ORIGIN)
        white_screen.set_z_index(20)
        title.set_z_index(21)

        # Fade in
        self.play(FadeIn(white_screen), FadeIn(title), run_time=1)
        # Hold
        self.wait(1)
        # Fade out
        self.play(FadeOut(white_screen), FadeOut(title), run_time=1)

    def _write_info(self, text, coloring):
        # Create a Text object with the given text
        words = text.split()
        lines = []
        current_line = ""
        max_line_length = 100  # Maximum length of each line

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
            font_size=27,
        )
        text_obj.set_color(BLACK)
        text_obj.move_to(DOWN*3.3 + LEFT*6, aligned_edge=LEFT)
        for word, color in coloring.items():
            text_obj.set_color_by_text(word, color)

        # text_obj.arrange(DOWN, buff=0.5)  # buff controls spacing
        self.text_obj = text_obj 
        # Add the text to the scene
        for letter in self.text_obj:
            self.play(FadeIn(letter), run_time=0.05)

    def _delete_info(self):
        # Remove the text from the scene
        self.play(FadeOut(self.text_obj))